"""Re-anchor evaluation dataset chunk IDs to actual production chunks.

The combined_eval_dataset.json has fabricated ground_truth_chunk_id values
(valid UUIDs that match zero rows in production). This script finds the real
production chunk that best matches each query based on page, section, and
fault-code text signals.

Usage:
    PYTHONPATH="../raptor/backend:." python -m ingestion.reanchor_chunk_ids \
      --input datasets/combined_eval_dataset.json \
      --output datasets/combined_eval_dataset_anchored.json \
      --doc-map "AS-AMM-01-000:87,SC10000AMM:88"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Import external model + session (READ-ONLY)
# ---------------------------------------------------------------------------

def _import_model_and_session(db_url: str | None = None):
    """Import DocumentChunkMultimodal and a session factory from the main project."""
    if db_url:
        import os
        os.environ.setdefault("DATABASE_URL", db_url)

    def _do_import():
        from core.database import SessionLocal
        from core.models import DocumentChunkMultimodal
        return DocumentChunkMultimodal, SessionLocal

    try:
        return _do_import()
    except (ImportError, ValueError):
        backend_path = str(Path(__file__).resolve().parent.parent.parent / "raptor" / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
            try:
                return _do_import()
            except (ImportError, ValueError) as exc:
                print(f"ERROR: Could not import from core.models.\n  Error: {exc}", file=sys.stderr)
                sys.exit(1)
        raise


# ---------------------------------------------------------------------------
# Page reference parsing
# ---------------------------------------------------------------------------

def parse_page_reference(page_ref: str) -> dict:
    """Parse page_reference into structured components.

    Formats:
      "109"           → absolute page 109
      "6.3.10 p2"    → section 6.3.10, relative page 2
      "77-5"         → chapter 77, page 5
      "28-5 to 28-6" → chapter 28, pages 5-6
    """
    result = {"raw": page_ref, "pages": [], "section_prefix": None}

    # Plain numeric
    if page_ref.strip().isdigit():
        result["pages"] = [int(page_ref.strip())]
        return result

    # "section p#" format (e.g., "6.3.10 p2")
    m = re.match(r"([\d.]+)\s+p(\d+)", page_ref)
    if m:
        result["section_prefix"] = m.group(1)
        result["pages"] = [int(m.group(2))]
        return result

    # Range format "CH-P1 to CH-P2" (e.g., "28-5 to 28-6")
    m = re.match(r"(\d+)-(\d+)\s+to\s+\d+-(\d+)", page_ref)
    if m:
        result["section_prefix"] = m.group(1)
        p1, p2 = int(m.group(2)), int(m.group(3))
        result["pages"] = list(range(p1, p2 + 1))
        return result

    # "chapter-page" format (e.g., "77-5")
    m = re.match(r"(\d+)-(\d+)$", page_ref.strip())
    if m:
        result["section_prefix"] = m.group(1)
        result["pages"] = [int(m.group(2))]
        return result

    # Fallback: extract all integers
    nums = re.findall(r"\d+", page_ref)
    if nums:
        result["pages"] = [int(n) for n in nums]

    return result


# ---------------------------------------------------------------------------
# Chunk page/section filtering
# ---------------------------------------------------------------------------

def chunk_contains_page(chunk: dict, target_pages: list[int]) -> bool:
    """Check if a chunk's page fields cover any of the target pages (absolute)."""
    ps = chunk.get("page_start")
    pe = chunk.get("page_end")
    if ps is not None and pe is not None:
        for p in target_pages:
            if ps <= p <= pe:
                return True

    # 'pages' is a comma-separated string of page numbers
    pages_str = chunk.get("pages") or ""
    if pages_str:
        page_nums_in_chunk = set(int(n) for n in re.findall(r"\d+", pages_str))
        if page_nums_in_chunk & set(target_pages):
            return True

    return False


def chunk_near_page(chunk: dict, target_pages: list[int], tolerance: int = 1) -> bool:
    """Check if chunk is within ±tolerance of target pages (absolute)."""
    expanded = set()
    for p in target_pages:
        for offset in range(-tolerance, tolerance + 1):
            expanded.add(p + offset)

    ps = chunk.get("page_start")
    pe = chunk.get("page_end")
    if ps is not None and pe is not None:
        chunk_pages = set(range(ps, pe + 1))
        if chunk_pages & expanded:
            return True

    pages_str = chunk.get("pages") or ""
    if pages_str:
        page_nums_in_chunk = set(int(n) for n in re.findall(r"\d+", pages_str))
        if page_nums_in_chunk & expanded:
            return True

    return False


def chunk_matches_section(chunk: dict, section_prefix: str) -> bool:
    """Check if chunk belongs to a given section/chapter.

    For SC10000AMM: section_prefix is like "6.3.10" → matches section_code
    For AS-AMM-01-000: section_prefix is like "77" → matches section_header "77-XX"
    """
    chunk_code = chunk.get("section_code") or ""
    chunk_header = chunk.get("section_header") or ""

    # Exact or prefix match on section_code (e.g., "6.3.10" matches "6.3.10")
    if chunk_code == section_prefix or chunk_code.startswith(section_prefix + "."):
        return True

    # ATA chapter match: section_header starts with "XX-" (e.g., "77-00 GENERAL")
    if re.match(rf"^{re.escape(section_prefix)}-", chunk_header):
        return True

    # section_prefix appears in section_code as a component
    if section_prefix in chunk_code:
        return True

    return False


def get_section_page_offset(chunk: dict, section_chunks: list[dict], relative_page: int) -> bool:
    """Check if chunk is on the Nth page within a section's chunks.

    section_chunks should be sorted by page_start. relative_page is 1-based.
    """
    # Get distinct pages in this section
    section_pages = sorted(set(
        c["page_start"] for c in section_chunks
        if c.get("page_start") is not None
    ))
    if not section_pages:
        return False

    # Convert 1-based relative page to absolute page
    idx = relative_page - 1
    if 0 <= idx < len(section_pages):
        target_abs_page = section_pages[idx]
        ps = chunk.get("page_start")
        pe = chunk.get("page_end") or ps
        if ps is not None and ps <= target_abs_page <= (pe or ps):
            return True
        # Also check with ±1 tolerance for edge cases
        if ps is not None and abs(ps - target_abs_page) <= 1:
            return True

    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def extract_section_number(section_system: str) -> str | None:
    """Extract section/chapter number from section_system field.

    Examples:
      "Engine (6.3.10)" → "6.3.10"
      "Lights (Ch 33)"  → "33"
      "Fuel (Ch 28)"    → "28"
    """
    # Try parenthesized section first
    m = re.search(r"\((Ch\s*)?(\d[\d.]*)\)", section_system)
    if m:
        return m.group(2)
    # Fallback: any dotted or plain number
    m = re.search(r"\d[\d.]*", section_system)
    return m.group() if m else None


def match_score(chunk: dict, fault_code: str, section_system: str) -> float:
    """Score how well a chunk matches the expected fault code and section."""
    score = 0.0
    fault_lower = fault_code.lower().strip()
    text_lower = (chunk.get("text") or "").lower()

    # Signal 1: Exact substring match of fault code in chunk text
    if fault_lower in text_lower:
        score += 100.0
    else:
        # Try without parenthetical content: "Too much vibration (exhaust)" → "Too much vibration"
        fault_stripped = re.sub(r"\s*\([^)]*\)\s*", " ", fault_lower).strip()
        if fault_stripped != fault_lower and fault_stripped in text_lower:
            score += 90.0

    # Signal 2: Word overlap ratio
    fault_words = set(re.findall(r"\w+", fault_lower))
    text_words = set(re.findall(r"\w+", text_lower))
    if fault_words:
        overlap = len(fault_words & text_words) / len(fault_words)
        score += overlap * 50.0

    # Signal 3: Section code match
    sec_code = extract_section_number(section_system)
    if sec_code:
        chunk_section = str(chunk.get("section_code") or "")
        chunk_header = str(chunk.get("section_header") or "")
        if sec_code in chunk_section:
            score += 30.0
        if sec_code in chunk_header:
            score += 20.0

    # Signal 4: Leaf node (level 0) preferred over RAPTOR summaries
    if chunk.get("level", 0) == 0:
        score += 10.0

    return score


# ---------------------------------------------------------------------------
# Matching pipeline
# ---------------------------------------------------------------------------

def find_best_match(query: dict, doc_chunks: list[dict]) -> dict:
    """Run the full matching cascade for a single query.

    Strategy: fault_code text match is the strongest signal, so we always
    check it first. Section/page filtering is used for disambiguation when
    multiple chunks contain the fault code.
    """
    fault_code = query["ground_truth_fault_code"]
    section_system = query["section_system"]
    page_ref = query["page_reference"]

    parsed = parse_page_reference(page_ref)
    target_pages = parsed["pages"]
    section_prefix = parsed["section_prefix"]
    sec_code = extract_section_number(section_system)

    result = {
        "original_fake_id": query["ground_truth_chunk_id"],
        "matched_chunk_id": None,
        "match_score": 0.0,
        "confidence": "none",
        "match_method": None,
        "runner_up_score": 0.0,
        "runner_up_chunk_id": None,
        "page_candidates_count": 0,
        "fallback_used": None,
    }

    # Step 1: Find all chunks containing the fault code as substring (strongest signal)
    fault_lower = fault_code.lower().strip()
    text_matches = [c for c in doc_chunks if fault_lower in (c.get("text") or "").lower()]

    # Try without parenthetical content if no matches
    if not text_matches and fault_lower != "none":
        fault_stripped = re.sub(r"\s*\([^)]*\)\s*", " ", fault_lower).strip()
        if fault_stripped != fault_lower:
            text_matches = [c for c in doc_chunks if fault_stripped in (c.get("text") or "").lower()]

    candidates = []
    match_method = None

    if text_matches:
        # If we have text matches, try to narrow by section/page for disambiguation
        # Try section-filtered subset of text matches
        section_filtered = []
        if section_prefix:
            section_filtered = [c for c in text_matches if chunk_matches_section(c, section_prefix)]
        if not section_filtered and sec_code:
            section_filtered = [c for c in text_matches if chunk_matches_section(c, sec_code)]

        if section_filtered:
            candidates = section_filtered
            match_method = "text+section"
        else:
            # Try page-filtered subset
            if target_pages and not section_prefix:
                page_filtered = [c for c in text_matches if chunk_contains_page(c, target_pages)]
                if page_filtered:
                    candidates = page_filtered
                    match_method = "text+page"

            if not candidates:
                candidates = text_matches
                match_method = "text_only"
    else:
        # No exact text match — fall back to section/page candidates with word scoring
        if section_prefix:
            candidates = [c for c in doc_chunks if chunk_matches_section(c, section_prefix)]
            if candidates:
                match_method = "section_only"

        if not candidates and sec_code:
            candidates = [c for c in doc_chunks if chunk_matches_section(c, sec_code)]
            if candidates:
                match_method = "section_system_only"

        if not candidates and target_pages and not section_prefix:
            candidates = [c for c in doc_chunks if chunk_contains_page(c, target_pages)]
            if candidates:
                match_method = "page_only"
            else:
                candidates = [c for c in doc_chunks if chunk_near_page(c, target_pages, tolerance=2)]
                if candidates:
                    match_method = "page_tolerance"

        if not candidates:
            # Last resort: score all chunks in document
            candidates = doc_chunks
            match_method = "full_scan"

    fallback_used = match_method if match_method not in ("text+section", "text+page", "text_only") else None

    result["page_candidates_count"] = len(candidates)
    result["fallback_used"] = fallback_used

    if not candidates:
        return result

    # Score all candidates
    scored = [(c, match_score(c, fault_code, section_system)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_chunk, best_score = scored[0]
    runner_up_score = scored[1][1] if len(scored) > 1 else 0.0
    runner_up_id = scored[1][0]["id"] if len(scored) > 1 else None

    # Use match_method from the cascade above
    method = match_method

    # Confidence classification
    if best_score >= 100:
        confidence = "high"
    elif best_score >= 50:
        confidence = "medium"
    elif fault_lower == "none" and method in ("page_only", "section_only", "section_system_only"):
        # For queries without a fault code, page/section match is the best signal
        confidence = "medium"
    elif best_score > 0:
        confidence = "low"
    else:
        confidence = "none"

    result.update({
        "matched_chunk_id": best_chunk["id"],
        "match_score": round(best_score, 1),
        "confidence": confidence,
        "match_method": method,
        "runner_up_score": round(runner_up_score, 1),
        "runner_up_chunk_id": runner_up_id,
    })

    return result


# ---------------------------------------------------------------------------
# Load chunks from DB
# ---------------------------------------------------------------------------

def load_chunks_by_document(SessionLocal, Model, doc_ids: list[int]) -> dict[int, list[dict]]:
    """Load all chunks for specified documents, grouped by document_id."""
    from sqlalchemy import inspect as sa_inspect

    mapper = sa_inspect(Model)
    columns = [c.key for c in mapper.column_attrs]
    # Skip heavy columns not needed for matching
    skip_cols = {"embedding", "tsv", "images"}
    use_cols = [c for c in columns if c not in skip_cols]

    chunks_by_doc: dict[int, list[dict]] = defaultdict(list)

    with SessionLocal() as db:
        for doc_id in doc_ids:
            rows = db.query(Model).filter(
                Model.document_id == doc_id,
                Model.blacklisted == False,  # noqa: E712
            ).all()

            for row in rows:
                node = {}
                for col in use_cols:
                    val = getattr(row, col)
                    if isinstance(val, (int, float, str, bool)) or val is None:
                        node[col] = val
                    elif hasattr(val, "value"):
                        node[col] = val.value
                    else:
                        node[col] = str(val)
                chunks_by_doc[doc_id].append(node)

    return chunks_by_doc


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(queries: list[dict], results: list[dict]):
    """Print the confidence report to console."""
    total = len(results)
    by_confidence = Counter(r["confidence"] for r in results)

    print("\nRe-anchoring Report")
    print("=" * 50)
    print(f"Total queries:     {total}")
    print(f"  High confidence:   {by_confidence.get('high', 0):>3}  (exact fault code match in page-filtered chunk)")
    print(f"  Medium confidence: {by_confidence.get('medium', 0):>3}  (strong word overlap)")
    print(f"  Low confidence:    {by_confidence.get('low', 0):>3}  (weak match — REVIEW THESE)")
    print(f"  No match:          {by_confidence.get('none', 0):>3}  (FAILED — no candidate found)")

    # By origin
    print("\nBy origin:")
    origin_conf = defaultdict(Counter)
    for q, r in zip(queries, results):
        origin_conf[q["origin"]][r["confidence"]] += 1
    for origin in sorted(origin_conf):
        c = origin_conf[origin]
        print(f"  {origin:<7} {c.get('high',0)} high, {c.get('medium',0)} medium, "
              f"{c.get('low',0)} low, {c.get('none',0)} none")

    # By manual_source
    print("\nBy manual_source:")
    manual_conf = defaultdict(Counter)
    for q, r in zip(queries, results):
        manual_conf[q["manual_source"]][r["confidence"]] += 1
    for manual in sorted(manual_conf):
        c = manual_conf[manual]
        print(f"  {manual:<14} {c.get('high',0)} high, {c.get('medium',0)} medium, "
              f"{c.get('low',0)} low, {c.get('none',0)} none")

    # Failed queries
    failed = [(q, r) for q, r in zip(queries, results) if r["confidence"] == "none"]
    if failed:
        print(f"\nFailed queries ({len(failed)}):")
        for q, r in failed[:20]:
            print(f"  {q['id']}: page_reference={q['page_reference']}, "
                  f"fault_code=\"{q['ground_truth_fault_code'][:50]}\", "
                  f"section=\"{q['section_system']}\" — {r['page_candidates_count']} candidates")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    # Low confidence queries
    low = [(q, r) for q, r in zip(queries, results) if r["confidence"] == "low"]
    if low:
        print(f"\nLow confidence queries ({len(low)}, showing first 10):")
        for q, r in low[:10]:
            print(f"  {q['id']}: score={r['match_score']}, method={r['match_method']}")


def print_spot_check(queries: list[dict], results: list[dict], chunks_by_doc: dict, manual_to_doc: dict):
    """Print 5 random high-confidence matches for manual verification."""
    random.seed(42)

    high = [(q, r) for q, r in zip(queries, results) if r["confidence"] == "high"]
    if not high:
        print("\nNo high-confidence matches to spot-check.")
        return

    samples = random.sample(high, min(5, len(high)))

    # Build chunk lookup
    chunk_lookup = {}
    for doc_id, chunks in chunks_by_doc.items():
        for c in chunks:
            chunk_lookup[c["id"]] = c

    print("\nSpot-check samples:")
    print("-" * 50)
    for q, r in samples:
        chunk = chunk_lookup.get(r["matched_chunk_id"], {})
        text_preview = (chunk.get("text") or "")[:120].replace("\n", " ")
        fault_in_text = q["ground_truth_fault_code"].lower() in (chunk.get("text") or "").lower()

        print(f"\n{q['id']}:")
        print(f"  Fault code: \"{q['ground_truth_fault_code']}\"")
        print(f"  Page ref:   {q['page_reference']}")
        print(f"  Matched chunk (score={r['match_score']}, confidence={r['confidence']}):")
        print(f"    ID: {r['matched_chunk_id']}")
        print(f"    Level: {chunk.get('level')}")
        print(f"    Pages: {chunk.get('pages')} (start={chunk.get('page_start')}, end={chunk.get('page_end')})")
        print(f"    Section: code={chunk.get('section_code')}, header={(chunk.get('section_header') or '')[:60]}")
        print(f"    Text preview: \"{text_preview}...\"")
        mark = "\u2713" if fault_in_text else "\u2717"
        print(f"  {mark} Fault code {'appears' if fault_in_text else 'NOT FOUND'} verbatim in chunk text")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-anchor eval dataset chunk IDs to production.")
    parser.add_argument("--input", required=True, help="Path to combined_eval_dataset.json")
    parser.add_argument("--output", required=True, help="Output path for anchored dataset")
    parser.add_argument("--doc-map", required=True,
                        help="manual_source:doc_id pairs, comma-separated (e.g., 'AS-AMM-01-000:87,SC10000AMM:88')")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    args = parser.parse_args()

    # Parse doc-map
    manual_to_doc = {}
    for pair in args.doc_map.split(","):
        name, doc_id = pair.strip().split(":")
        manual_to_doc[name.strip()] = int(doc_id.strip())

    print(f"Document mapping: {manual_to_doc}")

    # Load eval dataset
    input_path = Path(args.input)
    with open(input_path) as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries from {input_path}")

    # Validate all manual_source values have a mapping
    missing = set(q["manual_source"] for q in queries) - set(manual_to_doc.keys())
    if missing:
        print(f"ERROR: No doc-map entry for manual_source(s): {missing}", file=sys.stderr)
        sys.exit(1)

    # Connect to DB and load chunks
    Model, SessionLocal = _import_model_and_session(args.db_url)
    doc_ids = list(set(manual_to_doc.values()))
    print(f"Loading chunks for document_ids: {doc_ids} ...")
    chunks_by_doc = load_chunks_by_document(SessionLocal, Model, doc_ids)

    for doc_id, chunks in chunks_by_doc.items():
        print(f"  Document {doc_id}: {len(chunks)} chunks loaded")

    if not any(chunks_by_doc.values()):
        print("ERROR: No chunks loaded from database. Check doc-map and database connection.", file=sys.stderr)
        sys.exit(1)

    # Run matching
    print("\nMatching queries to production chunks...")
    results = []
    for query in queries:
        doc_id = manual_to_doc[query["manual_source"]]
        doc_chunks = chunks_by_doc[doc_id]
        result = find_best_match(query, doc_chunks)
        results.append(result)

    # Build output dataset
    output_queries = []
    for query, result in zip(queries, results):
        out = dict(query)
        if result["matched_chunk_id"]:
            out["ground_truth_chunk_id"] = result["matched_chunk_id"]
        # else: keep original fake ID (flagged as confidence=none)
        out["_reanchor_metadata"] = result
        output_queries.append(out)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_queries, f, indent=2)
    print(f"\nWrote anchored dataset to {output_path}")

    # Print report
    print_report(queries, results)
    print_spot_check(queries, results, chunks_by_doc, manual_to_doc)


if __name__ == "__main__":
    main()
