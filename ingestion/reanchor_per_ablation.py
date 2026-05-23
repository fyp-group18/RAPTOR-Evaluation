"""Re-anchor evaluation dataset chunk IDs per ablation tree.

Unlike reanchor_chunk_ids.py (which reads from the production DB), this script
reads chunks directly from each ablation tree's pickle file. This ensures that
ground_truth_chunk_id values match the tree the evaluator will actually search.

Trees are loaded from GCS via resolve_tree (with local .tree-cache mirror),
not from hardcoded local paths.

Matching strategy (same signals as original reanchor):
  1. Fault-code exact substring match (strongest signal, +100)
  2. Word overlap between fault code and chunk text (+0-50)
  3. Section code/header match (+20-30)
  4. Leaf node preference (+10)
  5. Page proximity as disambiguation

For each ablation, writes: combined_eval_dataset_anchored_{ablation_key}.json

Usage:
    python -m ingestion.reanchor_per_ablation \
        --input datasets/combined_eval_dataset.json \
        --output-dir datasets/ \
        --cache-keys "SC10000AMM:c313...,AS-AMM-01-000:112e..." \
        --gcs-bucket raptor-assets
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ingestion.tree_resolver import resolve_tree

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

ABLATION_KEY_TO_LABEL = {
    "full": "full_context_aware",
    "no_table_pc": "no_table_parent_child",
    "no_header_prop": "no_header_propagation",
    "no_caption_fold": "no_caption_folding",
    "no_context_aware": "baseline_naive_chunking",
    "semantic_chunking": "semantic_chunking_baseline",
    "flat_retrieval": "flat_no_raptor",
    "text_only_raptor": "original_raptor_text_only",
    "text_context_aware": "text_context_aware",
    "contextual_retrieval": "contextual_retrieval",
}


# ---------------------------------------------------------------------------
# Tree loading (via GCS resolve_tree)
# ---------------------------------------------------------------------------

def load_tree_chunks_from_gcs(
    cache_key: str,
    ablation_label: str,
    bucket: str,
    run_id: str | None = None,
) -> list[dict]:
    """Load nodes from a tree via GCS resolve_tree.

    Returns all nodes (not just leaves) — parent chunks contain table text
    that may be the best match for fault codes.
    """
    tree_bytes, manifest = resolve_tree(
        cache_key=cache_key,
        ablation_label=ablation_label,
        run_id=run_id,
        bucket=bucket,
    )
    nodes = pickle.loads(tree_bytes)
    resolved_run = manifest.get("run_id", "?")
    logger.info(f"  Loaded tree from GCS: run_id={resolved_run}, {len(nodes)} nodes")
    return nodes


# ---------------------------------------------------------------------------
# Matching logic (adapted from reanchor_chunk_ids.py)
# ---------------------------------------------------------------------------

def _extract_section_number(section_system: str) -> str | None:
    m = re.search(r"\((Ch\s*)?(\d[\d.]*)\)", section_system)
    if m:
        return m.group(2)
    m = re.search(r"\d[\d.]*", section_system)
    return m.group() if m else None


def _chunk_contains_page(chunk: dict, target_pages: list[int]) -> bool:
    ps = chunk.get("page_start")
    pe = chunk.get("page_end")
    if ps is not None and pe is not None and ps > 0:
        for p in target_pages:
            if ps <= p <= pe:
                return True

    pages = chunk.get("pages") or []
    if isinstance(pages, list) and pages:
        if set(pages) & set(target_pages):
            return True

    return False


def _chunk_matches_section(chunk: dict, section_prefix: str) -> bool:
    chunk_header = chunk.get("section_header") or ""

    # Direct match in section_header
    if section_prefix in chunk_header:
        return True

    # Check if section prefix appears in the chunk text header line
    text = chunk.get("text", "")
    # Context-aware chunks start with "[SECTION | Page X]"
    m = re.match(r"^\[([^|]+)", text)
    if m:
        header_in_text = m.group(1).strip()
        if section_prefix in header_in_text:
            return True

    return False


def _parse_page_reference(page_ref: str) -> tuple[list[int], str | None]:
    """Parse page_reference into (absolute_pages, section_prefix)."""
    if not page_ref:
        return [], None

    # Plain numeric
    if page_ref.strip().isdigit():
        return [int(page_ref.strip())], None

    # "section p#" format (e.g., "6.3.10 p2")
    m = re.match(r"([\d.]+)\s+p(\d+)", page_ref)
    if m:
        return [], m.group(1)  # relative page, need section context

    # Range format "CH-P1 to CH-P2"
    m = re.match(r"(\d+)-(\d+)\s+to\s+\d+-(\d+)", page_ref)
    if m:
        return [], m.group(1)

    # "chapter-page" format (e.g., "77-5")
    m = re.match(r"(\d+)-(\d+)$", page_ref.strip())
    if m:
        return [], m.group(1)

    # Fallback
    nums = re.findall(r"\d+", page_ref)
    return [int(n) for n in nums] if nums else ([], None)


def match_score(chunk: dict, fault_code: str, section_system: str) -> float:
    score = 0.0
    fault_lower = fault_code.lower().strip()
    text_lower = (chunk.get("text") or "").lower()

    # Signal 1: Exact substring match
    if fault_lower and fault_lower != "none" and fault_lower in text_lower:
        score += 100.0
    elif fault_lower and fault_lower != "none":
        fault_stripped = re.sub(r"\s*\([^)]*\)\s*", " ", fault_lower).strip()
        if fault_stripped != fault_lower and fault_stripped in text_lower:
            score += 90.0

    # Signal 2: Word overlap ratio
    if fault_lower and fault_lower != "none":
        fault_words = set(re.findall(r"\w+", fault_lower))
        text_words = set(re.findall(r"\w+", text_lower))
        if fault_words:
            overlap = len(fault_words & text_words) / len(fault_words)
            score += overlap * 50.0

    # Signal 3: Section match
    sec_code = _extract_section_number(section_system)
    if sec_code:
        chunk_header = str(chunk.get("section_header") or "")
        chunk_text_start = (chunk.get("text") or "")[:200]
        if sec_code in chunk_header:
            score += 30.0
        elif sec_code in chunk_text_start:
            score += 20.0

    # Signal 4: Leaf node preferred — but only prefer SEARCHABLE leaves
    # (is_parent chunks aren't searchable, so matching to them is unhelpful)
    if chunk.get("level", 0) == 0 and not chunk.get("is_parent"):
        score += 10.0
    elif chunk.get("is_parent"):
        score -= 5.0  # Penalize parent chunks (not in searchable index)

    return score


def find_best_match(query: dict, tree_chunks: list[dict]) -> dict:
    """Match a query to the best chunk in a tree, preferring searchable nodes."""
    fault_code = query.get("ground_truth_fault_code", "")
    section_system = query.get("section_system", "")
    page_ref = query.get("page_reference", "")

    target_pages, section_prefix = _parse_page_reference(page_ref)
    sec_code = _extract_section_number(section_system)

    result = {
        "original_id": query.get("ground_truth_chunk_id"),
        "matched_chunk_id": None,
        "match_score": 0.0,
        "confidence": "none",
        "match_method": None,
        "candidates_count": 0,
    }

    fault_lower = (fault_code or "").lower().strip()

    # Step 1: Text matches (fault code substring)
    text_matches = []
    if fault_lower and fault_lower != "none":
        text_matches = [
            c for c in tree_chunks
            if fault_lower in (c.get("text") or "").lower()
        ]
        if not text_matches:
            fault_stripped = re.sub(r"\s*\([^)]*\)\s*", " ", fault_lower).strip()
            if fault_stripped != fault_lower:
                text_matches = [
                    c for c in tree_chunks
                    if fault_stripped in (c.get("text") or "").lower()
                ]

    candidates = []
    match_method = None

    if text_matches:
        # Narrow by section
        if section_prefix:
            section_filtered = [c for c in text_matches if _chunk_matches_section(c, section_prefix)]
            if section_filtered:
                candidates = section_filtered
                match_method = "text+section"
        if not candidates and sec_code:
            section_filtered = [c for c in text_matches if _chunk_matches_section(c, sec_code)]
            if section_filtered:
                candidates = section_filtered
                match_method = "text+section"
        if not candidates and target_pages:
            page_filtered = [c for c in text_matches if _chunk_contains_page(c, target_pages)]
            if page_filtered:
                candidates = page_filtered
                match_method = "text+page"
        if not candidates:
            candidates = text_matches
            match_method = "text_only"
    else:
        # No text match — fall back to page/section
        if section_prefix:
            candidates = [c for c in tree_chunks if _chunk_matches_section(c, section_prefix)]
            if candidates:
                match_method = "section_only"
        if not candidates and sec_code:
            candidates = [c for c in tree_chunks if _chunk_matches_section(c, sec_code)]
            if candidates:
                match_method = "section_only"
        if not candidates and target_pages:
            candidates = [c for c in tree_chunks if _chunk_contains_page(c, target_pages)]
            if candidates:
                match_method = "page_only"
        if not candidates:
            candidates = tree_chunks
            match_method = "full_scan"

    result["candidates_count"] = len(candidates)

    if not candidates:
        return result

    # Score and rank
    scored = [(c, match_score(c, fault_code, section_system)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_chunk, best_score = scored[0]

    # Confidence
    if best_score >= 100:
        confidence = "high"
    elif best_score >= 50:
        confidence = "medium"
    elif fault_lower in ("none", "") and match_method in ("page_only", "section_only"):
        confidence = "medium"
    elif best_score > 0:
        confidence = "low"
    else:
        confidence = "none"

    result.update({
        "matched_chunk_id": best_chunk.get("id"),
        "match_score": round(best_score, 1),
        "confidence": confidence,
        "match_method": match_method,
    })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reanchor_for_ablation(
    queries: list[dict],
    manual_to_cache_key: dict[str, str],
    ablation_key: str,
    ablation_label: str,
    bucket: str = "raptor-assets",
    run_id: str | None = None,
) -> tuple[list[dict], dict]:
    """Run reanchoring for a single ablation.

    Returns (anchored_queries, stats_dict).
    """
    # Load tree chunks per manual from GCS via resolve_tree
    chunks_by_manual: dict[str, list[dict]] = {}
    for manual, cache_key in manual_to_cache_key.items():
        try:
            chunks_by_manual[manual] = load_tree_chunks_from_gcs(
                cache_key, ablation_label, bucket=bucket, run_id=run_id,
            )
        except FileNotFoundError:
            logger.warning(f"No tree found for {manual}/{ablation_label}")
            chunks_by_manual[manual] = []

    # Match each query
    output_queries = []
    confidence_counts = Counter()

    for query in queries:
        manual = query["manual_source"]
        tree_chunks = chunks_by_manual.get(manual, [])

        if not tree_chunks:
            out = dict(query)
            out["_reanchor_metadata"] = {
                "matched_chunk_id": None,
                "match_score": 0,
                "confidence": "none",
                "match_method": "no_tree",
                "candidates_count": 0,
            }
            output_queries.append(out)
            confidence_counts["none"] += 1
            continue

        result = find_best_match(query, tree_chunks)

        out = dict(query)
        if result["matched_chunk_id"]:
            out["ground_truth_chunk_id"] = result["matched_chunk_id"]
        out["_reanchor_metadata"] = result
        output_queries.append(out)
        confidence_counts[result["confidence"]] += 1

    stats = {
        "ablation": ablation_key,
        "label": ablation_label,
        "total": len(queries),
        "confidence": dict(confidence_counts),
    }
    return output_queries, stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Re-anchor eval dataset per ablation tree"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to combined_eval_dataset.json (raw, un-anchored)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for per-ablation anchored datasets",
    )
    parser.add_argument(
        "--cache-keys", required=True,
        help="manual_source:cache_key pairs (e.g., 'SC10000AMM:c313...,AS-AMM-01-000:112e...')",
    )
    parser.add_argument(
        "--ablations", default=None,
        help="Comma-separated ablation keys to process (default: all)",
    )
    parser.add_argument(
        "--gcs-bucket", default="raptor-assets",
        help="GCS bucket containing tree artifacts (default: raptor-assets)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Pin to a specific run_id (default: latest per ablation)",
    )
    args = parser.parse_args()

    # Parse cache keys
    manual_to_cache_key = {}
    for pair in args.cache_keys.split(","):
        parts = pair.strip().split(":")
        if len(parts) != 2:
            logger.error(f"Invalid cache-key pair: {pair}")
            sys.exit(1)
        manual_to_cache_key[parts[0].strip()] = parts[1].strip()

    logger.info(f"Manual→cache_key: {manual_to_cache_key}")

    # Load input dataset
    input_path = Path(args.input)
    with open(input_path) as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries from {input_path}")

    # Determine which ablations to process
    if args.ablations:
        ablation_keys = [k.strip() for k in args.ablations.split(",")]
    else:
        ablation_keys = list(ABLATION_KEY_TO_LABEL.keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for ablation_key in ablation_keys:
        ablation_label = ABLATION_KEY_TO_LABEL.get(ablation_key)
        if not ablation_label:
            logger.warning(f"Unknown ablation key: {ablation_key}, skipping")
            continue

        logger.info(f"\nProcessing: {ablation_key} ({ablation_label})")
        anchored, stats = reanchor_for_ablation(
            queries, manual_to_cache_key, ablation_key, ablation_label,
            bucket=args.gcs_bucket, run_id=args.run_id,
        )

        # Write output
        out_path = output_dir / f"combined_eval_dataset_anchored_{ablation_key}.json"
        with open(out_path, "w") as f:
            json.dump(anchored, f, indent=2)
        logger.info(f"  Wrote {out_path}")
        logger.info(f"  Confidence: {stats['confidence']}")
        all_stats.append(stats)

    # Summary report
    print("\n" + "=" * 70)
    print("Per-ablation reanchoring summary")
    print("=" * 70)
    for stats in all_stats:
        conf = stats["confidence"]
        print(f"  {stats['label']:<30s}: "
              f"high={conf.get('high', 0):>3}, "
              f"medium={conf.get('medium', 0):>3}, "
              f"low={conf.get('low', 0):>3}, "
              f"none={conf.get('none', 0):>3}")


if __name__ == "__main__":
    main()
