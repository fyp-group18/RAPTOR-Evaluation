"""CR Context Quality Audit.

Classifies all LLM-generated CR contexts as:
  A - Correct and specific
  B - Correct but vague
  C - Hallucinated (wrong attribution)
  D - Hallucinated (fabricated details)

Uses Gemini Flash to classify each chunk against ground-truth section headers
derived from Docling document structure.

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
    GOOGLE_CLOUD_PROJECT=<project> \
    python -m audit.cr_context_audit
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TREE_CACHE_DIR = PROJECT_ROOT / ".tree-cache"
DOCLING_CACHE_DIR = TREE_CACHE_DIR / "docling-cache"
OUTPUT_DIR = PROJECT_ROOT / "audit"

GCS_BUCKET = "raptor-assets"

DOC_MAP = {
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
}

# CR manifest run IDs (the actual CR ingestion runs)
CR_RUN_IDS = {
    "AS-AMM-01-000": "20260526_102628_d1d7acd8",
    "SC10000AMM": "20260526_092658_bb2dd851",
}

# Full context-aware run IDs (for side-by-side comparison)
FULL_RUN_IDS = {
    "AS-AMM-01-000": "20260517_075801_9044c50d",
    "SC10000AMM": "20260517_075856_639c04b2",
}

DATASET_DIR = PROJECT_ROOT / "datasets"
COMBINED_DATASET = DATASET_DIR / "combined_eval_dataset_anchored_contextual_retrieval.json"
IMAGE_DATASET = DATASET_DIR / "image_eval_dataset.json"

K_VALUES = [1, 3, 5, 10]
K_PRIME = 30

CLASSIFICATION_PROMPT = """\
You are auditing the quality of LLM-generated context for a retrieval system.

GROUND TRUTH section header for this chunk's pages: "{section_header}"
Pages: {pages}

GENERATED CONTEXT:
"{generated_context}"

ORIGINAL CHUNK TEXT (first 300 chars):
"{original_text_preview}"

Classify this generated context into exactly ONE category:
A - CORRECT and SPECIFIC: Correctly identifies the section/table/topic AND adds discriminative info
B - CORRECT but VAGUE: Not factually wrong but too generic to help retrieval
C - HALLUCINATED wrong attribution: Attributes chunk to wrong section/table/chapter/system
D - HALLUCINATED fabricated details: Invents specific details (section numbers, part numbers) not present in the document

Respond with ONLY: category letter, then a pipe, then a 10-word max justification.
Example: "B|Generic description without naming specific section"
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChunkAudit:
    chunk_idx: int
    chunk_id: str
    original_text_preview: str
    generated_context: str
    pages: list[int]
    ground_truth_section: str
    category: str = ""
    justification: str = ""
    chunk_type: str = ""  # "table" or "prose"
    error: str | None = None


@dataclass
class DocumentAudit:
    doc_name: str
    chunks: list[ChunkAudit] = field(default_factory=list)

    @property
    def valid_chunks(self) -> list[ChunkAudit]:
        return [c for c in self.chunks if c.error is None and c.category in "ABCD"]

    def count_by_category(self, chunk_type: str | None = None) -> dict[str, int]:
        filtered = self.valid_chunks
        if chunk_type:
            filtered = [c for c in filtered if c.chunk_type == chunk_type]
        counts = Counter(c.category for c in filtered)
        return {cat: counts.get(cat, 0) for cat in "ABCD"}

    def pct_by_category(self, chunk_type: str | None = None) -> dict[str, float]:
        counts = self.count_by_category(chunk_type)
        total = sum(counts.values())
        if total == 0:
            return {cat: 0.0 for cat in "ABCD"}
        return {cat: n / total * 100 for cat, n in counts.items()}


# ---------------------------------------------------------------------------
# Step 1: Build page→section ground truth from Docling
# ---------------------------------------------------------------------------


def build_page_sections(cache_key: str) -> dict[int, str | None]:
    """Load Docling output and build page_no → section_header mapping."""
    from docling_core.types.doc import DoclingDocument as DoclingDocumentModel
    from docling_core.types.doc import SectionHeaderItem

    docling_path = DOCLING_CACHE_DIR / cache_key / "docling_output.json"
    if not docling_path.exists():
        raise FileNotFoundError(f"Docling output not found: {docling_path}")

    logger.info(f"Loading Docling output from {docling_path}")
    docling_dict = json.loads(docling_path.read_text())
    doc = DoclingDocumentModel.model_validate(docling_dict)

    # Detect heading level (same heuristic as tree_builder)
    heading_level_counts: Counter[int] = Counter()
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem) and item.text.strip():
            heading_level_counts[item.level] += 1

    max_section_level = 2
    for lvl in sorted(heading_level_counts):
        if heading_level_counts[lvl] > 1:
            max_section_level = lvl
            break

    pages_dict: dict[int, str | None] = {}
    current_section: str | None = None

    for item, _ in doc.iterate_items():
        page_no = 1
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no

        if isinstance(item, SectionHeaderItem) and item.text.strip():
            if item.level <= max_section_level:
                current_section = item.text.strip()

        pages_dict[page_no] = current_section

    logger.info(f"Built page→section mapping: {len(pages_dict)} pages, "
                f"{len(set(pages_dict.values()) - {None})} unique sections")
    return pages_dict


# ---------------------------------------------------------------------------
# Step 2: Load CR context logs from manifest
# ---------------------------------------------------------------------------


def load_cr_chunks(doc_name: str) -> list[dict]:
    """Load CR context log entries from the tree manifest."""
    cache_key = DOC_MAP[doc_name]
    run_id = CR_RUN_IDS[doc_name]
    manifest_path = (
        TREE_CACHE_DIR / cache_key / run_id / "contextual_retrieval" / "tree_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"CR manifest not found: {manifest_path}")

    logger.info(f"Loading CR manifest for {doc_name}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    chunks = manifest["tree_stats"]["cr_context_log"]["chunks"]
    logger.info(f"  Loaded {len(chunks)} CR chunks")
    return chunks


# ---------------------------------------------------------------------------
# Step 3: LLM classification via Gemini Flash
# ---------------------------------------------------------------------------


def _parse_classification(text: str) -> tuple[str, str]:
    """Parse LLM response into (category, justification)."""
    text = text.strip()
    if "|" in text:
        cat, justification = text.split("|", 1)
        cat = cat.strip().upper()
        if cat in "ABCD":
            return cat, justification.strip()
    # Fallback: extract first valid letter
    for char in text:
        if char.upper() in "ABCD":
            return char.upper(), text
    return "", f"[PARSE_FAILED] {text}"


def _classify_worker(
    chunks: list[ChunkAudit],
    project: str,
    location: str,
    worker_id: int,
) -> int:
    """Worker function: creates its own client and classifies a batch of chunks."""
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(vertexai=True, project=project, location=location)
    model_name = "gemini-2.5-flash"
    config = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=50,
    )
    errors = 0

    for chunk in chunks:
        prompt = CLASSIFICATION_PROMPT.format(
            section_header=chunk.ground_truth_section or "(no section header — front matter)",
            pages=chunk.pages,
            generated_context=chunk.generated_context,
            original_text_preview=chunk.original_text_preview[:300],
        )
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                cat, justification = _parse_classification(response.text)
                if cat:
                    chunk.category = cat
                    chunk.justification = justification
                else:
                    chunk.category = "B"
                    chunk.justification = justification
                break
            except Exception as e:
                err_str = str(e)
                if attempt < 4:
                    delay = min(2 ** (attempt + 1), 30)
                    if "429" in err_str:
                        delay = max(delay, 10)
                    time.sleep(delay)
                else:
                    chunk.category = "?"
                    chunk.justification = f"[API_ERROR] {err_str[:80]}"
                    errors += 1

    return errors


def classify_all_chunks(audits: list[DocumentAudit]) -> None:
    """Run LLM classification with concurrent workers (each has own client)."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dotenv import load_dotenv

    backend_env = Path("/Users/joanne/Desktop/PycharmProjects/hitl-dss-react/backend/.env")
    if backend_env.exists():
        load_dotenv(backend_env, override=False)

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set")

    all_chunks = []
    for audit in audits:
        for chunk in audit.chunks:
            if chunk.error is None and not chunk.category:
                all_chunks.append(chunk)

    n_workers = 5
    batch_size = (len(all_chunks) + n_workers - 1) // n_workers
    batches = [all_chunks[i:i + batch_size] for i in range(0, len(all_chunks), batch_size)]

    logger.info(f"Classifying {len(all_chunks)} chunks via Gemini Flash "
                f"({n_workers} workers, {batch_size} chunks/worker)...")

    total_errors = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_classify_worker, batch, project, location, i): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            errors = future.result()
            total_errors += errors
            classified = sum(1 for c in batches[worker_id] if c.category in "ABCD")
            logger.info(f"  Worker {worker_id} done: {classified}/{len(batches[worker_id])} "
                        f"classified, {errors} errors")

    total_classified = sum(1 for c in all_chunks if c.category in "ABCD")
    logger.info(f"  Classification complete: {total_classified}/{len(all_chunks)} classified, "
                f"{total_errors} errors")


# ---------------------------------------------------------------------------
# Step 4: Table vs Prose classification
# ---------------------------------------------------------------------------


def classify_chunk_type(text_preview: str) -> str:
    """Determine if a chunk is table or prose based on its text."""
    if text_preview.startswith("|") or "|---|" in text_preview or "| ---" in text_preview:
        return "table"
    # Also check for pipe-separated rows
    lines = text_preview.split("\n")
    pipe_lines = sum(1 for line in lines[:5] if line.strip().startswith("|"))
    if pipe_lines >= 2:
        return "table"
    return "prose"


# ---------------------------------------------------------------------------
# Step 5: Side-by-side comparison (Phase 3)
# ---------------------------------------------------------------------------


def load_full_tree_headers(doc_name: str) -> dict[int, str]:
    """Load full_context_aware tree and extract page→section from leaf nodes."""
    cache_key = DOC_MAP[doc_name]
    run_id = FULL_RUN_IDS[doc_name]
    tree_path = TREE_CACHE_DIR / cache_key / run_id / "full_context_aware" / "tree.pkl"

    if not tree_path.exists():
        raise FileNotFoundError(f"Full context-aware tree not found: {tree_path}")

    logger.info(f"Loading full_context_aware tree for {doc_name}...")
    with open(tree_path, "rb") as f:
        nodes = pickle.load(f)

    page_to_header: dict[int, str] = {}
    for node in nodes:
        if node.get("level", 0) != 0:
            continue
        section = node.get("section_header")
        if not section:
            continue
        pages = node.get("pages", [])
        for p in pages:
            if p not in page_to_header:
                page_to_header[p] = section

    logger.info(f"  Extracted headers for {len(page_to_header)} pages")
    return page_to_header


def generate_side_by_side(audit: DocumentAudit, n_samples: int = 20) -> list[dict]:
    """Generate side-by-side comparison of header prefix vs CR context."""
    valid = [c for c in audit.chunks if c.error is None and c.ground_truth_section]
    if len(valid) < n_samples:
        sample = valid
    else:
        random.seed(42)
        sample = random.sample(valid, n_samples)

    comparisons = []
    for chunk in sample:
        header_prefix = f"[{chunk.ground_truth_section} | Page {chunk.pages[0] if chunk.pages else '?'}]"
        # Rough token count (words)
        header_tokens = len(header_prefix.split())
        cr_tokens = len(chunk.generated_context.split())

        # Determine which is more specific
        cr_lower = chunk.generated_context.lower()
        section_lower = (chunk.ground_truth_section or "").lower()

        if section_lower and section_lower in cr_lower:
            verdict = "Roughly equivalent (CR mentions section name)"
        elif len(chunk.generated_context) > len(header_prefix) * 2:
            verdict = "CR is MORE verbose but may not be more specific"
        else:
            verdict = "Header is MORE specific (names exact section)"

        comparisons.append({
            "chunk_idx": chunk.chunk_idx,
            "pages": chunk.pages,
            "header_prefix": header_prefix,
            "cr_context": chunk.generated_context,
            "verdict": verdict,
            "header_tokens": header_tokens,
            "cr_tokens": cr_tokens,
        })

    return comparisons


# ---------------------------------------------------------------------------
# Step 6: Per-query failure analysis (Phase 4)
# ---------------------------------------------------------------------------


def run_per_query_failure_analysis(
    doc_name: str, audit: DocumentAudit
) -> dict:
    """Re-run evaluation and cross-reference failures with CR context categories."""
    from evaluation.eval_runner import (
        load_and_merge_datasets,
        resolve_ablation_dataset,
    )
    from evaluation.retrieval_evaluator import (
        evaluate_single_query,
        load_tree_index,
        QueryResult,
    )
    from modules.embeddings import embed

    import numpy as np

    cache_key = DOC_MAP[doc_name]

    # Load datasets
    dataset = load_and_merge_datasets(COMBINED_DATASET, IMAGE_DATASET)

    # Get per-ablation datasets
    cr_dataset = resolve_ablation_dataset(
        "contextual_retrieval", DATASET_DIR, dataset,
        image_path=IMAGE_DATASET, manual_source=doc_name,
    )
    naive_dataset = resolve_ablation_dataset(
        "no_context_aware", DATASET_DIR, dataset,
        image_path=IMAGE_DATASET, manual_source=doc_name,
    )

    # Load trees
    logger.info(f"Loading CR tree for {doc_name}...")
    cr_index, _ = load_tree_index(cache_key, "contextual_retrieval", bucket=GCS_BUCKET)
    logger.info(f"Loading naive tree for {doc_name}...")
    naive_index, _ = load_tree_index(cache_key, "no_context_aware", bucket=GCS_BUCKET)

    # Build chunk_id → category mapping from audit
    chunk_categories = {}
    for chunk in audit.chunks:
        if chunk.category:
            chunk_categories[chunk.chunk_id] = chunk.category

    # Run per-query evaluation
    cr_queries = cr_dataset["queries"]
    naive_queries = naive_dataset["queries"]

    # Build query_id → query mapping for naive
    naive_by_id = {q["query_id"]: q for q in naive_queries}

    naive_hits: set[str] = set()  # query_ids where naive hits @10
    cr_hits: set[str] = set()    # query_ids where CR hits @10
    cr_results_map: dict[str, QueryResult] = {}
    naive_results_map: dict[str, QueryResult] = {}

    logger.info(f"Running per-query eval for {doc_name} ({len(cr_queries)} CR queries)...")

    for query in cr_queries:
        query_emb = embed(text=query["query_text"])
        if query_emb is None:
            continue
        query_emb = np.array(query_emb, dtype=np.float32)

        cr_qr, _ = evaluate_single_query(cr_index, query, query_emb, K_PRIME)
        cr_results_map[query["query_id"]] = cr_qr

        # Check if hit @10
        gt_chunks = query.get("relevant_chunks", [])
        if gt_chunks:
            gt_id = gt_chunks[0]["chunk_id"]
            cr_top10 = set(cr_qr.retrieved_ids[:10])
            # Check exact match or parent-child
            if gt_id in cr_top10:
                cr_hits.add(query["query_id"])

    logger.info(f"Running per-query eval for {doc_name} ({len(naive_queries)} naive queries)...")

    for query in naive_queries:
        query_emb = embed(text=query["query_text"])
        if query_emb is None:
            continue
        query_emb = np.array(query_emb, dtype=np.float32)

        naive_qr, _ = evaluate_single_query(naive_index, query, query_emb, K_PRIME)
        naive_results_map[query["query_id"]] = naive_qr

        gt_chunks = query.get("relevant_chunks", [])
        if gt_chunks:
            gt_id = gt_chunks[0]["chunk_id"]
            naive_top10 = set(naive_qr.retrieved_ids[:10])
            if gt_id in naive_top10:
                naive_hits.add(query["query_id"])

    # Queries where naive hit but CR missed
    naive_hit_cr_miss = naive_hits - cr_hits
    # Queries where CR hit but naive missed
    cr_hit_naive_miss = cr_hits - naive_hits

    # For missed queries, find the GT chunk and check its CR context category
    category_of_missed = Counter()
    missed_details = []

    for qid in naive_hit_cr_miss:
        # Find the CR query to get ground truth chunk
        cr_query = next((q for q in cr_queries if q["query_id"] == qid), None)
        if cr_query is None:
            continue
        gt_chunks = cr_query.get("relevant_chunks", [])
        if not gt_chunks:
            continue
        gt_id = gt_chunks[0]["chunk_id"]
        cat = chunk_categories.get(gt_id, "unknown")
        category_of_missed[cat] += 1
        missed_details.append({
            "query_id": qid,
            "query_text": cr_query["query_text"][:100],
            "gt_chunk_id": gt_id,
            "cr_context_category": cat,
        })

    result = {
        "total_queries": len(cr_queries),
        "naive_hits_at_10": len(naive_hits),
        "cr_hits_at_10": len(cr_hits),
        "naive_hit_cr_miss": len(naive_hit_cr_miss),
        "cr_hit_naive_miss": len(cr_hit_naive_miss),
        "missed_chunk_categories": dict(category_of_missed),
        "missed_details": missed_details[:20],  # Top 20 for report
    }

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    audits: list[DocumentAudit],
    side_by_side: dict[str, list[dict]],
    failure_analysis: dict[str, dict],
) -> str:
    """Generate the full text report."""
    lines = []
    lines.append("=" * 72)
    lines.append("CR CONTEXT QUALITY AUDIT")
    lines.append("=" * 72)
    lines.append("")

    # Phase 1 summary
    for audit in audits:
        n = len(audit.valid_chunks)
        pcts = audit.pct_by_category()
        counts = audit.count_by_category()
        lines.append(f"{audit.doc_name} (n={n} chunks):")
        lines.append(f"  A (correct+specific):  {counts['A']:3d} ({pcts['A']:5.1f}%)")
        lines.append(f"  B (correct+vague):     {counts['B']:3d} ({pcts['B']:5.1f}%)")
        lines.append(f"  C (wrong attribution): {counts['C']:3d} ({pcts['C']:5.1f}%)")
        lines.append(f"  D (fabricated details): {counts['D']:3d} ({pcts['D']:5.1f}%)")
        lines.append("")

    # By chunk type
    lines.append("TABLE chunks only:")
    for audit in audits:
        pcts = audit.pct_by_category("table")
        lines.append(f"  {audit.doc_name}:  A={pcts['A']:.0f}% B={pcts['B']:.0f}% "
                     f"C={pcts['C']:.0f}% D={pcts['D']:.0f}%")
    lines.append("")
    lines.append("PROSE chunks only:")
    for audit in audits:
        pcts = audit.pct_by_category("prose")
        lines.append(f"  {audit.doc_name}:  A={pcts['A']:.0f}% B={pcts['B']:.0f}% "
                     f"C={pcts['C']:.0f}% D={pcts['D']:.0f}%")
    lines.append("")

    # Phase 2: Worst offenders
    lines.append("=" * 72)
    lines.append("WORST OFFENDERS (Category C and D)")
    lines.append("=" * 72)

    for audit in audits:
        lines.append(f"\n{audit.doc_name}:")
        lines.append("-" * 72)

        # Prioritize C and D
        worst = sorted(
            [c for c in audit.valid_chunks if c.category in ("C", "D")],
            key=lambda c: ("D", "C").index(c.category) if c.category in ("D", "C") else 2,
        )[:10]

        if not worst:
            worst = sorted(
                [c for c in audit.valid_chunks if c.category == "B"],
                key=lambda c: len(c.generated_context),
            )[:10]

        for chunk in worst:
            lines.append(f"\nCHUNK #{chunk.chunk_idx} ({audit.doc_name}, "
                         f"page {chunk.pages[0] if chunk.pages else '?'}, "
                         f"type: {chunk.chunk_type.upper()})")
            lines.append("─" * 50)
            lines.append(f"Ground truth header: [{chunk.ground_truth_section} | "
                         f"Page {chunk.pages[0] if chunk.pages else '?'}]")
            lines.append(f"CR context:          \"{chunk.generated_context}\"")
            lines.append(f"Original chunk text: \"{chunk.original_text_preview[:120]}\"")
            lines.append(f"Category:            {chunk.category} ({chunk.justification})")
            lines.append("")

    # Phase 3: Side-by-side
    lines.append("=" * 72)
    lines.append("SIDE-BY-SIDE: HEADER PREFIX vs CR CONTEXT (20 samples per doc)")
    lines.append("=" * 72)

    header_more_specific = Counter()
    cr_more_specific = Counter()
    equivalent = Counter()

    for doc_name, comparisons in side_by_side.items():
        lines.append(f"\n{doc_name}:")
        lines.append("-" * 72)
        for comp in comparisons:
            lines.append(f"\nCHUNK #{comp['chunk_idx']} (page {comp['pages'][0] if comp['pages'] else '?'})")
            lines.append(f"  Header prefix:  \"{comp['header_prefix']}\"")
            lines.append(f"  CR context:     \"{comp['cr_context']}\"")
            lines.append(f"  Verdict:        {comp['verdict']}")
            lines.append(f"  Token overhead: Header={comp['header_tokens']} tokens, "
                         f"CR={comp['cr_tokens']} tokens")

            if "MORE specific" in comp["verdict"] and "Header" in comp["verdict"]:
                header_more_specific[doc_name] += 1
            elif "MORE" in comp["verdict"] and "CR" in comp["verdict"]:
                cr_more_specific[doc_name] += 1
            else:
                equivalent[doc_name] += 1

    lines.append(f"\n\nSummary:")
    for doc_name in side_by_side:
        lines.append(f"  {doc_name}: Header more specific: {header_more_specific[doc_name]}, "
                     f"CR more specific: {cr_more_specific[doc_name]}, "
                     f"Equivalent: {equivalent[doc_name]}")

    # Phase 4: Failure analysis
    lines.append("")
    lines.append("=" * 72)
    lines.append("RETRIEVAL FAILURE ANALYSIS")
    lines.append("=" * 72)

    for doc_name, analysis in failure_analysis.items():
        lines.append(f"\n{doc_name}:")
        lines.append(f"  Total queries: {analysis['total_queries']}")
        lines.append(f"  Naive hits @10: {analysis['naive_hits_at_10']}")
        lines.append(f"  CR hits @10: {analysis['cr_hits_at_10']}")
        lines.append(f"  Queries where naive hit but CR missed: {analysis['naive_hit_cr_miss']}")
        lines.append(f"  Queries where CR hit but naive missed: {analysis['cr_hit_naive_miss']}")
        lines.append("")
        lines.append(f"  Of the chunks that SHOULD have been retrieved (naive hit, CR miss):")
        cats = analysis.get("missed_chunk_categories", {})
        total_missed = sum(cats.values())
        if total_missed:
            for cat in "ABCD":
                n = cats.get(cat, 0)
                pct = n / total_missed * 100 if total_missed else 0
                label = {"A": "correct+specific", "B": "vague context diluted signal",
                         "C": "wrong context pushed embedding away",
                         "D": "fabricated details misled embedding"}.get(cat, "unknown")
                lines.append(f"    CR context was Category {cat}: {n} ({pct:.0f}%) ← {label}")
            unknown = cats.get("unknown", 0)
            if unknown:
                lines.append(f"    Unknown (chunk not in CR tree): {unknown}")
        else:
            lines.append("    (no failures to analyze)")

    return "\n".join(lines)


def generate_summary_json(
    audits: list[DocumentAudit],
    failure_analysis: dict[str, dict],
) -> dict:
    """Generate structured JSON summary."""
    result = {}
    for audit in audits:
        counts = audit.count_by_category()
        table_counts = audit.count_by_category("table")
        prose_counts = audit.count_by_category("prose")
        result[audit.doc_name] = {
            "total_chunks": len(audit.valid_chunks),
            "category_A": counts["A"],
            "category_B": counts["B"],
            "category_C": counts["C"],
            "category_D": counts["D"],
            "table_chunks": table_counts,
            "prose_chunks": prose_counts,
            "failure_analysis": failure_analysis.get(audit.doc_name, {}),
        }
    return result


def generate_worst_offenders_json(audits: list[DocumentAudit]) -> dict:
    """Generate JSON with 10 worst offenders per document."""
    result = {}
    for audit in audits:
        worst = sorted(
            [c for c in audit.valid_chunks if c.category in ("C", "D")],
            key=lambda c: (0 if c.category == "D" else 1, c.chunk_idx),
        )[:10]

        if len(worst) < 10:
            extra_b = sorted(
                [c for c in audit.valid_chunks if c.category == "B"],
                key=lambda c: len(c.generated_context),
            )[:10 - len(worst)]
            worst.extend(extra_b)

        result[audit.doc_name] = [
            {
                "chunk_idx": c.chunk_idx,
                "chunk_id": c.chunk_id,
                "pages": c.pages,
                "chunk_type": c.chunk_type,
                "ground_truth_section": c.ground_truth_section,
                "generated_context": c.generated_context,
                "original_text_preview": c.original_text_preview[:200],
                "category": c.category,
                "justification": c.justification,
            }
            for c in worst
        ]
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 72)
    logger.info("CR CONTEXT QUALITY AUDIT")
    logger.info("=" * 72)

    # Step 1: Build page→section ground truth
    page_sections: dict[str, dict[int, str | None]] = {}
    for doc_name, cache_key in DOC_MAP.items():
        page_sections[doc_name] = build_page_sections(cache_key)

    # Step 2: Load CR chunks and build audit objects
    audits: list[DocumentAudit] = []
    for doc_name in DOC_MAP:
        cr_chunks = load_cr_chunks(doc_name)
        doc_audit = DocumentAudit(doc_name=doc_name)

        for chunk_entry in cr_chunks:
            pages = chunk_entry.get("pages", [])
            # Resolve ground truth section: use the last page's section (most specific)
            gt_section = None
            for p in pages:
                s = page_sections[doc_name].get(p)
                if s:
                    gt_section = s

            chunk_type = classify_chunk_type(chunk_entry.get("original_text_preview", ""))

            doc_audit.chunks.append(ChunkAudit(
                chunk_idx=chunk_entry["chunk_idx"],
                chunk_id=chunk_entry["chunk_id"],
                original_text_preview=chunk_entry.get("original_text_preview", ""),
                generated_context=chunk_entry.get("generated_context", ""),
                pages=pages,
                ground_truth_section=gt_section or "",
                chunk_type=chunk_type,
                error=chunk_entry.get("error"),
            ))

        audits.append(doc_audit)
        logger.info(f"  {doc_name}: {len(doc_audit.chunks)} chunks, "
                    f"{sum(1 for c in doc_audit.chunks if c.error)} errors")

    # Step 3: LLM classification
    classify_all_chunks(audits)

    # Log quick stats
    for audit in audits:
        pcts = audit.pct_by_category()
        logger.info(f"  {audit.doc_name}: A={pcts['A']:.1f}% B={pcts['B']:.1f}% "
                    f"C={pcts['C']:.1f}% D={pcts['D']:.1f}%")

    # Step 5: Side-by-side comparison
    side_by_side: dict[str, list[dict]] = {}
    for audit in audits:
        side_by_side[audit.doc_name] = generate_side_by_side(audit, n_samples=20)

    # Step 6: Per-query failure analysis
    failure_analysis: dict[str, dict] = {}
    for doc_name in DOC_MAP:
        audit = next(a for a in audits if a.doc_name == doc_name)
        try:
            failure_analysis[doc_name] = run_per_query_failure_analysis(doc_name, audit)
        except Exception as e:
            logger.error(f"Failure analysis failed for {doc_name}: {e}")
            failure_analysis[doc_name] = {"error": str(e)}

    # Generate outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_report(audits, side_by_side, failure_analysis)
    report_path = OUTPUT_DIR / "cr_audit_report.txt"
    report_path.write_text(report)
    logger.info(f"Report saved: {report_path}")

    summary = generate_summary_json(audits, failure_analysis)
    summary_path = OUTPUT_DIR / "cr_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary saved: {summary_path}")

    worst = generate_worst_offenders_json(audits)
    worst_path = OUTPUT_DIR / "cr_worst_offenders.json"
    worst_path.write_text(json.dumps(worst, indent=2))
    logger.info(f"Worst offenders saved: {worst_path}")

    # Print report to stdout
    print(report)

    logger.info("\nAudit complete.")


if __name__ == "__main__":
    main()
