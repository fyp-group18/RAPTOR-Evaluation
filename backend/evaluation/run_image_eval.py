"""Run retrieval evaluation on the 45 fixed image queries.

Loads enriched image annotations (from fix_image_query_annotations.py),
constructs per-ablation query sets with real chunk IDs, and runs:
  - Seq 1: full_context_aware vs original_raptor_text_only (image queries only)
  - Seq 3: ablation study on sub-innovations

Uses the existing evaluation infrastructure (TreeIndex, metrics, bootstrap).
Does NOT modify any existing result files — all output goes to results/image_queries/.

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
    GOOGLE_CLOUD_PROJECT=<your-gcp-project> \
    PYTHONPATH="/path/to/raptor/backend:." \
    python -m backend.evaluation.run_image_eval
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "image_queries"

GCS_BUCKET = "raptor-assets"

DOC_MAP = {
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
}

K_VALUES = [1, 3, 5, 10]
K_PRIME = 30

# Sequence definitions (mirror eval_runner.py)
SEQ1_COMPARISONS = [
    ("text_only_raptor", "original_raptor_text_only"),
    ("full", "full_context_aware"),
]

SEQ3_ABLATIONS = [
    ("full", "full_context_aware"),
    ("no_table_pc", "no_table_parent_child"),
    ("no_header_prop", "no_header_propagation"),
    ("no_caption_fold", "no_caption_folding"),
    ("no_context_aware", "baseline_naive_chunking"),
]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_normalized_queries(
    fixed_queries: list[dict],
    gt_per_ablation: dict[str, dict[str, str | None]],
    ablation_label: str,
    manual_source: str | None = None,
) -> list[dict]:
    """Build normalized query dicts for a specific ablation.

    Uses per-ablation chunk IDs for exact matching, with corrected
    page numbers as fallback.
    """
    queries = []
    skipped = 0

    for q in fixed_queries:
        if q["confidence"] == "unmatched":
            skipped += 1
            continue

        if manual_source and q["manual_source"] != manual_source:
            continue

        qid = q["id"]
        chunk_id = gt_per_ablation.get(qid, {}).get(ablation_label)

        relevant_chunks = []
        if chunk_id:
            relevant_chunks = [{"chunk_id": chunk_id, "relevance": 2}]

        relevant_pages = []
        if q.get("page_reference") is not None:
            relevant_pages = [int(q["page_reference"])]

        queries.append({
            "query_id": qid,
            "query_text": q["observation"],
            "query_type": "image",
            "relevant_chunks": relevant_chunks,
            "relevant_pages": relevant_pages,
            "answer_text": q.get("ground_truth_fault_code", ""),
            "difficulty": q.get("difficulty_tier", "unknown"),
            "manual_source": q["manual_source"],
        })

    if skipped:
        logger.info(f"  Skipped {skipped} unmatched queries")

    return queries


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_image_evaluation(
    fixed_queries: list[dict],
    gt_per_ablation: dict[str, dict[str, str | None]],
    cache_key: str,
    comparisons: list[tuple[str, str]],
    seq_num: int,
    manual_source: str | None = None,
    doc_label: str | None = None,
) -> dict:
    """Run evaluation for a set of ablation comparisons on image queries."""
    from evaluation.retrieval_evaluator import (
        evaluate_ablation,
        run_statistical_comparisons,
    )

    max_k = max(K_VALUES)
    results: dict[str, dict] = {}
    all_scores: dict[str, dict[str, dict[str, list[float]]]] = {}

    for ablation_name, ablation_label in comparisons:
        logger.info(f"  Evaluating: {ablation_label}")

        queries = build_normalized_queries(
            fixed_queries, gt_per_ablation, ablation_label,
            manual_source=manual_source,
        )

        if not queries:
            logger.warning(f"  No queries for {ablation_label}, skipping")
            continue

        dataset = {
            "metadata": {"version": "image_eval_fixed"},
            "queries": queries,
        }

        metrics, scores = evaluate_ablation(
            cache_key=cache_key,
            ablation_name=ablation_name,
            dataset=dataset,
            k_values=K_VALUES,
            run_id=None,
            bucket=GCS_BUCKET,
            k_prime=K_PRIME,
            retrieval_mode="embedding_only",
        )
        results[ablation_label] = metrics
        all_scores[ablation_label] = scores

    if not results:
        return {}

    # Statistical tests
    metrics_to_test = [f"recall@{k}" for k in K_VALUES] + ["mrr", f"ndcg@{max_k}"]
    stat_tests = {}

    if seq_num in (1, 2) and len(results) == 2:
        baseline_label = comparisons[0][1]
        proposed_label = comparisons[-1][1]
        if baseline_label in all_scores and proposed_label in all_scores:
            stat_tests = run_statistical_comparisons(
                all_scores[baseline_label],
                all_scores[proposed_label],
                baseline_label,
                proposed_label,
                metrics_to_test,
            )
    elif seq_num == 3:
        full_label = comparisons[0][1]
        if full_label in all_scores:
            for _, ablation_label in comparisons[1:]:
                if ablation_label in all_scores:
                    tests = run_statistical_comparisons(
                        all_scores[full_label],
                        all_scores[ablation_label],
                        full_label,
                        ablation_label,
                        metrics_to_test,
                    )
                    stat_tests.update(tests)

    # Print table
    from evaluation.eval_runner import print_comparison_table, print_contribution_ranking
    title_suffix = f" [{doc_label}]" if doc_label else ""
    if seq_num == 1:
        title = f"Image Queries: Multimodal RAPTOR vs Original RAPTOR{title_suffix}"
    elif seq_num == 3:
        title = f"Image Queries: Ablation Study{title_suffix}"
    else:
        title = f"Image Queries: Seq {seq_num}{title_suffix}"

    active_comparisons = [(n, l) for n, l in comparisons if l in results]
    if active_comparisons:
        print_comparison_table(
            seq_num, title, "latest",
            "image_eval_fixed",
            results, stat_tests, K_VALUES, active_comparisons,
        )
        if seq_num == 3:
            print_contribution_ranking(results, K_VALUES)

    return {
        "sequence": seq_num,
        "document": doc_label or "all",
        "retrieval_mode": "embedding_only",
        "k_prime": K_PRIME,
        "run_id": "latest",
        "dataset_version": "image_eval_fixed",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "k_values": K_VALUES,
        "results": results,
        "statistical_tests": stat_tests,
        "num_queries": sum(
            r.get("overall", {}).get("num_queries", 0) for r in results.values()
        ) // max(len(results), 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 72)
    logger.info("Image Query Evaluation")
    logger.info("=" * 72)

    # Load fixed dataset
    fixed_path = DATA_DIR / "image_eval_dataset_fixed.json"
    if not fixed_path.exists():
        logger.error(
            f"Fixed dataset not found at {fixed_path}. "
            f"Run fix_image_query_annotations.py first."
        )
        sys.exit(1)

    fixed_queries = json.loads(fixed_path.read_text())
    logger.info(f"Loaded {len(fixed_queries)} fixed image queries")

    # Load per-ablation ground truth
    gt_path = DATA_DIR / "image_eval_gt_per_ablation.json"
    if not gt_path.exists():
        logger.error(f"Per-ablation GT not found at {gt_path}")
        sys.exit(1)

    gt_per_ablation = json.loads(gt_path.read_text())
    logger.info(f"Loaded per-ablation ground truth for {len(gt_per_ablation)} queries")

    # Report skipped queries
    unmatched = [q for q in fixed_queries if q["confidence"] == "unmatched"]
    if unmatched:
        logger.warning(f"Skipping {len(unmatched)} unmatched queries:")
        for q in unmatched:
            logger.warning(f"  {q['id']}: {q.get('ground_truth_figure', '?')}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_outputs = []
    summary_lines = []

    # -----------------------------------------------------------------------
    # Seq 1: full_context_aware vs original_raptor_text_only
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Running Seq 1: Multimodal RAPTOR vs Original RAPTOR")
    logger.info("=" * 60)

    # Per-document evaluation
    for doc_label, cache_key in DOC_MAP.items():
        logger.info(f"\n--- {doc_label} ---")
        output = run_image_evaluation(
            fixed_queries, gt_per_ablation, cache_key,
            SEQ1_COMPARISONS, seq_num=1,
            manual_source=doc_label, doc_label=doc_label,
        )
        if output:
            all_outputs.append(output)

    # Combined evaluation (use first cache key — queries are filtered by manual_source
    # inside evaluate_ablation, but we need separate calls per document)
    # Already handled per-document above, just save combined results

    # -----------------------------------------------------------------------
    # Seq 3: Ablation study (if enough queries)
    # -----------------------------------------------------------------------
    by_source = Counter(q["manual_source"] for q in fixed_queries if q["confidence"] != "unmatched")
    run_seq3 = all(count >= 10 for count in by_source.values())

    if run_seq3:
        logger.info("\n" + "=" * 60)
        logger.info("Running Seq 3: Ablation Study")
        logger.info("=" * 60)

        for doc_label, cache_key in DOC_MAP.items():
            doc_count = by_source.get(doc_label, 0)
            if doc_count < 5:
                logger.info(f"  Skipping {doc_label} ({doc_count} queries < 5)")
                continue

            logger.info(f"\n--- {doc_label} ({doc_count} queries) ---")
            output = run_image_evaluation(
                fixed_queries, gt_per_ablation, cache_key,
                SEQ3_ABLATIONS, seq_num=3,
                manual_source=doc_label, doc_label=doc_label,
            )
            if output:
                all_outputs.append(output)
    else:
        logger.info("\nSkipping Seq 3: insufficient queries per document")
        logger.info(f"  Per-doc counts: {dict(by_source)}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    # Save all results as JSON
    seq1_outputs = [o for o in all_outputs if o.get("sequence") == 1]
    seq3_outputs = [o for o in all_outputs if o.get("sequence") == 3]

    if seq1_outputs:
        seq1_path = RESULTS_DIR / "seq1_image_results.json"
        seq1_path.write_text(json.dumps(seq1_outputs, indent=2))
        logger.info(f"\nSaved Seq 1 results to {seq1_path}")

    if seq3_outputs:
        seq3_path = RESULTS_DIR / "seq3_image_results.json"
        seq3_path.write_text(json.dumps(seq3_outputs, indent=2))
        logger.info(f"Saved Seq 3 results to {seq3_path}")

    # Save summary text
    summary_path = RESULTS_DIR / "seq1_image_summary.txt"
    summary_lines = [
        "Image Query Evaluation Summary",
        f"Date: {datetime.now(timezone.utc).isoformat()}",
        f"Total queries: {len(fixed_queries)}",
        f"Unmatched (skipped): {len(unmatched)}",
        f"Evaluated: {len(fixed_queries) - len(unmatched)}",
        "",
    ]

    for output in all_outputs:
        doc = output.get("document", "all")
        seq = output.get("sequence", "?")
        summary_lines.append(f"\nSeq {seq} — {doc}:")

        for abl_label, metrics in output.get("results", {}).items():
            overall = metrics.get("overall", {})
            n = overall.get("num_queries", 0)
            summary_lines.append(
                f"  {abl_label:<30s} n={n:>3d}  "
                f"R@1={overall.get('recall@1', 0):.3f}  "
                f"R@5={overall.get('recall@5', 0):.3f}  "
                f"R@10={overall.get('recall@10', 0):.3f}  "
                f"MRR={overall.get('mrr', 0):.3f}  "
                f"NDCG@10={overall.get('ndcg@10', 0):.3f}"
            )

        # Add stat test p-values
        tests = output.get("statistical_tests", {})
        if tests and seq == 1:
            baseline = SEQ1_COMPARISONS[0][1]
            proposed = SEQ1_COMPARISONS[-1][1]
            for metric in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]:
                key = f"{proposed}_vs_{baseline}_overall_{metric}"
                if key in tests:
                    p = tests[key].get("p_value", float("nan"))
                    delta = tests[key].get("delta", 0)
                    summary_lines.append(f"    p({metric}): {p:.4f}  delta={delta:+.3f}")

    summary_path.write_text("\n".join(summary_lines))
    logger.info(f"Saved summary to {summary_path}")

    # Copy anchoring report to results dir
    anchoring_report_src = DATA_DIR / "anchoring_report.json"
    if anchoring_report_src.exists():
        anchoring_report_dst = RESULTS_DIR / "anchoring_report.json"
        anchoring_report_dst.write_text(anchoring_report_src.read_text())
        logger.info(f"Copied anchoring report to {anchoring_report_dst}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
