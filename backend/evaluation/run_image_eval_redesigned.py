"""Run retrieval evaluation on the 45 redesigned (content-based) image queries.

Loads redesigned queries from data/image_queries_redesigned.json, uses
per-ablation ground truth from data/image_eval_gt_per_ablation.json, and runs:
  - Seq 1: full_context_aware vs original_raptor_text_only
  - Seq 3: ablation study on sub-innovations

Also loads original query results from results/image_queries/ to produce
a side-by-side comparison (old symptom-based vs new content-based).

Uses the same evaluation infrastructure as run_image_eval.py (TreeIndex,
metrics, bootstrap). Does NOT modify any existing result files.

All output goes to results/image_queries_redesigned/.

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
    GOOGLE_CLOUD_PROJECT=raptor-496700 \
    PYTHONPATH="/Users/joanne/Desktop/PycharmProjects/hitl-dss-react/backend:." \
    python -m backend.evaluation.run_image_eval_redesigned
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# NumPy 2.x → 1.x compat: tree pickles reference numpy._core (NumPy 2.x)
# but this env has NumPy 1.x which uses numpy.core
import numpy.core  # noqa: E402
sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "image_queries_redesigned"
ORIGINAL_RESULTS_DIR = PROJECT_ROOT / "results" / "image_queries"

GCS_BUCKET = "raptor-assets"

DOC_MAP = {
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
}

K_VALUES = [1, 3, 5, 10]
K_PRIME = 30

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
    redesigned_queries: dict[str, dict],
    gt_per_ablation: dict[str, dict[str, str | None]],
    ablation_label: str,
    manual_source: str | None = None,
) -> list[dict]:
    """Build normalized query dicts using redesigned query text.

    Uses the redesigned_query text instead of the original observation,
    with per-ablation chunk IDs for exact matching.
    """
    queries = []

    for qid, entry in sorted(redesigned_queries.items()):
        if manual_source and entry.get("manual_source") != manual_source:
            continue

        chunk_id = gt_per_ablation.get(qid, {}).get(ablation_label)

        relevant_chunks = []
        if chunk_id:
            relevant_chunks = [{"chunk_id": chunk_id, "relevance": 2}]

        queries.append({
            "query_id": qid,
            "query_text": entry["redesigned_query"],
            "query_type": "image",
            "relevant_chunks": relevant_chunks,
            "relevant_pages": [],
            "answer_text": "",
            "difficulty": "unknown",
            "manual_source": entry["manual_source"],
        })

    return queries


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_redesigned_evaluation(
    redesigned_queries: dict[str, dict],
    gt_per_ablation: dict[str, dict[str, str | None]],
    cache_key: str,
    comparisons: list[tuple[str, str]],
    seq_num: int,
    manual_source: str | None = None,
    doc_label: str | None = None,
) -> dict:
    """Run evaluation for redesigned queries on a set of ablation comparisons."""
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
            redesigned_queries, gt_per_ablation, ablation_label,
            manual_source=manual_source,
        )

        if not queries:
            logger.warning(f"  No queries for {ablation_label}, skipping")
            continue

        dataset = {
            "metadata": {"version": "image_eval_redesigned"},
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
        title = f"REDESIGNED Image Queries: Multimodal RAPTOR vs Original RAPTOR{title_suffix}"
    elif seq_num == 3:
        title = f"REDESIGNED Image Queries: Ablation Study{title_suffix}"
    else:
        title = f"REDESIGNED Image Queries: Seq {seq_num}{title_suffix}"

    active_comparisons = [(n, l) for n, l in comparisons if l in results]
    if active_comparisons:
        print_comparison_table(
            seq_num, title, "latest",
            "image_eval_redesigned",
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
        "dataset_version": "image_eval_redesigned",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "k_values": K_VALUES,
        "results": results,
        "statistical_tests": stat_tests,
        "num_queries": sum(
            r.get("overall", {}).get("num_queries", 0) for r in results.values()
        ) // max(len(results), 1),
    }


# ---------------------------------------------------------------------------
# Step 5: Old vs new comparison
# ---------------------------------------------------------------------------


def _load_original_results() -> list[dict]:
    """Load original image query evaluation results."""
    seq1_path = ORIGINAL_RESULTS_DIR / "seq1_image_results.json"
    seq3_path = ORIGINAL_RESULTS_DIR / "seq3_image_results.json"
    outputs = []
    for p in (seq1_path, seq3_path):
        if p.exists():
            data = json.loads(p.read_text())
            if isinstance(data, list):
                outputs.extend(data)
            else:
                outputs.append(data)
    return outputs


def build_comparison_table(
    original_outputs: list[dict],
    redesigned_outputs: list[dict],
) -> str:
    """Step 5: Build old vs new comparison table."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("COMPARISON: Original (symptom-based) vs Redesigned (content-based) Image Queries")
    lines.append("=" * 100)
    lines.append("")

    header = (
        f"{'Document':<14s} | {'Query Version':<30s} | "
        f"{'R@1':>5s} | {'R@5':>5s} | {'R@10':>5s} | {'MRR':>5s} | {'NDCG@10':>7s} | {'n':>3s}"
    )
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    # Collect per-doc metrics for both versions
    def _extract_rows(outputs: list[dict], version_label: str) -> list[tuple[str, str, dict]]:
        rows = []
        for output in outputs:
            if output.get("sequence") != 1:
                continue
            doc = output.get("document", "all")
            for ablation_label, metrics in output.get("results", {}).items():
                if ablation_label == "full_context_aware":
                    overall = metrics.get("overall", {})
                    rows.append((doc, version_label, overall))
        return rows

    original_rows = _extract_rows(original_outputs, "Original (symptom-based)")
    redesigned_rows = _extract_rows(redesigned_outputs, "Redesigned (content-based)")

    # Interleave by document
    all_docs = sorted(set(
        [r[0] for r in original_rows] + [r[0] for r in redesigned_rows]
    ))

    for doc in all_docs:
        orig = [r for r in original_rows if r[0] == doc]
        rede = [r for r in redesigned_rows if r[0] == doc]
        for _, version, overall in orig + rede:
            n = overall.get("num_queries", 0)
            lines.append(
                f"{doc:<14s} | {version:<30s} | "
                f"{overall.get('recall@1', 0):5.3f} | "
                f"{overall.get('recall@5', 0):5.3f} | "
                f"{overall.get('recall@10', 0):5.3f} | "
                f"{overall.get('mrr', 0):5.3f} | "
                f"{overall.get('ndcg@10', 0):7.3f} | "
                f"{n:3d}"
            )
        lines.append(sep)

    # Delta summary
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  If redesigned queries score HIGHER → original queries had a query design problem")
    lines.append("  If redesigned queries score SIMILAR/LOWER → the issue is architectural (chunking fragmentation)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 72)
    logger.info("Redesigned Image Query Evaluation")
    logger.info("=" * 72)

    # Load redesigned queries
    redesigned_path = DATA_DIR / "image_queries_redesigned.json"
    if not redesigned_path.exists():
        logger.error(
            f"Redesigned queries not found at {redesigned_path}. "
            f"Run redesign_image_queries.py first."
        )
        sys.exit(1)

    redesigned_queries = json.loads(redesigned_path.read_text())
    logger.info(f"Loaded {len(redesigned_queries)} redesigned queries")

    # Load per-ablation ground truth
    gt_path = DATA_DIR / "image_eval_gt_per_ablation.json"
    if not gt_path.exists():
        logger.error(f"Per-ablation GT not found at {gt_path}")
        sys.exit(1)

    gt_per_ablation = json.loads(gt_path.read_text())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    # -----------------------------------------------------------------------
    # Seq 1: full_context_aware vs original_raptor_text_only
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Running Seq 1: Multimodal RAPTOR vs Original RAPTOR (Redesigned Queries)")
    logger.info("=" * 60)

    for doc_label, cache_key in DOC_MAP.items():
        logger.info(f"\n--- {doc_label} ---")
        output = run_redesigned_evaluation(
            redesigned_queries, gt_per_ablation, cache_key,
            SEQ1_COMPARISONS, seq_num=1,
            manual_source=doc_label, doc_label=doc_label,
        )
        if output:
            all_outputs.append(output)

    # -----------------------------------------------------------------------
    # Seq 3: Ablation study
    # -----------------------------------------------------------------------
    by_source = Counter(
        entry["manual_source"]
        for entry in redesigned_queries.values()
    )
    run_seq3 = all(count >= 10 for count in by_source.values())

    if run_seq3:
        logger.info("\n" + "=" * 60)
        logger.info("Running Seq 3: Ablation Study (Redesigned Queries)")
        logger.info("=" * 60)

        for doc_label, cache_key in DOC_MAP.items():
            doc_count = by_source.get(doc_label, 0)
            if doc_count < 5:
                logger.info(f"  Skipping {doc_label} ({doc_count} queries < 5)")
                continue

            logger.info(f"\n--- {doc_label} ({doc_count} queries) ---")
            output = run_redesigned_evaluation(
                redesigned_queries, gt_per_ablation, cache_key,
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
    seq1_outputs = [o for o in all_outputs if o.get("sequence") == 1]
    seq3_outputs = [o for o in all_outputs if o.get("sequence") == 3]

    if seq1_outputs:
        seq1_path = RESULTS_DIR / "seq1_results.json"
        seq1_path.write_text(json.dumps(seq1_outputs, indent=2))
        logger.info(f"\nSaved Seq 1 results to {seq1_path}")

    if seq3_outputs:
        seq3_path = RESULTS_DIR / "seq3_results.json"
        seq3_path.write_text(json.dumps(seq3_outputs, indent=2))
        logger.info(f"Saved Seq 3 results to {seq3_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary_lines = [
        "Redesigned Image Query Evaluation Summary",
        f"Date: {datetime.now(timezone.utc).isoformat()}",
        f"Total redesigned queries: {len(redesigned_queries)}",
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

    summary_path = RESULTS_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    logger.info(f"Saved summary to {summary_path}")

    # -----------------------------------------------------------------------
    # Step 5: Old vs new comparison
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Comparing Original vs Redesigned queries")
    logger.info("=" * 60)

    original_outputs = _load_original_results()
    if original_outputs:
        comparison = build_comparison_table(original_outputs, all_outputs)
        print("\n" + comparison)

        comparison_path = RESULTS_DIR / "old_vs_new_comparison.txt"
        comparison_path.write_text(comparison)
        logger.info(f"Saved comparison to {comparison_path}")
    else:
        logger.warning("No original results found — skipping comparison")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
