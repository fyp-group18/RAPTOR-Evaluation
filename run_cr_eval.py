"""Contextual Retrieval baseline evaluation.

Evaluates the CR ablation tree against comparison baselines and computes
paired bootstrap significance tests. Outputs paper-ready formatted results.

Comparisons:
  - CR vs naive baseline (no_context_aware): Does LLM-generated context help?
  - CR vs proposed system (full): Does the full system beat CR-only?
  - CR vs no_header_prop: CR context vs structural P-C+captions without headers

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
    GOOGLE_CLOUD_PROJECT=<project> \
    PYTHONPATH="/path/to/backend:." \
    python run_cr_eval.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TREE_CACHE_DIR = PROJECT_ROOT / ".tree-cache"

GCS_BUCKET = "raptor-assets"

DOC_MAP = {
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
}

COMBINED_DATASET = PROJECT_ROOT / "datasets" / "combined_eval_dataset_anchored.json"
IMAGE_DATASET = PROJECT_ROOT / "datasets" / "image_eval_dataset.json"
DATASET_DIR = PROJECT_ROOT / "datasets"

K_VALUES = [1, 3, 5, 10]
K_PRIME = 30

# Ablations to evaluate and compare
CR_ABLATION = ("contextual_retrieval", "contextual_retrieval")
COMPARISONS = [
    ("no_context_aware", "baseline_naive_chunking"),
    ("contextual_retrieval", "contextual_retrieval"),
    ("no_header_prop", "no_header_propagation"),
    ("full", "full_context_aware"),
]

SIGNIFICANCE_PAIRS = [
    # (baseline_key, baseline_label, proposed_key, proposed_label, description)
    ("no_context_aware", "baseline_naive_chunking", "contextual_retrieval", "contextual_retrieval",
     "CR vs naive baseline"),
    ("contextual_retrieval", "contextual_retrieval", "full", "full_context_aware",
     "Proposed system vs CR"),
    ("no_header_prop", "no_header_propagation", "contextual_retrieval", "contextual_retrieval",
     "CR vs no_header_prop"),
]


def verify_cr_trees() -> bool:
    """Check that contextual_retrieval trees exist for both documents."""
    all_present = True
    for source, cache_key in DOC_MAP.items():
        cache_dir = TREE_CACHE_DIR / cache_key
        if not cache_dir.exists():
            logger.error(f"[{source}] No tree cache directory found")
            all_present = False
            continue

        found = False
        for run_dir in cache_dir.iterdir():
            if not run_dir.is_dir():
                continue
            tree_file = run_dir / "contextual_retrieval" / "tree.pkl"
            if tree_file.exists():
                found = True
                break

        if not found:
            logger.error(f"[{source}] contextual_retrieval tree not found")
            all_present = False
        else:
            logger.info(f"[{source}] contextual_retrieval tree found")

    return all_present


def run_cr_evaluation():
    """Run CR evaluation for both documents."""
    from evaluation.eval_runner import (
        load_and_merge_datasets,
        resolve_ablation_dataset,
        save_results,
    )
    from evaluation.retrieval_evaluator import (
        evaluate_ablation,
        run_statistical_comparisons,
    )

    dataset = load_and_merge_datasets(COMBINED_DATASET, IMAGE_DATASET)
    max_k = max(K_VALUES)

    all_results = {}

    for source, cache_key in DOC_MAP.items():
        logger.info(f"\n{'='*72}")
        logger.info(f"EVALUATING: {source}")
        logger.info(f"{'='*72}")

        doc_results: dict[str, dict] = {}
        doc_scores: dict[str, dict] = {}

        for ablation_name, ablation_label in COMPARISONS:
            logger.info(f"  Evaluating: {ablation_label}")

            ablation_dataset = resolve_ablation_dataset(
                ablation_name, DATASET_DIR, dataset,
                image_path=IMAGE_DATASET,
                manual_source=source,
            )

            metrics, scores = evaluate_ablation(
                cache_key=cache_key,
                ablation_name=ablation_name,
                dataset=ablation_dataset,
                k_values=K_VALUES,
                run_id=None,
                bucket=GCS_BUCKET,
                k_prime=K_PRIME,
                retrieval_mode="embedding_only",
                reranker=None,
            )
            doc_results[ablation_label] = metrics
            doc_scores[ablation_label] = scores

        # Statistical tests
        metrics_to_test = [f"recall@{k}" for k in K_VALUES] + ["mrr", f"ndcg@{max_k}"]
        stat_tests = {}
        for base_key, base_label, prop_key, prop_label, desc in SIGNIFICANCE_PAIRS:
            if base_label in doc_scores and prop_label in doc_scores:
                tests = run_statistical_comparisons(
                    doc_scores[base_label],
                    doc_scores[prop_label],
                    base_label,
                    prop_label,
                    metrics_to_test,
                )
                stat_tests.update(tests)

        all_results[source] = {
            "results": doc_results,
            "stat_tests": stat_tests,
        }

        # Save results
        results_dir = PROJECT_ROOT / "evaluation" / "results" / source / "cr_baseline"
        results_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "experiment": "contextual_retrieval_baseline",
            "document": source,
            "cache_key": cache_key,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "k_values": K_VALUES,
            "k_prime": K_PRIME,
            "results": doc_results,
            "statistical_tests": stat_tests,
        }
        out_file = results_dir / "cr_eval_results.json"

        def _default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        out_file.write_text(json.dumps(output, indent=2, default=_default))
        logger.info(f"  Results saved to {out_file}")

    return all_results


def print_formatted_results(all_results: dict):
    """Print paper-ready formatted output."""
    max_k = max(K_VALUES)
    ndcg_key = f"ndcg@{max_k}"

    print("\n")
    print("=" * 72)
    print("CONTEXTUAL RETRIEVAL BASELINE RESULTS")
    print("=" * 72)

    for source, data in all_results.items():
        results = data["results"]
        stat_tests = data["stat_tests"]

        cr_metrics = results.get("contextual_retrieval", {})
        cr_overall = cr_metrics.get("overall", {})
        n_queries = cr_overall.get("num_queries", 0)

        print(f"\nOverall ({source}, n={n_queries}):")
        print(f"  {'':26s}  R@1    R@5    R@10   MRR    NDCG@{max_k}")
        print(f"  {'-'*65}")

        for _, ablation_label in COMPARISONS:
            m = results.get(ablation_label, {}).get("overall", {})
            if not m:
                continue
            r1 = m.get("recall@1", 0)
            r5 = m.get("recall@5", 0)
            r10 = m.get("recall@10", 0)
            mrr = m.get("mrr", 0)
            ndcg = m.get(ndcg_key, 0)
            print(f"  {ablation_label:<26s}  {r1:.3f}  {r5:.3f}  {r10:.3f}  {mrr:.3f}  {ndcg:.3f}")

        # By query type
        by_type = cr_metrics.get("by_query_type", {})
        if by_type:
            print(f"\n  By query type ({source}):")
            for qt in sorted(by_type.keys()):
                type_m = by_type[qt]
                n_type = type_m.get("num_queries", 0)
                print(
                    f"    {qt.upper()} (n={n_type}):  "
                    f"R@1={type_m.get('recall@1', 0):.3f}  "
                    f"R@5={type_m.get('recall@5', 0):.3f}  "
                    f"MRR={type_m.get('mrr', 0):.3f}  "
                    f"NDCG={type_m.get(ndcg_key, 0):.3f}"
                )

        # Significance tests
        print(f"\n  Significance tests (NDCG@{max_k}, paired bootstrap 10K, seed 42):")
        for base_key, base_label, prop_key, prop_label, desc in SIGNIFICANCE_PAIRS:
            p_key = f"{prop_label}_vs_{base_label}_overall_{ndcg_key}"
            p_val = stat_tests.get(p_key, {}).get("p_value", float("nan"))
            delta_key_base = results.get(base_label, {}).get("overall", {}).get(ndcg_key, 0)
            delta_key_prop = results.get(prop_label, {}).get("overall", {}).get(ndcg_key, 0)
            delta = delta_key_prop - delta_key_base
            sig = ""
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            print(f"    {desc:<30s}  Δ={delta:+.3f}  p={p_val:.3f} {sig}")

    # LaTeX row for paper
    print(f"\n{'─'*72}")
    print("LaTeX table row (contextual_retrieval):")
    print(f"{'─'*72}")
    for source, data in all_results.items():
        cr_m = data["results"].get("contextual_retrieval", {}).get("overall", {})
        if not cr_m:
            continue
        doc_short = {"AS-AMM-01-000": "AS", "SC10000AMM": "SC"}.get(source, source)
        r1 = cr_m.get("recall@1", 0)
        r5 = cr_m.get("recall@5", 0)
        r10 = cr_m.get("recall@10", 0)
        mrr = cr_m.get("mrr", 0)
        ndcg = cr_m.get(ndcg_key, 0)
        print(
            f"  {doc_short} & Contextual Retrieval "
            f"& {r1:.3f} & {r5:.3f} & {r10:.3f} "
            f"& {mrr:.3f} & {ndcg:.3f} \\\\"
        )

    print(f"\n{'='*72}")


def main():
    logger.info("=" * 72)
    logger.info("Contextual Retrieval Baseline Evaluation")
    logger.info(f"Started: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 72)

    # Verify trees exist
    if not verify_cr_trees():
        logger.error(
            "Cannot proceed: contextual_retrieval tree missing for one or more documents. "
            "Run: python -m ingestion.orchestrator --ablations contextual_retrieval"
        )
        sys.exit(1)

    # Run evaluation
    all_results = run_cr_evaluation()

    # Print formatted results
    print_formatted_results(all_results)

    logger.info("\nCR baseline evaluation complete.")


if __name__ == "__main__":
    main()
