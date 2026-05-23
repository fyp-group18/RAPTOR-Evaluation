"""Re-run CR significance tests against the correct proposed system tree.

Pins run_ids to the trees that produced the paper's reported metrics,
rather than resolving "latest" which may pick a different tree.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

GCS_BUCKET = "raptor-assets"

DOC_MAP = {
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
}

# Pin to the run_ids that produced the paper's reported metrics
PAPER_RUN_IDS = {
    "AS-AMM-01-000": "20260517_075801_9044c50d",
    "SC10000AMM": "20260517_075856_639c04b2",
}

# Expected paper metrics for verification (NDCG@10)
EXPECTED_NDCG = {
    "AS-AMM-01-000": {"full_context_aware": 0.445, "baseline_naive_chunking": None, "no_header_propagation": None},
    "SC10000AMM": {"full_context_aware": 0.600, "baseline_naive_chunking": None, "no_header_propagation": None},
}

DATASET_DIR = PROJECT_ROOT / "datasets"
COMBINED_DATASET = DATASET_DIR / "combined_eval_dataset_anchored.json"
IMAGE_DATASET = DATASET_DIR / "image_eval_dataset.json"

K_VALUES = [1, 3, 5, 10]
K_PRIME = 30

ABLATIONS_TO_EVAL = [
    ("no_context_aware", "baseline_naive_chunking"),
    ("contextual_retrieval", "contextual_retrieval"),
    ("no_header_prop", "no_header_propagation"),
    ("full", "full_context_aware"),
]

SIGNIFICANCE_PAIRS = [
    ("baseline_naive_chunking", "contextual_retrieval", "CR vs naive baseline"),
    ("contextual_retrieval", "full_context_aware", "CR vs proposed system"),
    ("no_header_propagation", "contextual_retrieval", "CR vs no_header_prop"),
]


def main():
    from evaluation.eval_runner import (
        load_and_merge_datasets,
        resolve_ablation_dataset,
    )
    from evaluation.retrieval_evaluator import (
        evaluate_ablation,
        run_statistical_comparisons,
    )

    dataset = load_and_merge_datasets(COMBINED_DATASET, IMAGE_DATASET)
    max_k = max(K_VALUES)
    ndcg_key = f"ndcg@{max_k}"

    all_output: list[str] = []
    all_output.append("CORRECTED SIGNIFICANCE TESTS (paired bootstrap, 10K iter, seed 42)")
    all_output.append("=" * 68)

    for source, cache_key in DOC_MAP.items():
        run_id = PAPER_RUN_IDS[source]
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING: {source} (pinned run_id={run_id})")
        logger.info(f"{'='*60}")

        doc_results: dict[str, dict] = {}
        doc_scores: dict[str, dict] = {}

        for ablation_name, ablation_label in ABLATIONS_TO_EVAL:
            logger.info(f"  Evaluating: {ablation_label}")

            ablation_dataset = resolve_ablation_dataset(
                ablation_name, DATASET_DIR, dataset,
                image_path=IMAGE_DATASET, manual_source=source,
            )

            # Pin full_context_aware to the paper run_id; others use latest
            eval_run_id = run_id if ablation_name == "full" else None

            metrics, scores = evaluate_ablation(
                cache_key=cache_key,
                ablation_name=ablation_name,
                dataset=ablation_dataset,
                k_values=K_VALUES,
                run_id=eval_run_id,
                bucket=GCS_BUCKET,
                k_prime=K_PRIME,
                retrieval_mode="embedding_only",
                reranker=None,
            )
            doc_results[ablation_label] = metrics
            doc_scores[ablation_label] = scores

            ndcg = metrics["overall"].get(ndcg_key, 0)
            logger.info(f"    NDCG@{max_k} = {ndcg:.3f}")

        # Verify paper metrics
        all_output.append(f"\n{source} (pinned run_id={run_id}):")
        for ablation_label in ["full_context_aware", "baseline_naive_chunking",
                                "no_header_propagation", "contextual_retrieval"]:
            m = doc_results[ablation_label]["overall"]
            n = m.get("num_queries", 0)
            all_output.append(
                f"  {ablation_label:<26s}  n={n:>3d}  "
                f"R@1={m.get('recall@1',0):.3f}  R@5={m.get('recall@5',0):.3f}  "
                f"R@10={m.get('recall@10',0):.3f}  MRR={m.get('mrr',0):.3f}  "
                f"NDCG={m.get(ndcg_key,0):.3f}"
            )

        expected = EXPECTED_NDCG[source]["full_context_aware"]
        actual = doc_results["full_context_aware"]["overall"].get(ndcg_key, 0)
        if abs(actual - expected) > 0.002:
            all_output.append(f"  *** WARNING: full NDCG={actual:.3f} != expected {expected:.3f} ***")

        # Significance tests
        metrics_to_test = [f"recall@{k}" for k in K_VALUES] + ["mrr", ndcg_key]
        stat_tests = {}
        for base_label, prop_label, desc in SIGNIFICANCE_PAIRS:
            if base_label in doc_scores and prop_label in doc_scores:
                tests = run_statistical_comparisons(
                    doc_scores[base_label], doc_scores[prop_label],
                    base_label, prop_label, metrics_to_test,
                )
                stat_tests.update(tests)

        all_output.append(f"\n  Significance tests (NDCG@{max_k}):")
        for base_label, prop_label, desc in SIGNIFICANCE_PAIRS:
            p_key = f"{prop_label}_vs_{base_label}_overall_{ndcg_key}"
            result = stat_tests.get(p_key, {})
            p_val = result.get("p_value", float("nan"))
            base_ndcg = doc_results[base_label]["overall"].get(ndcg_key, 0)
            prop_ndcg = doc_results[prop_label]["overall"].get(ndcg_key, 0)
            delta = prop_ndcg - base_ndcg
            sig = ""
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            all_output.append(f"    {desc:<30s}  Δ={delta:+.3f}  p={p_val:.3f} {sig}")

    all_output.append(f"\n{'='*68}")

    # Print and save
    for line in all_output:
        print(line)

    out_path = PROJECT_ROOT / "evaluation" / "results" / "cr_significance_corrected.txt"
    out_path.write_text("\n".join(all_output))
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
