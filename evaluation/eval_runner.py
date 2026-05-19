"""CLI entrypoint for the retrieval-only evaluation harness.

Usage:
    python -m evaluation.eval_runner \
        --combined-dataset datasets/combined_eval_dataset.json \
        --image-dataset datasets/image_eval_dataset.json \
        --gcs-bucket my-raptor-bucket \
        --cache-key abc123... \
        --sequences 1,2,3 \
        --top-k 1,3,5,10
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluation.retrieval_evaluator import (
    evaluate_ablation,
    run_statistical_comparisons,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset normalization
# ---------------------------------------------------------------------------


def _parse_page_reference(page_ref: str) -> list[int]:
    """Extract absolute page numbers from a page_reference string.

    Handles formats:
      - "109"           → [109]
      - "6.3.10 p2"     → [] (relative to section, not absolute)
      - "77-5"          → [] (chapter-page, not absolute)
      - "28-5 to 28-6"  → [] (chapter-page range, not absolute)

    Only returns pages when they are clearly absolute PDF page numbers.
    Relative/chapter-page references cannot be resolved without a mapping
    table, so they return empty — the evaluator will rely on exact chunk ID
    matching (from ablation-specific anchored datasets) for those queries.
    """
    if not page_ref:
        return []

    page_ref = page_ref.strip()

    # Plain integer → absolute page
    if page_ref.isdigit():
        return [int(page_ref)]

    # "section p#" format (e.g., "6.3.10 p2") → relative, skip
    if re.match(r"[\d.]+\s+p\d+", page_ref):
        return []

    # Range "CH-P to CH-P" → chapter-page range, skip
    if re.match(r"\d+-\d+\s+to\s+\d+-\d+", page_ref):
        return []

    # "chapter-page" format (e.g., "77-5") → relative, skip
    if re.match(r"\d+-\d+$", page_ref):
        return []

    # Fallback: try plain int parse
    try:
        return [int(page_ref)]
    except (ValueError, TypeError):
        return []


def normalize_combined_query(raw: dict) -> dict:
    """Normalize a combined_eval_dataset entry to the internal query format."""
    chunk_id = raw.get("ground_truth_chunk_id")
    relevant_chunks = (
        [{"chunk_id": chunk_id, "relevance": 2}] if chunk_id else []
    )

    page_ref = raw.get("page_reference", "")
    relevant_pages = _parse_page_reference(page_ref)

    return {
        "query_id": raw["id"],
        "query_text": raw["observation"],
        "query_type": raw.get("origin", "unknown"),
        "relevant_chunks": relevant_chunks,
        "relevant_pages": relevant_pages,
        "answer_text": raw.get("ground_truth_fault_code", ""),
        "difficulty": raw.get("difficulty_tier", "unknown"),
        "manual_source": raw.get("manual_source", "unknown"),
    }


def normalize_image_query(raw: dict) -> dict:
    """Normalize an image_eval_dataset entry to the internal query format."""
    fig_ref = raw.get("figure_reference") or {}
    page = fig_ref.get("page")
    relevant_pages = [int(page)] if page is not None else []

    return {
        "query_id": raw["id"],
        "query_text": raw["observation"],
        "query_type": "image",
        "relevant_chunks": [],
        "relevant_pages": relevant_pages,
        "answer_text": raw.get("ground_truth_fault_code", ""),
        "difficulty": raw.get("difficulty_tier", "unknown"),
        "manual_source": raw.get("manual_source", "unknown"),
    }


def filter_dataset_by_document(dataset: dict, manual_source: str) -> dict:
    """Filter a normalized dataset to only queries for a specific document."""
    filtered = [q for q in dataset["queries"] if q.get("manual_source") == manual_source]
    return {
        "metadata": dataset["metadata"],
        "queries": filtered,
    }


def load_and_merge_datasets(
    combined_path: Path | None,
    image_path: Path | None,
) -> dict:
    """Load, normalize, and merge dataset files into the evaluator format.

    Returns:
        {"metadata": {...}, "queries": [normalized_query, ...]}
    """
    queries: list[dict] = []

    if combined_path:
        raw_combined = json.loads(combined_path.read_text())
        for entry in raw_combined:
            queries.append(normalize_combined_query(entry))
        logger.info(f"Loaded {len(raw_combined)} queries from combined dataset")

    if image_path:
        raw_image = json.loads(image_path.read_text())
        for entry in raw_image:
            queries.append(normalize_image_query(entry))
        logger.info(f"Loaded {len(raw_image)} queries from image dataset")

    # Check for duplicate IDs
    seen_ids = set()
    for q in queries:
        if q["query_id"] in seen_ids:
            logger.warning(f"Duplicate query_id: {q['query_id']}")
        seen_ids.add(q["query_id"])

    return {
        "metadata": {"version": "merged"},
        "queries": queries,
    }


# ---------------------------------------------------------------------------
# Per-ablation dataset resolution
# ---------------------------------------------------------------------------

# Maps ablation key → file suffix used by reanchor_per_ablation.py
_ABLATION_KEY_TO_FILE_SUFFIX = {
    "full": "full",
    "no_table_pc": "no_table_pc",
    "no_header_prop": "no_header_prop",
    "no_caption_fold": "no_caption_fold",
    "no_context_aware": "no_context_aware",
    "semantic_chunking": "semantic_chunking",
    "flat_retrieval": "flat_retrieval",
    "text_only_raptor": "text_only_raptor",
}


def resolve_ablation_dataset(
    ablation_name: str,
    dataset_dir: Path | None,
    fallback_dataset: dict,
    image_path: Path | None = None,
    manual_source: str | None = None,
) -> dict:
    """Load the per-ablation anchored dataset if available, else use fallback.

    Looks for: {dataset_dir}/combined_eval_dataset_anchored_{suffix}.json
    When manual_source is provided, filters to only that document's queries.
    """
    if dataset_dir is None:
        ds = fallback_dataset
    else:
        suffix = _ABLATION_KEY_TO_FILE_SUFFIX.get(ablation_name)
        if suffix is None:
            logger.warning(f"No dataset suffix for ablation '{ablation_name}', using fallback")
            ds = fallback_dataset
        else:
            ablation_file = dataset_dir / f"combined_eval_dataset_anchored_{suffix}.json"
            if not ablation_file.exists():
                logger.warning(
                    f"Per-ablation dataset NOT FOUND for {ablation_name} at "
                    f"{ablation_file.resolve()}, falling back to generic dataset. "
                    f"Ground-truth chunk IDs may not match this tree!"
                )
                ds = fallback_dataset
            else:
                logger.info(f"Loading per-ablation dataset: {ablation_file}")
                raw = json.loads(ablation_file.read_text())

                queries: list[dict] = []
                for entry in raw:
                    queries.append(normalize_combined_query(entry))

                # Merge image queries if available
                if image_path and image_path.exists():
                    raw_image = json.loads(image_path.read_text())
                    for entry in raw_image:
                        queries.append(normalize_image_query(entry))

                ds = {
                    "metadata": {"version": f"per_ablation_{suffix}"},
                    "queries": queries,
                }

    if manual_source:
        ds = filter_dataset_by_document(ds, manual_source)

    return ds


# ---------------------------------------------------------------------------
# Sequence definitions
# ---------------------------------------------------------------------------

SEQ1_COMPARISONS = [
    ("text_only_raptor", "original_raptor_text_only"),
    ("full", "full_context_aware"),
]

SEQ2_COMPARISONS = [
    ("semantic_chunking", "semantic_chunking_baseline"),
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
# Output formatting
# ---------------------------------------------------------------------------


def _format_table_header(title: str, run_id: str, dataset_version: str, num_queries: int) -> str:
    width = 75
    lines = [
        "\u2550" * width,
        title,
        f"Run: {run_id} | Dataset v{dataset_version} | {num_queries} queries",
        "\u2550" * width,
    ]
    return "\n".join(lines)


def _format_metric_row(label: str, metrics: dict, k_values: list[int]) -> str:
    max_k = max(k_values)
    cols = []
    for k in [1, 5, 10]:
        if k in k_values:
            cols.append(f"{metrics.get(f'recall@{k}', 0.0):7.3f}")
    cols.append(f"{metrics.get('mrr', 0.0):7.3f}")
    cols.append(f"{metrics.get(f'ndcg@{max_k}', 0.0):7.3f}")
    return f"  {label:<26s}" + "".join(cols)


def _format_delta_row(label: str, metrics_a: dict, metrics_b: dict, k_values: list[int]) -> str:
    max_k = max(k_values)
    cols = []
    for k in [1, 5, 10]:
        if k in k_values:
            delta = metrics_b.get(f"recall@{k}", 0) - metrics_a.get(f"recall@{k}", 0)
            cols.append(f"{delta:+7.3f}")
    delta_mrr = metrics_b.get("mrr", 0) - metrics_a.get("mrr", 0)
    delta_ndcg = metrics_b.get(f"ndcg@{max_k}", 0) - metrics_a.get(f"ndcg@{max_k}", 0)
    cols.append(f"{delta_mrr:+7.3f}")
    cols.append(f"{delta_ndcg:+7.3f}")
    return f"  {label:<26s}" + "".join(cols)


def _format_stat_row(stat_tests: dict, label_a: str, label_b: str, scope: str, k_values: list[int]) -> str:
    max_k = max(k_values)
    cols = []
    for k in [1, 5, 10]:
        if k in k_values:
            key = f"{label_b}_vs_{label_a}_{scope}_recall@{k}"
            p = stat_tests.get(key, {}).get("p_value", float("nan"))
            cols.append(f"{p:7.3f}")
    key_mrr = f"{label_b}_vs_{label_a}_{scope}_mrr"
    p_mrr = stat_tests.get(key_mrr, {}).get("p_value", float("nan"))
    cols.append(f"{p_mrr:7.3f}")
    key_ndcg = f"{label_b}_vs_{label_a}_{scope}_ndcg@{max_k}"
    p_ndcg = stat_tests.get(key_ndcg, {}).get("p_value", float("nan"))
    cols.append(f"{p_ndcg:7.3f}")
    return f"  {'p-value (bootstrap)':<26s}" + "".join(cols)


def print_comparison_table(
    seq_num: int,
    title: str,
    run_id: str,
    dataset_version: str,
    results: dict[str, dict],
    stat_tests: dict,
    k_values: list[int],
    comparisons: list[tuple[str, str]],
):
    """Print a formatted comparison table to stdout."""
    max_k = max(k_values)
    num_queries = results[comparisons[0][1]]["overall"]["num_queries"]
    header_labels = []
    for k in [1, 5, 10]:
        if k in k_values:
            header_labels.append(f"Recall@{k}")
    header_labels.extend(["MRR", f"NDCG@{max_k}"])
    col_header = "  " + " " * 26 + "".join(f"{h:>7s}" for h in header_labels)

    print()
    print(_format_table_header(f"Seq {seq_num}: {title}", run_id, dataset_version, num_queries))
    print()
    print(col_header)
    print("\u2500" * 75)

    # Determine baseline and proposed
    baseline_name, baseline_label = comparisons[0]
    proposed_name, proposed_label = comparisons[-1]

    # OVERALL
    print(f"OVERALL (n={num_queries})")
    for ablation_name, ablation_label in comparisons:
        metrics = results[ablation_label]["overall"]
        print(_format_metric_row(ablation_label, metrics, k_values))

    if len(comparisons) == 2:
        print(_format_delta_row(
            f"\u0394 ({proposed_label[:12]} - {baseline_label[:12]})",
            results[baseline_label]["overall"],
            results[proposed_label]["overall"],
            k_values,
        ))
        print(_format_stat_row(stat_tests, baseline_label, proposed_label, "overall", k_values))

    # By query type
    query_types = sorted(results[comparisons[0][1]].get("by_query_type", {}).keys())
    for qt in query_types:
        n_type = results[comparisons[0][1]]["by_query_type"][qt]["num_queries"]
        print(f"\n{qt.upper()} (n={n_type})")
        for ablation_name, ablation_label in comparisons:
            type_metrics = results[ablation_label].get("by_query_type", {}).get(qt, {})
            if type_metrics:
                print(_format_metric_row(ablation_label, type_metrics, k_values))
        if len(comparisons) == 2:
            m_a = results[baseline_label].get("by_query_type", {}).get(qt, {})
            m_b = results[proposed_label].get("by_query_type", {}).get(qt, {})
            if m_a and m_b:
                print(_format_delta_row("\u0394", m_a, m_b, k_values))

    print("\u2550" * 75)


def print_contribution_ranking(results: dict[str, dict], k_values: list[int]):
    """Print sub-innovation contribution ranking for Seq 3."""
    max_k = max(k_values)
    ndcg_key = f"ndcg@{max_k}"

    full_metrics = results["full_context_aware"]
    full_ndcg = full_metrics["overall"].get(ndcg_key, 0)

    contributions = []
    ablation_info = {
        "no_table_parent_child": ("2a", "table_parent_child"),
        "no_header_propagation": ("2b", "header_propagation"),
        "no_caption_folding": ("2c", "caption_folding"),
    }

    for ablation_label, (code, name) in ablation_info.items():
        if ablation_label not in results:
            continue
        ablated_ndcg = results[ablation_label]["overall"].get(ndcg_key, 0)
        drop = full_ndcg - ablated_ndcg

        # Find which query type has the largest drop
        max_type_drop = 0.0
        max_type_name = ""
        for qt in results[ablation_label].get("by_query_type", {}):
            full_type_ndcg = full_metrics.get("by_query_type", {}).get(qt, {}).get(ndcg_key, 0)
            abl_type_ndcg = results[ablation_label]["by_query_type"][qt].get(ndcg_key, 0)
            type_drop = full_type_ndcg - abl_type_ndcg
            if type_drop > max_type_drop:
                max_type_drop = type_drop
                max_type_name = qt.upper()

        contributions.append((drop, code, name, max_type_drop, max_type_name))

    contributions.sort(key=lambda x: x[0], reverse=True)

    print(f"\nSub-innovation Contribution Ranking (by {ndcg_key} drop when removed):")
    for rank, (drop, code, name, type_drop, type_name) in enumerate(contributions, 1):
        detail = f"[largest drop on {type_name} queries: {-type_drop:.3f}]" if type_name else ""
        print(f"  {rank}. {code} ({name}): {-drop:+.3f}  {detail}")


# ---------------------------------------------------------------------------
# Sequence runners
# ---------------------------------------------------------------------------


def run_sequence(
    seq_num: int,
    dataset: dict,
    cache_key: str,
    bucket: str,
    run_id: str | None,
    k_values: list[int],
    dataset_dir: Path | None = None,
    image_path: Path | None = None,
    retrieval_mode: str = "embedding_only",
    k_prime: int = 30,
    manual_source: str | None = None,
    doc_label: str | None = None,
) -> dict | list[dict]:
    """Run a single evaluation sequence and return results dict.

    When dataset_dir is provided, each ablation loads its own anchored
    dataset (produced by reanchor_per_ablation.py). This ensures chunk IDs
    in the ground truth match the tree being evaluated.

    When manual_source is provided, queries are filtered to that document only.
    """
    if seq_num == 1:
        comparisons = SEQ1_COMPARISONS
        title = "Multimodal RAPTOR vs Original RAPTOR"
    elif seq_num == 2:
        comparisons = SEQ2_COMPARISONS
        title = "Context-Aware Chunking vs Semantic Chunking"
    elif seq_num == 3:
        comparisons = SEQ3_ABLATIONS
        title = "Ablation on 2a, 2b, 2c"
    else:
        raise ValueError(f"Unknown sequence: {seq_num}")

    modes_to_run = (
        ["embedding_only", "embedding_reranked"]
        if retrieval_mode == "both"
        else [retrieval_mode]
    )

    outputs = []
    for mode in modes_to_run:
        reranker_instance = None
        if mode == "embedding_reranked":
            from evaluation.reranker import EvalReranker
            reranker_instance = EvalReranker(min_score=3, top_k=max(k_values))

        max_k = max(k_values)
        results: dict[str, dict] = {}
        all_scores: dict[str, dict[str, dict[str, list[float]]]] = {}

        for ablation_name, ablation_label in comparisons:
            logger.info(f"[Seq {seq_num}] [{mode}] Evaluating: {ablation_label}")

            ablation_dataset = resolve_ablation_dataset(
                ablation_name, dataset_dir, dataset, image_path=image_path,
                manual_source=manual_source,
            )

            metrics, scores = evaluate_ablation(
                cache_key=cache_key,
                ablation_name=ablation_name,
                dataset=ablation_dataset,
                k_values=k_values,
                run_id=run_id,
                bucket=bucket,
                k_prime=k_prime,
                retrieval_mode=mode,
                reranker=reranker_instance,
            )
            results[ablation_label] = metrics
            all_scores[ablation_label] = scores

        # Statistical tests
        stat_tests = {}
        metrics_to_test = [f"recall@{k}" for k in k_values] + ["mrr", f"ndcg@{max_k}"]

        if seq_num in (1, 2):
            baseline_label = comparisons[0][1]
            proposed_label = comparisons[-1][1]
            stat_tests = run_statistical_comparisons(
                all_scores[baseline_label],
                all_scores[proposed_label],
                baseline_label,
                proposed_label,
                metrics_to_test,
            )
        elif seq_num == 3:
            full_label = comparisons[0][1]
            for ablation_name, ablation_label in comparisons[1:]:
                tests = run_statistical_comparisons(
                    all_scores[full_label],
                    all_scores[ablation_label],
                    full_label,
                    ablation_label,
                    metrics_to_test,
                )
                stat_tests.update(tests)

        mode_label = "embedding_only" if mode == "embedding_only" else "embedding_reranked"
        doc_suffix = f" \u2014 {doc_label}" if doc_label else ""
        print_comparison_table(
            seq_num, f"{title} [{mode_label}]{doc_suffix}", run_id or "latest",
            dataset.get("metadata", {}).get("version", "unknown"),
            results, stat_tests, k_values, comparisons,
        )

        if seq_num == 3:
            print_contribution_ranking(results, k_values)

        output = {
            "sequence": seq_num,
            "document": doc_label or "all",
            "retrieval_mode": mode,
            "k_prime": k_prime,
            "run_id": run_id or "latest",
            "dataset_version": dataset.get("metadata", {}).get("version", "unknown"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "k_values": k_values,
            "results": results,
            "statistical_tests": stat_tests,
        }
        outputs.append(output)

    return outputs[0] if len(outputs) == 1 else outputs


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


def save_results(output: dict, results_dir: Path, run_id: str | None):
    """Save sequence results to JSON. Never overwrites — uses timestamp subdir."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rid = run_id or "latest"
    out_dir = results_dir / f"{rid}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_num = output["sequence"]
    out_file = out_dir / f"seq{seq_num}_results.json"

    # Custom serializer for sets
    def _default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    out_file.write_text(json.dumps(output, indent=2, default=_default))
    logger.info(f"Results saved to {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="RAPTOR retrieval-only evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--combined-dataset", type=Path, default=None,
        help="Path to combined_eval_dataset.json (text queries)",
    )
    parser.add_argument(
        "--image-dataset", type=Path, default=None,
        help="Path to image_eval_dataset.json (image queries)",
    )
    parser.add_argument(
        "--gcs-bucket", required=True,
        help="GCS bucket containing tree artifacts",
    )
    parser.add_argument(
        "--cache-key", default=None,
        help="Content-hash cache key for a single document (legacy, prefer --cache-keys)",
    )
    parser.add_argument(
        "--cache-keys", default=None,
        help="manual_source:cache_key pairs for multi-document eval "
             "(e.g., 'SC10000AMM:c313...,AS-AMM-01-000:112e...')",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Pin to a specific ingestion run_id (default: latest)",
    )
    parser.add_argument(
        "--sequences", default="1,2,3",
        help="Comma-separated sequence numbers to run (default: 1,2,3)",
    )
    parser.add_argument(
        "--top-k", default="1,3,5,10",
        help="Comma-separated k values for metrics (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Output directory for results (default: evaluation/results/)",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Directory containing per-ablation anchored datasets "
             "(e.g., combined_eval_dataset_anchored_full.json). "
             "When set, each ablation loads its own ground-truth file.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["embedding_only", "embedding_reranked", "both"],
        default="embedding_only",
        help="Retrieval pipeline mode (default: embedding_only)",
    )
    parser.add_argument(
        "--k-prime", type=int, default=30,
        help="Initial cosine candidate pool size (default: 30)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse arguments
    sequences = [int(s.strip()) for s in args.sequences.split(",")]
    k_values = [int(k.strip()) for k in args.top_k.split(",")]

    results_dir = args.results_dir or Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Parse cache-key(s)
    if args.cache_keys:
        doc_to_cache_key = {}
        for pair in args.cache_keys.split(","):
            parts = pair.strip().split(":")
            if len(parts) != 2:
                logger.error(f"Invalid cache-key pair: {pair}")
                sys.exit(1)
            doc_to_cache_key[parts[0].strip()] = parts[1].strip()
    elif args.cache_key:
        doc_to_cache_key = {"all": args.cache_key}
    else:
        logger.error("Either --cache-key or --cache-keys is required")
        sys.exit(1)

    # Load and merge datasets
    if not args.combined_dataset and not args.image_dataset:
        logger.error("At least one of --combined-dataset or --image-dataset is required")
        sys.exit(1)

    dataset = load_and_merge_datasets(args.combined_dataset, args.image_dataset)
    num_queries = len(dataset["queries"])
    logger.info(f"Merged dataset: {num_queries} queries")

    # Resolve dataset directory for per-ablation anchoring
    dataset_dir = args.dataset_dir
    if dataset_dir is None and args.combined_dataset:
        candidate_dir = args.combined_dataset.parent
        if (candidate_dir / "combined_eval_dataset_anchored_full.json").exists():
            dataset_dir = candidate_dir
            logger.info(f"Auto-detected per-ablation datasets in {dataset_dir}")

    # Run sequences per document
    for doc_label, cache_key in doc_to_cache_key.items():
        manual_source = doc_label if doc_label != "all" else None

        if manual_source:
            doc_queries = [q for q in dataset["queries"] if q.get("manual_source") == manual_source]
            logger.info(f"\n{'#'*60}")
            logger.info(f"Document: {doc_label} ({len(doc_queries)} queries, cache_key={cache_key[:12]}...)")
            logger.info(f"{'#'*60}")
        else:
            logger.info(f"Single-document mode: cache_key={cache_key[:12]}...")

        for seq_num in sequences:
            logger.info(f"{'='*60}")
            logger.info(f"Running Sequence {seq_num} — {doc_label}")
            logger.info(f"{'='*60}")

            output = run_sequence(
                seq_num=seq_num,
                dataset=dataset,
                cache_key=cache_key,
                bucket=args.gcs_bucket,
                run_id=args.run_id,
                k_values=k_values,
                dataset_dir=dataset_dir,
                image_path=args.image_dataset,
                retrieval_mode=args.retrieval_mode,
                k_prime=args.k_prime,
                manual_source=manual_source,
                doc_label=doc_label,
            )

            doc_results_dir = results_dir / doc_label if manual_source else results_dir
            if isinstance(output, list):
                for o in output:
                    save_results(o, doc_results_dir, args.run_id)
            else:
                save_results(output, doc_results_dir, args.run_id)

    logger.info("All sequences complete.")


if __name__ == "__main__":
    main()
