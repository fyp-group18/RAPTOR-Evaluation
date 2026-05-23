"""Cloud Run Job worker for RAPTOR evaluation pipeline.

Uses CLOUD_RUN_TASK_INDEX to dispatch work from a pipe-delimited task config:

  TASK_TYPE=rebuild_tree:
    TASK_CONFIGS="cache_key:ablation_name:filename|cache_key:ablation_name:filename|..."
    Each task index picks one entry.

  TASK_TYPE=evaluate:
    EVAL_TASKS="cache_key:doc_source:combined_gcs:image_gcs|..."
    Each task index picks one entry.

Designed for maximum parallelism via Cloud Run Jobs task-level concurrency.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _get_task_index() -> int:
    return int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))


def _parse_task_config(env_var: str) -> list[str]:
    raw = os.environ.get(env_var, "")
    return [t.strip() for t in raw.split("|") if t.strip()]


def rebuild_tree():
    """Rebuild a single ablation tree from cached Docling output in GCS."""
    from ingestion.ablation_configs import ABLATION_CASES
    from ingestion.gcs_cache import upload_tree_artifact, check_docling_cache
    from ingestion.tree_builder import build_tree_for_ablation

    task_idx = _get_task_index()
    configs = _parse_task_config("TASK_CONFIGS")

    if task_idx >= len(configs):
        logger.info(f"Task index {task_idx} exceeds config count {len(configs)}, nothing to do")
        return

    parts = configs[task_idx].split(":")
    cache_key, ablation_name, filename = parts[0], parts[1], parts[2]

    bucket = os.environ["GCS_BUCKET"]
    config = ABLATION_CASES[ablation_name]
    label = config["label"]
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    logger.info(f"[Task {task_idx}] Rebuilding: {label} for {filename}")
    logger.info(f"Cache key: {cache_key[:12]}..., Run ID: {run_id}")

    # Load Docling output from GCS cache
    logger.info("Loading Docling output from GCS cache...")
    docling_output, metadata = check_docling_cache(bucket, cache_key)
    if docling_output is None:
        logger.error(f"Docling cache not found for {cache_key[:12]}!")
        sys.exit(1)

    logger.info("Docling output loaded. Building tree...")
    t0 = time.perf_counter()

    tree_bytes, tree_stats = build_tree_for_ablation(
        docling_output, config, filename, cache_key=cache_key
    )
    elapsed = time.perf_counter() - t0

    nodes = pickle.loads(tree_bytes)
    has_emb = sum(1 for n in nodes if n.get("embedding") is not None)
    logger.info(f"[{label}] Built: {len(nodes)} nodes, {has_emb} with embedding [{elapsed:.1f}s]")

    if len(nodes) == 0:
        logger.error(f"[{label}] Build produced 0 nodes!")
        sys.exit(1)

    manifest = {
        "run_id": run_id,
        "cache_key": cache_key,
        "ablation_label": label,
        "ablation_config": config,
        "source_document": filename,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tree_stats": tree_stats,
        "rebuild": True,
        "runtime": "cloud_run",
    }

    upload_tree_artifact(bucket, cache_key, run_id, label, tree_bytes, manifest)
    logger.info(f"[{label}] Uploaded to GCS + local mirror ✓")


def reanchor():
    """Re-anchor eval dataset against fresh full_context_aware trees.

    Downloads raw dataset from GCS, resolves the latest full_context_aware
    tree for each document (populating local mirror), runs reanchoring,
    splits by document source, and uploads per-doc datasets to GCS.
    """
    from google.cloud import storage as gcs_storage

    from ingestion.reanchor_per_ablation import reanchor_for_ablation
    from ingestion.tree_resolver import resolve_tree

    bucket = os.environ["GCS_BUCKET"]
    cache_keys_raw = os.environ["CACHE_KEYS"]
    raw_combined_gcs = os.environ.get(
        "RAW_COMBINED_GCS", "eval-datasets/combined_eval_dataset.json"
    )
    raw_image_gcs = os.environ.get(
        "RAW_IMAGE_GCS", "eval-datasets/image_eval_dataset.json"
    )

    client = gcs_storage.Client()
    bucket_obj = client.bucket(bucket)

    # Parse cache keys: "AS-AMM-01-000:112e...|SC10000AMM:c313..."
    manual_to_cache_key = {}
    for pair in cache_keys_raw.split("|"):
        manual, key = pair.strip().split(":")
        manual_to_cache_key[manual.strip()] = key.strip()
    logger.info(f"Manual->cache_key: {manual_to_cache_key}")

    # Pre-populate local mirror by resolving trees from GCS
    for manual, cache_key in manual_to_cache_key.items():
        logger.info(f"Resolving full_context_aware tree for {manual}...")
        resolve_tree(cache_key, "full_context_aware", bucket=bucket)
        logger.info(f"  Tree cached locally for {manual}")

    # Download raw datasets
    tmp_dir = Path(tempfile.mkdtemp(prefix="reanchor_"))

    combined_path = tmp_dir / "combined.json"
    bucket_obj.blob(raw_combined_gcs).download_to_filename(str(combined_path))
    with open(combined_path) as f:
        queries = json.load(f)
    logger.info(f"Downloaded {len(queries)} queries from gs://{bucket}/{raw_combined_gcs}")

    # Download image dataset (split by source for later upload)
    image_queries_by_source: dict[str, list[dict]] = {}
    try:
        image_path = tmp_dir / "image.json"
        bucket_obj.blob(raw_image_gcs).download_to_filename(str(image_path))
        with open(image_path) as f:
            image_queries = json.load(f)
        for iq in image_queries:
            source = iq.get("manual_source", "unknown")
            image_queries_by_source.setdefault(source, []).append(iq)
        logger.info(f"Downloaded {len(image_queries)} image queries")
    except Exception as e:
        logger.warning(f"Image dataset not available: {e}")

    # Reanchor against full_context_aware
    anchored, stats = reanchor_for_ablation(
        queries, manual_to_cache_key, "full", "full_context_aware"
    )
    logger.info(f"Reanchoring complete. Confidence: {stats['confidence']}")

    # Split by document source and upload
    by_source: dict[str, list[dict]] = {}
    for q in anchored:
        source = q.get("manual_source", "unknown")
        by_source.setdefault(source, []).append(q)

    source_prefix = {
        "AS-AMM-01-000": "as_amm",
        "SC10000AMM": "sc10k",
    }

    for source, qs in by_source.items():
        pfx = source_prefix.get(source, source.lower().replace("-", "_"))

        out_path = tmp_dir / f"{pfx}_combined.json"
        with open(out_path, "w") as f:
            json.dump(qs, f, indent=2)
        gcs_key = f"eval-datasets/{pfx}_combined.json"
        bucket_obj.blob(gcs_key).upload_from_filename(str(out_path))
        logger.info(f"Uploaded {len(qs)} queries -> gs://{bucket}/{gcs_key}")

        img_qs = image_queries_by_source.get(source, [])
        if img_qs:
            img_path = tmp_dir / f"{pfx}_image.json"
            with open(img_path, "w") as f:
                json.dump(img_qs, f, indent=2)
            img_gcs_key = f"eval-datasets/{pfx}_image.json"
            bucket_obj.blob(img_gcs_key).upload_from_filename(str(img_path))
            logger.info(f"Uploaded {len(img_qs)} image queries -> gs://{bucket}/{img_gcs_key}")

    logger.info("Reanchoring and upload complete.")


def evaluate():
    """Run evaluation sequences for a single document."""
    from google.cloud import storage as gcs_storage

    from evaluation.eval_runner import (
        load_and_merge_datasets,
        run_sequence,
        save_results,
    )

    task_idx = _get_task_index()
    configs = _parse_task_config("EVAL_TASKS")

    if task_idx >= len(configs):
        logger.info(f"Task index {task_idx} exceeds config count {len(configs)}, nothing to do")
        return

    parts = configs[task_idx].split(":")
    cache_key, doc_source, combined_gcs, image_gcs = parts[0], parts[1], parts[2], parts[3]

    bucket = os.environ["GCS_BUCKET"]
    sequences = [int(s) for s in os.environ.get("SEQUENCES", "1,2,3").split(",")]
    k_values = [int(k) for k in os.environ.get("TOP_K", "1,3,5,10").split(",")]

    logger.info(f"[Task {task_idx}] Evaluating: {doc_source} (cache_key={cache_key[:12]}...)")
    logger.info(f"Sequences: {sequences}, k_values: {k_values}")

    client = gcs_storage.Client()
    bucket_obj = client.bucket(bucket)
    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_"))

    combined_path = None
    image_path = None

    if combined_gcs:
        combined_path = tmp_dir / "combined.json"
        bucket_obj.blob(combined_gcs).download_to_filename(str(combined_path))
        logger.info(f"Downloaded combined dataset: {combined_gcs}")

    if image_gcs:
        image_path = tmp_dir / "image.json"
        bucket_obj.blob(image_gcs).download_to_filename(str(image_path))
        logger.info(f"Downloaded image dataset: {image_gcs}")

    dataset = load_and_merge_datasets(combined_path, image_path)
    num_queries = len(dataset["queries"])
    logger.info(f"Merged dataset: {num_queries} queries")

    if num_queries == 0:
        logger.warning("No queries, exiting")
        return

    results_dir = tmp_dir / "results" / doc_source
    results_dir.mkdir(parents=True, exist_ok=True)

    for seq_num in sequences:
        logger.info(f"[{doc_source}] Running Sequence {seq_num}")
        output = run_sequence(
            seq_num=seq_num,
            dataset=dataset,
            cache_key=cache_key,
            bucket=bucket,
            run_id=None,
            k_values=k_values,
        )
        output["document_source"] = doc_source
        output["cache_key"] = cache_key

        out_file = save_results(output, results_dir, None)

        # Upload results to GCS
        result_gcs_path = f"eval-results/{doc_source}/{out_file.name}"
        bucket_obj.blob(result_gcs_path).upload_from_filename(str(out_file))
        logger.info(f"Uploaded: gs://{bucket}/{result_gcs_path}")

    logger.info(f"[{doc_source}] All {len(sequences)} sequences complete.")


def cr_eval():
    """Run the full CR baseline evaluation: reanchor + evaluate + upload results.

    Env vars:
      CACHE_KEYS: "AS-AMM-01-000:112e...|SC10000AMM:c313..."
      GCS_BUCKET: bucket name (default: raptor-assets)
    """
    from google.cloud import storage as gcs_storage

    from evaluation.eval_runner import (
        load_and_merge_datasets,
        resolve_ablation_dataset,
    )
    from evaluation.retrieval_evaluator import (
        evaluate_ablation,
        run_statistical_comparisons,
    )
    from ingestion.reanchor_per_ablation import reanchor_for_ablation

    bucket = os.environ.get("GCS_BUCKET", "raptor-assets")
    cache_keys_raw = os.environ["CACHE_KEYS"]

    manual_to_cache_key = {}
    for pair in cache_keys_raw.split("|"):
        manual, key = pair.strip().split(":")
        manual_to_cache_key[manual.strip()] = key.strip()

    client = gcs_storage.Client()
    bucket_obj = client.bucket(bucket)

    # --- Phase 1: Download raw dataset from GCS ---
    logger.info("=== Phase 1: Download datasets ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="cr_eval_"))

    raw_combined_gcs = os.environ.get(
        "RAW_COMBINED_GCS", "eval-datasets/combined_eval_dataset.json"
    )
    combined_path = tmp_dir / "combined.json"
    bucket_obj.blob(raw_combined_gcs).download_to_filename(str(combined_path))
    with open(combined_path) as f:
        queries = json.load(f)
    logger.info(f"Downloaded {len(queries)} queries")

    image_path = None
    raw_image_gcs = os.environ.get("RAW_IMAGE_GCS", "eval-datasets/image_eval_dataset.json")
    try:
        image_path = tmp_dir / "image.json"
        bucket_obj.blob(raw_image_gcs).download_to_filename(str(image_path))
        logger.info(f"Downloaded image dataset")
    except Exception:
        image_path = None
        logger.info("No image dataset found, skipping")

    # --- Phase 2: Reanchor for CR ---
    logger.info("=== Phase 2: Reanchor for contextual_retrieval ===")
    anchored, stats = reanchor_for_ablation(
        queries, manual_to_cache_key,
        "contextual_retrieval", "contextual_retrieval",
        bucket=bucket,
    )
    logger.info(f"Reanchoring complete. Confidence: {stats['confidence']}")

    # Write anchored dataset to temp
    anchored_path = tmp_dir / "combined_anchored_cr.json"
    with open(anchored_path, "w") as f:
        json.dump(anchored, f, indent=2)

    # Upload anchored dataset to GCS
    gcs_anchored_key = "eval-datasets/combined_eval_dataset_anchored_contextual_retrieval.json"
    bucket_obj.blob(gcs_anchored_key).upload_from_filename(str(anchored_path))
    logger.info(f"Uploaded anchored dataset to gs://{bucket}/{gcs_anchored_key}")

    # --- Phase 3: Evaluate CR + comparisons ---
    logger.info("=== Phase 3: Evaluate CR baseline ===")

    k_values = [1, 3, 5, 10]
    k_prime = 30
    max_k = max(k_values)

    comparisons = [
        ("no_context_aware", "baseline_naive_chunking"),
        ("contextual_retrieval", "contextual_retrieval"),
        ("no_header_prop", "no_header_propagation"),
        ("full", "full_context_aware"),
    ]

    significance_pairs = [
        ("no_context_aware", "baseline_naive_chunking",
         "contextual_retrieval", "contextual_retrieval", "CR vs naive baseline"),
        ("contextual_retrieval", "contextual_retrieval",
         "full", "full_context_aware", "Proposed system vs CR"),
        ("no_header_prop", "no_header_propagation",
         "contextual_retrieval", "contextual_retrieval", "CR vs no_header_prop"),
    ]

    # Build dataset dir with the anchored CR file
    dataset_dir = tmp_dir / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy(anchored_path, dataset_dir / "combined_eval_dataset_anchored_contextual_retrieval.json")

    # Also download per-ablation anchored datasets for comparison ablations
    for abl_key, _ in comparisons:
        if abl_key == "contextual_retrieval":
            continue
        gcs_key = f"eval-datasets/combined_eval_dataset_anchored_{abl_key}.json"
        local_path = dataset_dir / f"combined_eval_dataset_anchored_{abl_key}.json"
        try:
            bucket_obj.blob(gcs_key).download_to_filename(str(local_path))
            logger.info(f"Downloaded anchored dataset for {abl_key}")
        except Exception:
            logger.warning(f"No anchored dataset for {abl_key}, will use fallback")

    dataset = load_and_merge_datasets(combined_path, image_path)

    all_output_lines: list[str] = []
    all_results: dict[str, dict] = {}

    for source, cache_key in manual_to_cache_key.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {source}")
        logger.info(f"{'='*60}")

        doc_results: dict[str, dict] = {}
        doc_scores: dict[str, dict] = {}

        for ablation_name, ablation_label in comparisons:
            logger.info(f"  Evaluating: {ablation_label}")
            ablation_dataset = resolve_ablation_dataset(
                ablation_name, dataset_dir, dataset,
                image_path=image_path, manual_source=source,
            )
            metrics, scores = evaluate_ablation(
                cache_key=cache_key,
                ablation_name=ablation_name,
                dataset=ablation_dataset,
                k_values=k_values,
                run_id=None,
                bucket=bucket,
                k_prime=k_prime,
                retrieval_mode="embedding_only",
                reranker=None,
            )
            doc_results[ablation_label] = metrics
            doc_scores[ablation_label] = scores

        # Statistical tests
        metrics_to_test = [f"recall@{k}" for k in k_values] + ["mrr", f"ndcg@{max_k}"]
        stat_tests = {}
        for base_key, base_label, prop_key, prop_label, desc in significance_pairs:
            if base_label in doc_scores and prop_label in doc_scores:
                tests = run_statistical_comparisons(
                    doc_scores[base_label], doc_scores[prop_label],
                    base_label, prop_label, metrics_to_test,
                )
                stat_tests.update(tests)

        all_results[source] = {
            "results": doc_results,
            "stat_tests": stat_tests,
        }

        # Format output
        ndcg_key = f"ndcg@{max_k}"
        cr_overall = doc_results.get("contextual_retrieval", {}).get("overall", {})
        n_queries = cr_overall.get("num_queries", 0)

        lines = [f"\nOverall ({source}, n={n_queries}):"]
        lines.append(f"  {'':26s}  R@1    R@5    R@10   MRR    NDCG@{max_k}")
        lines.append(f"  {'-'*65}")
        for _, abl_label in comparisons:
            m = doc_results.get(abl_label, {}).get("overall", {})
            if not m:
                continue
            lines.append(
                f"  {abl_label:<26s}  {m.get('recall@1',0):.3f}  {m.get('recall@5',0):.3f}  "
                f"{m.get('recall@10',0):.3f}  {m.get('mrr',0):.3f}  {m.get(ndcg_key,0):.3f}"
            )

        lines.append(f"\n  Significance tests (NDCG@{max_k}, paired bootstrap 10K, seed 42):")
        for base_key, base_label, prop_key, prop_label, desc in significance_pairs:
            p_key = f"{prop_label}_vs_{base_label}_overall_{ndcg_key}"
            p_val = stat_tests.get(p_key, {}).get("p_value", float("nan"))
            base_ndcg = doc_results.get(base_label, {}).get("overall", {}).get(ndcg_key, 0)
            prop_ndcg = doc_results.get(prop_label, {}).get("overall", {}).get(ndcg_key, 0)
            delta = prop_ndcg - base_ndcg
            lines.append(f"    {desc:<30s}  Δ={delta:+.3f}  p={p_val:.3f}")

        for line in lines:
            logger.info(line)
        all_output_lines.extend(lines)

    # --- Phase 4: Save results to GCS ---
    logger.info("=== Phase 4: Upload results to GCS ===")

    results_payload = {
        "experiment": "contextual_retrieval_baseline",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "k_values": k_values,
        "k_prime": k_prime,
        "documents": all_results,
        "formatted_output": "\n".join(all_output_lines),
    }

    def _default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_json = json.dumps(results_payload, indent=2, default=_default)
    results_gcs_key = "eval-results/cr_baseline/cr_eval_results.json"
    bucket_obj.blob(results_gcs_key).upload_from_string(
        results_json, content_type="application/json"
    )
    logger.info(f"Results uploaded to gs://{bucket}/{results_gcs_key}")

    # Also upload a plain-text summary
    summary_gcs_key = "eval-results/cr_baseline/cr_eval_summary.txt"
    bucket_obj.blob(summary_gcs_key).upload_from_string(
        "\n".join(all_output_lines), content_type="text/plain"
    )
    logger.info(f"Summary uploaded to gs://{bucket}/{summary_gcs_key}")

    logger.info("CR baseline evaluation complete.")


def main():
    task_type = os.environ.get("TASK_TYPE", "evaluate")
    task_idx = _get_task_index()
    task_count = os.environ.get("CLOUD_RUN_TASK_COUNT", "1")

    logger.info(f"Worker started. TASK_TYPE={task_type}, index={task_idx}/{task_count}")

    if task_type == "rebuild_tree":
        rebuild_tree()
    elif task_type == "reanchor":
        reanchor()
    elif task_type == "evaluate":
        evaluate()
    elif task_type == "cr_eval":
        cr_eval()
    else:
        logger.error(f"Unknown TASK_TYPE: {task_type}")
        sys.exit(1)

    logger.info("Worker finished successfully.")


if __name__ == "__main__":
    main()
