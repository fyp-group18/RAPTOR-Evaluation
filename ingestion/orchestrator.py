"""Docling ingestion + ablation worker orchestrator.

Two-phase pipeline:
  Phase 1: Parse document with Docling, cache to GCS
  Phase 2: Build RAPTOR trees under N ablation configurations in parallel

Usage (build ablation trees):
    PYTHONPATH=/path/to/backend python -m ingestion.orchestrator \
        --document /path/to/manual.pdf \
        --gcs-bucket my-bucket \
        --ablations full,no_table_pc,flat_retrieval

Usage (register an existing production tree):
    python -m ingestion.orchestrator \
        --register-existing-tree /path/to/existing/tree.pkl \
        --document /path/to/aircraft_manual.pdf \
        --gcs-bucket my-raptor-bucket \
        --ablation-label full_context_aware
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Ensure backend is importable (same pattern as run_eval.py)
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_BACKEND_DIR = str(_PROJECT_ROOT.parent)  # parent of RAPTOR-Evaluation = backend/
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ingestion.ablation_configs import ABLATION_CASES, ALL_ABLATION_NAMES, validate_ablation_selection
from ingestion.docling_runner import get_docling_version, run_docling
from ingestion.gcs_cache import (
    check_docling_cache,
    compute_cache_key,
    upload_docling_cache,
    upload_tree_artifact,
)
from ingestion.tree_builder import build_tree_for_ablation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(_PROJECT_ROOT),
        ).decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _run_phase_1(
    document_path: Path,
    bucket: str,
    force_reparse: bool,
) -> tuple[dict, str, bool]:
    """Phase 1: Parse with Docling, cache to GCS.

    Returns:
        (docling_dict, cache_key, was_cache_hit)
    """
    file_bytes = document_path.read_bytes()
    cache_key = compute_cache_key(file_bytes)
    logger.info(f"Document: {document_path.name} (cache_key: {cache_key[:12]}...)")

    if not force_reparse:
        cached_output, cached_metadata = check_docling_cache(bucket, cache_key)
        if cached_output is not None:
            # Check for version mismatch warning
            if cached_metadata:
                try:
                    cached_version = cached_metadata.get("docling_version", "unknown")
                    current_version = get_docling_version()
                    if cached_version != current_version:
                        logger.warning(
                            f"Docling version mismatch: cached={cached_version}, "
                            f"current={current_version}. Using cached output. "
                            f"Pass --force-reparse to regenerate."
                        )
                except Exception:
                    pass
            logger.info(f"Cache hit for {cache_key[:12]}...")
            return cached_output, cache_key, True

    # Cache miss or forced: run Docling
    logger.info(f"Cache miss. Running Docling on {document_path.name}...")
    docling_dict = run_docling(document_path)

    # Upload to cache
    metadata = {
        "original_filename": document_path.name,
        "sha256": cache_key,
        "docling_version": get_docling_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": len(file_bytes),
    }
    upload_docling_cache(bucket, cache_key, docling_dict, metadata)
    logger.info(f"Cache miss. Docling complete. Cached at {cache_key[:12]}...")

    return docling_dict, cache_key, False


def _run_phase_2(
    docling_output: dict,
    cache_key: str,
    bucket: str,
    ablation_selections: list[tuple[str, dict]],
    original_filename: str,
    run_id: str,
) -> list[dict]:
    """Phase 2: Build trees for each ablation config in parallel.

    Returns list of result dicts with status per ablation.
    """
    git_hash = _get_git_hash()
    results: list[dict] = []

    def _worker(name: str, config: dict) -> dict:
        label = config["label"]
        t0 = time.perf_counter()
        try:
            tree_bytes, tree_stats = build_tree_for_ablation(
                docling_output, config, original_filename
            )
            elapsed = time.perf_counter() - t0

            # Build manifest
            try:
                docling_version = get_docling_version()
            except Exception:
                docling_version = "unknown"

            manifest = {
                "run_id": run_id,
                "cache_key": cache_key,
                "ablation_label": label,
                "ablation_config": config,
                "source_document": original_filename,
                "docling_cache_path": f"gs://{bucket}/docling-cache/{cache_key}/docling_output.json",
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "tree_stats": tree_stats,
                "embedding_model": "gemini-embedding-2" if config.get("multimodal_embedding") else "text-only",
                "docling_version": docling_version,
                "pipeline_git_hash": git_hash,
            }

            # Upload to GCS
            upload_tree_artifact(bucket, cache_key, run_id, label, tree_bytes, manifest)

            return {
                "name": name,
                "label": label,
                "success": True,
                "stats": tree_stats,
                "elapsed_s": elapsed,
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(f"[{label}] FAILED: {e}")
            logger.debug(traceback.format_exc())
            return {
                "name": name,
                "label": label,
                "success": False,
                "error": str(e),
                "elapsed_s": elapsed,
            }

    # Dispatch workers in parallel
    max_workers = min(4, len(ablation_selections))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, name, config): name
            for name, config in ablation_selections
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    # Sort results by ablation name for consistent output
    results.sort(key=lambda r: ALL_ABLATION_NAMES.index(r["name"]) if r["name"] in ALL_ABLATION_NAMES else 999)
    return results


def _print_summary(
    run_id: str,
    original_filename: str,
    cache_key: str,
    was_cache_hit: bool,
    results: list[dict],
) -> None:
    """Print a summary table of the run."""
    print("\n" + "=" * 72)
    print(f"Run ID: {run_id}")
    print(f"Document: {original_filename} (cache_key: {cache_key[:12]}...)")
    print(f"Docling: {'CACHED (hit)' if was_cache_hit else 'PARSED (miss)'}")
    print()
    print("Ablation Results:")

    for r in results:
        label = r["label"]
        if r["success"]:
            stats = r["stats"]
            elapsed = r["elapsed_s"]
            print(
                f"  {label:<30} \u2713  "
                f"{stats['num_leaf_nodes']:>3} leaves, "
                f"{stats['num_summary_nodes']:>3} summaries, "
                f"{stats['num_levels']} {'level' if stats['num_levels'] == 1 else 'levels'}"
                f"  [{elapsed:>5.1f}s]"
            )
        else:
            error = r.get("error", "Unknown error")
            # Truncate long errors
            if len(error) > 50:
                error = error[:47] + "..."
            print(f"  {label:<30} \u2717  FAILED: {error}")

    print("=" * 72)

    success_count = sum(1 for r in results if r["success"])
    fail_count = sum(1 for r in results if not r["success"])
    if fail_count:
        print(f"\n{success_count} succeeded, {fail_count} failed.")
    else:
        print(f"\nAll {success_count} ablations completed successfully.")


def _print_dry_run(
    document_path: Path,
    cache_key: str,
    bucket: str,
    ablation_selections: list[tuple[str, dict]],
    force_reparse: bool,
) -> None:
    """Print what would happen without executing."""
    print("\n[DRY RUN] No operations will be performed.\n")
    print(f"Document: {document_path.name}")
    print(f"Cache key: {cache_key}")
    print(f"GCS bucket: {bucket}")
    print(f"Force reparse: {force_reparse}")
    print(f"\nPhase 1:")
    print(f"  Check cache: gs://{bucket}/docling-cache/{cache_key}/docling_output.json")
    if force_reparse:
        print(f"  Will SKIP cache check and run Docling")
    else:
        print(f"  If miss: run Docling, upload to cache")
        print(f"  If hit: use cached output")

    print(f"\nPhase 2 ({len(ablation_selections)} ablations):")
    for name, config in ablation_selections:
        label = config["label"]
        features = []
        if config.get("table_parent_child"):
            features.append("table-P/C")
        if config.get("header_propagation"):
            features.append("headers")
        if config.get("caption_folding"):
            features.append("captions")
        if config.get("multimodal_embedding"):
            features.append("multimodal")
        if config.get("raptor_tree"):
            features.append("RAPTOR")
        else:
            features.append("flat")
        if config.get("chunking_strategy") == "semantic":
            features.append("semantic-chunk")

        print(f"  {label:<30} [{', '.join(features)}]")
        print(f"    → gs://{bucket}/trees/{cache_key}/<run_id>/{label}/tree.pkl")


def _register_existing_tree(
    tree_path: Path,
    document_path: Path,
    bucket: str,
    ablation_label: str,
) -> None:
    """Register a pre-built tree pickle without rebuilding.

    Computes cache_key from the source document, extracts stats from the
    tree nodes, writes manifest, and uploads to GCS + local mirror.
    """
    # 1. Compute cache_key from source document
    file_bytes = document_path.read_bytes()
    cache_key = compute_cache_key(file_bytes)
    logger.info(f"Document: {document_path.name} (cache_key: {cache_key[:12]}...)")

    # 2. Generate run_id
    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    logger.info(f"Run ID: {run_id}")

    # 3. Load tree and extract stats
    tree_bytes = tree_path.read_bytes()
    nodes: list[dict] = pickle.loads(tree_bytes)
    logger.info(f"Loaded {len(nodes)} nodes from {tree_path.name}")

    num_leaf = sum(1 for n in nodes if n.get("level", 0) == 0)
    num_summary = sum(1 for n in nodes if n.get("level", 0) > 0)
    levels = set(n.get("level", 0) for n in nodes)
    num_levels = max(levels) + 1 if levels else 0
    num_image_nodes = sum(1 for n in nodes if n.get("images"))
    num_table_chunks = sum(
        1 for n in nodes if n.get("parent_id") or n.get("is_parent")
    )

    tree_stats = {
        "num_leaf_nodes": num_leaf,
        "num_summary_nodes": num_summary,
        "num_levels": num_levels,
        "num_image_nodes": num_image_nodes,
        "num_table_chunks": num_table_chunks,
    }

    # 3b. Detect page-related fields on nodes
    page_field_candidates = [
        "pages", "page_start", "page_end", "page", "page_number",
        "source_page", "page_no",
    ]
    sample = nodes[:50] if len(nodes) >= 50 else nodes
    found_fields = []
    for field in page_field_candidates:
        count = sum(1 for n in sample if n.get(field) is not None)
        if count > 0:
            found_fields.append((field, count, len(sample)))

    print(f"\nPage-related fields found on nodes (sampled {len(sample)}):")
    for field, count, total in found_fields:
        print(f"  {field}: present on {count}/{total} sampled nodes")
    if not found_fields:
        print("  (none found)")

    # Also check for nested metadata.page
    meta_page_count = sum(
        1 for n in sample
        if isinstance(n.get("metadata"), dict) and n["metadata"].get("page") is not None
    )
    if meta_page_count > 0:
        print(f"  metadata.page: present on {meta_page_count}/{len(sample)} sampled nodes")

    # 4. Resolve ablation config
    config = None
    for cfg in ABLATION_CASES.values():
        if cfg["label"] == ablation_label:
            config = cfg
            break
    if config is None:
        logger.error(
            f"Unknown ablation label: {ablation_label}. "
            f"Valid labels: {[c['label'] for c in ABLATION_CASES.values()]}"
        )
        sys.exit(1)

    # 5. Build manifest
    git_hash = _get_git_hash()
    manifest = {
        "run_id": run_id,
        "cache_key": cache_key,
        "ablation_label": ablation_label,
        "ablation_config": config,
        "source_document": document_path.name,
        "docling_cache_path": f"gs://{bucket}/docling-cache/{cache_key}/docling_output.json",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tree_stats": tree_stats,
        "registered_from": str(tree_path.resolve()),
        "pipeline_git_hash": git_hash,
    }

    # 6. Upload to GCS + local mirror
    gcs_path = upload_tree_artifact(
        bucket, cache_key, run_id, ablation_label, tree_bytes, manifest
    )

    # Summary
    print(f"\n{'=' * 72}")
    print(f"Registered existing tree as: {ablation_label}")
    print(f"Run ID:     {run_id}")
    print(f"Cache key:  {cache_key}")
    print(f"GCS path:   {gcs_path}")
    print(f"Stats:      {num_leaf} leaves, {num_summary} summaries, "
          f"{num_levels} levels, {num_image_nodes} image nodes, "
          f"{num_table_chunks} table chunks")
    print(f"{'=' * 72}")


def main():
    parser = argparse.ArgumentParser(
        description="Docling ingestion + ablation RAPTOR tree builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--document",
        type=Path,
        required=True,
        help="Path to the PDF document (used for cache_key in all modes)",
    )
    parser.add_argument(
        "--gcs-bucket",
        required=True,
        help="GCS bucket name for cache and tree storage",
    )
    parser.add_argument(
        "--register-existing-tree",
        type=Path,
        default=None,
        metavar="TREE_PKL",
        help="Register a pre-built tree pickle instead of rebuilding",
    )
    parser.add_argument(
        "--ablation-label",
        type=str,
        default=None,
        help="Ablation label for --register-existing-tree (e.g. full_context_aware)",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default=",".join(ALL_ABLATION_NAMES),
        help=f"Comma-separated ablation cases to run (default: all). "
        f"Options: {', '.join(ALL_ABLATION_NAMES)}",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only run Phase 1 (Docling parse + cache), skip tree building",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Skip cache check and always run Docling, overwriting existing cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )

    args = parser.parse_args()

    # Validate document path
    if not args.document.exists():
        parser.error(f"Document not found: {args.document}")

    # ---------- Register existing tree mode ----------
    if args.register_existing_tree is not None:
        if not args.register_existing_tree.exists():
            parser.error(f"Tree file not found: {args.register_existing_tree}")
        if not args.ablation_label:
            parser.error("--ablation-label is required with --register-existing-tree")
        _register_existing_tree(
            tree_path=args.register_existing_tree,
            document_path=args.document,
            bucket=args.gcs_bucket,
            ablation_label=args.ablation_label,
        )
        return

    # Validate ablation selections
    ablation_names = [n.strip() for n in args.ablations.split(",") if n.strip()]
    try:
        ablation_selections = validate_ablation_selection(ablation_names)
    except ValueError as e:
        parser.error(str(e))

    # Dry run mode
    if args.dry_run:
        file_bytes = args.document.read_bytes()
        cache_key = compute_cache_key(file_bytes)
        _print_dry_run(args.document, cache_key, args.gcs_bucket, ablation_selections, args.force_reparse)
        return

    # Phase 1: Docling parsing with GCS cache
    docling_output, cache_key, was_cache_hit = _run_phase_1(
        args.document, args.gcs_bucket, args.force_reparse
    )

    if args.parse_only:
        print(f"\n--parse-only: Phase 1 complete. Cache key: {cache_key}")
        print(f"Cached at: gs://{args.gcs_bucket}/docling-cache/{cache_key}/")
        return

    # Phase 2: Ablation tree building
    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    logger.info(f"Run ID: {run_id}")

    results = _run_phase_2(
        docling_output,
        cache_key,
        args.gcs_bucket,
        ablation_selections,
        args.document.name,
        run_id,
    )

    # Print summary
    _print_summary(run_id, args.document.name, cache_key, was_cache_hit, results)

    # Exit code: 1 if ALL failed, 0 otherwise
    all_failed = all(not r["success"] for r in results)
    if all_failed:
        logger.error("All ablation workers failed.")
        sys.exit(1)

    failed_count = sum(1 for r in results if not r["success"])
    if failed_count:
        logger.warning(f"{failed_count} ablation(s) failed. See above for details.")


if __name__ == "__main__":
    main()
