"""Unified pipeline: verify trees → split datasets by document → run evaluation.

Runs the custom dataset evaluation (combined + image) for all documents,
automatically splitting queries by manual_source and evaluating against
the correct tree cache_key.

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
    GOOGLE_CLOUD_PROJECT=<your-gcp-project> \
    PYTHONPATH="/path/to/raptor/backend:." \
    python run_custom_eval_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
TREE_CACHE_DIR = PROJECT_ROOT / ".tree-cache"

GCS_BUCKET = "raptor-assets"

# Document mapping: manual_source → cache_key
DOC_MAP = {
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
}

# Datasets
COMBINED_DATASET = PROJECT_ROOT / "datasets" / "combined_eval_dataset_anchored.json"
IMAGE_DATASET = PROJECT_ROOT / "datasets" / "image_eval_dataset.json"

# Evaluation config
SEQUENCES = [1, 2, 3]
K_VALUES = [1, 3, 5, 10]
RETRIEVAL_MODE = "embedding_only"
K_PRIME = 30

# Ablations required per sequence
REQUIRED_ABLATIONS = {
    1: ["original_raptor_text_only", "full_context_aware"],
    2: ["semantic_chunking_baseline", "full_context_aware"],
    3: [
        "full_context_aware",
        "no_table_parent_child",
        "no_header_propagation",
        "no_caption_folding",
        "baseline_naive_chunking",
    ],
}


# ---------------------------------------------------------------------------
# Phase 1: Verify trees exist
# ---------------------------------------------------------------------------


def verify_trees() -> dict[str, list[str]]:
    """Check that all required ablation trees exist in local cache.

    Returns dict of {cache_key: [missing_labels]} — empty lists means all present.
    """
    all_labels = set()
    for labels in REQUIRED_ABLATIONS.values():
        all_labels.update(labels)

    missing: dict[str, list[str]] = {}
    for source, cache_key in DOC_MAP.items():
        doc_missing = []
        cache_dir = TREE_CACHE_DIR / cache_key
        if not cache_dir.exists():
            doc_missing = list(all_labels)
        else:
            for label in sorted(all_labels):
                # Check across all run_ids
                found = False
                for run_dir in cache_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    tree_file = run_dir / label / "tree.pkl"
                    if tree_file.exists():
                        found = True
                        break
                if not found:
                    doc_missing.append(label)

        missing[cache_key] = doc_missing
        if doc_missing:
            logger.warning(f"[{source}] Missing trees: {doc_missing}")
        else:
            logger.info(f"[{source}] All {len(all_labels)} required trees present ✓")

    return missing


# ---------------------------------------------------------------------------
# Phase 2: Split datasets by document
# ---------------------------------------------------------------------------


def split_dataset_by_source(
    dataset_path: Path,
) -> dict[str, list[dict]]:
    """Split a dataset JSON array by manual_source field."""
    if not dataset_path.exists():
        return {}

    entries = json.loads(dataset_path.read_text())
    by_source: dict[str, list[dict]] = {}
    for entry in entries:
        source = entry.get("manual_source", "unknown")
        by_source.setdefault(source, []).append(entry)

    return by_source


# ---------------------------------------------------------------------------
# Phase 3: Run evaluation
# ---------------------------------------------------------------------------


def run_evaluation_for_document(
    source: str,
    cache_key: str,
    combined_queries: list[dict],
    image_queries: list[dict],
    results_base_dir: Path,
    retrieval_mode: str = "embedding_only",
    k_prime: int = 30,
):
    """Run eval_runner.run_sequence for all sequences against one document."""
    from evaluation.eval_runner import (
        load_and_merge_datasets,
        run_sequence,
        save_results,
    )

    logger.info(f"\n{'='*72}")
    logger.info(f"EVALUATING: {source} (cache_key: {cache_key[:12]}...)")
    logger.info(f"  Combined queries: {len(combined_queries)}")
    logger.info(f"  Image queries: {len(image_queries)}")
    logger.info(f"{'='*72}")

    # Write temporary split dataset files
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"raptor_eval_{source}_"))
    combined_path = None
    image_path = None

    if combined_queries:
        combined_path = tmp_dir / "combined.json"
        combined_path.write_text(json.dumps(combined_queries, indent=2))

    if image_queries:
        image_path = tmp_dir / "image.json"
        image_path.write_text(json.dumps(image_queries, indent=2))

    # Load and merge into evaluator format
    dataset = load_and_merge_datasets(combined_path, image_path)
    num_queries = len(dataset["queries"])
    logger.info(f"Merged dataset: {num_queries} queries for {source}")

    if num_queries == 0:
        logger.warning(f"No queries for {source}, skipping")
        return

    # Results subdirectory per document
    results_dir = results_base_dir / source
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run all sequences
    for seq_num in SEQUENCES:
        logger.info(f"\n{'─'*60}")
        logger.info(f"[{source}] Running Sequence {seq_num}")
        logger.info(f"{'─'*60}")

        output = run_sequence(
            seq_num=seq_num,
            dataset=dataset,
            cache_key=cache_key,
            bucket=GCS_BUCKET,
            run_id=None,  # use latest tree
            k_values=K_VALUES,
            dataset_dir=PROJECT_ROOT / "datasets",
            retrieval_mode=retrieval_mode,
            k_prime=k_prime,
            manual_source=source,
            doc_label=source,
        )

        # Tag with document source and save
        if isinstance(output, list):
            for o in output:
                o["document_source"] = source
                o["cache_key"] = cache_key
                save_results(o, results_dir, None)
        else:
            output["document_source"] = source
            output["cache_key"] = cache_key
            save_results(output, results_dir, None)

    # Cleanup temp files
    if combined_path and combined_path.exists():
        combined_path.unlink()
    if image_path and image_path.exists():
        image_path.unlink()
    tmp_dir.rmdir()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 72)
    logger.info("RAPTOR Custom Dataset Evaluation Pipeline")
    logger.info(f"Started: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 72)

    # Phase 1: Verify trees
    logger.info("\n[Phase 1] Verifying tree availability...")
    missing = verify_trees()

    total_missing = sum(len(v) for v in missing.values())
    if total_missing > 0:
        logger.error(
            f"Cannot proceed: {total_missing} tree(s) missing. "
            f"Run ingestion/orchestrator.py to build them first."
        )
        for cache_key, labels in missing.items():
            if labels:
                source = next(s for s, k in DOC_MAP.items() if k == cache_key)
                logger.error(f"  [{source}] missing: {labels}")
        sys.exit(1)

    logger.info("All trees verified. Proceeding to evaluation.\n")

    # Phase 2: Split datasets
    logger.info("[Phase 2] Splitting datasets by document source...")
    combined_by_source = split_dataset_by_source(COMBINED_DATASET)
    image_by_source = split_dataset_by_source(IMAGE_DATASET)

    for source in DOC_MAP:
        n_combined = len(combined_by_source.get(source, []))
        n_image = len(image_by_source.get(source, []))
        logger.info(f"  {source}: {n_combined} combined + {n_image} image = {n_combined + n_image} total")

    # Phase 3: Run evaluations (immediately, no manual control)
    logger.info("\n[Phase 3] Running evaluations...")
    results_base = PROJECT_ROOT / "evaluation" / "results"

    for source, cache_key in DOC_MAP.items():
        combined_queries = combined_by_source.get(source, [])
        image_queries = image_by_source.get(source, [])

        if not combined_queries and not image_queries:
            logger.info(f"[{source}] No queries, skipping")
            continue

        run_evaluation_for_document(
            source=source,
            cache_key=cache_key,
            combined_queries=combined_queries,
            image_queries=image_queries,
            results_base_dir=results_base,
            retrieval_mode=RETRIEVAL_MODE,
            k_prime=K_PRIME,
        )

    logger.info("\n" + "=" * 72)
    logger.info("Pipeline complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
