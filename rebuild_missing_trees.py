"""Rebuild empty/failed trees from cached Docling output.

Scans .tree-cache for trees with 0 nodes and rebuilds them from
the locally cached docling_output.json.

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
    GOOGLE_CLOUD_PROJECT=raptor-496700 \
    PYTHONPATH="/Users/joanne/Desktop/PycharmProjects/hitl-dss-react/backend:." \
    python3 rebuild_missing_trees.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TREE_CACHE_DIR = PROJECT_ROOT / ".tree-cache"
GCS_BUCKET = "raptor-assets"

# Trees that need rebuilding: (cache_key, ablation_name, ablation_label)
TREES_TO_REBUILD = [
    (
        "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
        "text_only_raptor",
        "original_raptor_text_only",
        "AS_AMM_01_000_I1_R1_Feb_2_2018_compressed.pdf",
    ),
    (
        "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
        "semantic_chunking",
        "semantic_chunking_baseline",
        "AS_AMM_01_000_I1_R1_Feb_2_2018_compressed.pdf",
    ),
    (
        "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
        "semantic_chunking",
        "semantic_chunking_baseline",
        "SC10000AMM_Rev_J.pdf",
    ),
]


def main():
    sys.path.insert(0, str(PROJECT_ROOT))

    from ingestion.ablation_configs import ABLATION_CASES
    from ingestion.gcs_cache import upload_tree_artifact
    from ingestion.tree_builder import build_tree_for_ablation

    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    logger.info(f"Rebuild run_id: {run_id}")

    for cache_key, ablation_name, ablation_label, filename in TREES_TO_REBUILD:
        logger.info(f"\n{'='*60}")
        logger.info(f"Rebuilding: {ablation_label} for {filename}")
        logger.info(f"{'='*60}")

        # Load Docling output from local cache
        docling_path = TREE_CACHE_DIR / "docling-cache" / cache_key / "docling_output.json"
        if not docling_path.exists():
            logger.error(f"Docling cache not found: {docling_path}")
            continue

        logger.info(f"Loading Docling output ({docling_path.stat().st_size / 1e6:.1f} MB)...")
        docling_output = json.loads(docling_path.read_text())

        config = ABLATION_CASES[ablation_name]
        t0 = time.perf_counter()

        try:
            tree_bytes, tree_stats = build_tree_for_ablation(
                docling_output, config, filename
            )
            elapsed = time.perf_counter() - t0

            # Verify non-empty
            nodes = pickle.loads(tree_bytes)
            if len(nodes) == 0:
                logger.error(f"[{ablation_label}] Build produced 0 nodes — still empty!")
                continue

            has_emb = sum(1 for n in nodes if n.get("embedding") is not None)
            logger.info(
                f"[{ablation_label}] Built: {len(nodes)} nodes, "
                f"{has_emb} with embedding [{elapsed:.1f}s]"
            )

            # Build manifest
            manifest = {
                "run_id": run_id,
                "cache_key": cache_key,
                "ablation_label": ablation_label,
                "ablation_config": config,
                "source_document": filename,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "tree_stats": tree_stats,
                "rebuild": True,
            }

            # Upload to GCS + local mirror
            upload_tree_artifact(
                GCS_BUCKET, cache_key, run_id, ablation_label, tree_bytes, manifest
            )
            logger.info(f"[{ablation_label}] Uploaded to GCS and local mirror ✓")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(f"[{ablation_label}] FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\nRebuild complete.")


if __name__ == "__main__":
    main()
