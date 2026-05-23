"""Tree resolution for the evaluation harness.

Single point of truth for loading a tree artifact by cache_key + ablation_label,
with optional pinning to a specific run_id or time-travel query.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from ingestion.gcs_cache import (
    LOCAL_MIRROR_DIR,
    download_tree_artifact,
    list_run_ids_for_ablation,
    _read_local_mirror,
    _write_local_mirror,
)

logger = logging.getLogger(__name__)


def resolve_tree(
    cache_key: str,
    ablation_label: str,
    run_id: str | None = None,
    before: datetime | None = None,
    bucket: str | None = None,
) -> tuple[bytes, dict]:
    """Resolve which tree to use for a given ablation case.

    If run_id is provided, load that exact tree.
    If run_id is None, load the LATEST tree for this cache_key + ablation_label
    whose created_at <= before (default: now).

    Returns (tree_pickle_bytes, manifest_dict).

    This allows:
    - Pinning to a specific run_id for reproducibility
    - Defaulting to latest for iterative development
    - Time-travel queries if the pipeline changed mid-experiment

    Raises:
        FileNotFoundError: If no matching tree is found.
        ValueError: If bucket is not provided and GCS_BUCKET env var is not set.
    """
    bucket = bucket or os.environ.get("GCS_BUCKET") or os.environ.get("GCS_BUCKET_NAME")
    if not bucket:
        raise ValueError(
            "No GCS bucket specified. Pass bucket= or set GCS_BUCKET env var."
        )

    if before is None:
        before = datetime.now(timezone.utc)

    if run_id is not None:
        # Exact match: try local mirror first, then GCS
        result = download_tree_artifact(bucket, cache_key, run_id, ablation_label)
        if result is None:
            raise FileNotFoundError(
                f"Tree not found: cache_key={cache_key}, "
                f"run_id={run_id}, ablation={ablation_label}"
            )
        return result

    # No run_id: find latest tree before the cutoff
    # First, check local mirror for any cached manifests
    local_candidates = _find_local_candidates(cache_key, ablation_label, before)
    if local_candidates:
        # Use the most recent local candidate
        best_run_id, _ = local_candidates[0]
        result = _read_local_mirror(cache_key, best_run_id, ablation_label)
        if result is not None:
            logger.info(f"Resolved tree from local mirror: run_id={best_run_id}")
            return result

    # Fall back to GCS listing
    runs = list_run_ids_for_ablation(bucket, cache_key, ablation_label)
    if not runs:
        raise FileNotFoundError(
            f"No trees found for cache_key={cache_key}, ablation={ablation_label}"
        )

    # Filter by before timestamp
    for rid, manifest in runs:
        created_at_str = manifest.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if created_at <= before:
            # Download and cache locally
            result = download_tree_artifact(bucket, cache_key, rid, ablation_label)
            if result is not None:
                logger.info(f"Resolved tree from GCS: run_id={rid}")
                return result

    raise FileNotFoundError(
        f"No trees found before {before.isoformat()} for "
        f"cache_key={cache_key}, ablation={ablation_label}"
    )


def _find_local_candidates(
    cache_key: str, ablation_label: str, before: datetime
) -> list[tuple[str, dict]]:
    """Scan local mirror for matching trees, sorted by created_at descending."""
    cache_dir = LOCAL_MIRROR_DIR / cache_key
    if not cache_dir.exists():
        return []

    candidates: list[tuple[str, dict]] = []
    for run_dir in cache_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_file = run_dir / ablation_label / "tree_manifest.json"
        if not manifest_file.exists():
            continue
        try:
            manifest = json.loads(manifest_file.read_text())
            created_at_str = manifest.get("created_at", "")
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            if created_at <= before:
                candidates.append((run_dir.name, manifest))
        except (ValueError, json.JSONDecodeError):
            continue

    candidates.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
    return candidates
