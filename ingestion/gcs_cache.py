"""GCS cache operations for Docling output and tree artifacts.

Handles:
- Docling parse cache (content-hash keyed)
- Tree artifact upload/download
- Local mirror at .tree-cache/ for repeated evaluation runs
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)

# Local mirror root (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MIRROR_DIR = _PROJECT_ROOT / ".tree-cache"


def compute_cache_key(file_bytes: bytes) -> str:
    """Deterministic cache key from file content."""
    return hashlib.sha256(file_bytes).hexdigest()


def _get_client() -> storage.Client:
    """Get GCS client using Application Default Credentials."""
    return storage.Client()


# ---------------------------------------------------------------------------
# Docling cache operations
# ---------------------------------------------------------------------------


def check_docling_cache(
    bucket_name: str, cache_key: str
) -> tuple[dict | None, dict | None]:
    """Check if Docling output is cached (local mirror first, then GCS).

    Returns (docling_dict, metadata_dict) if cached, (None, None) if not.
    """
    # Check local mirror first
    local_dir = LOCAL_MIRROR_DIR / "docling-cache" / cache_key
    local_output = local_dir / "docling_output.json"
    if local_output.exists():
        logger.info(f"Docling cache hit (local): {cache_key[:12]}...")
        docling_dict = json.loads(local_output.read_text())
        metadata_dict = None
        local_meta = local_dir / "metadata.json"
        if local_meta.exists():
            metadata_dict = json.loads(local_meta.read_text())
        return docling_dict, metadata_dict

    # Fall back to GCS
    try:
        client = _get_client()
        bucket = client.bucket(bucket_name)

        output_blob = bucket.blob(f"docling-cache/{cache_key}/docling_output.json")
        if not output_blob.exists():
            return None, None

        metadata_blob = bucket.blob(f"docling-cache/{cache_key}/metadata.json")

        docling_dict = json.loads(output_blob.download_as_bytes())

        metadata_dict = None
        if metadata_blob.exists():
            metadata_dict = json.loads(metadata_blob.download_as_bytes())

        # Populate local mirror
        _write_local_docling_cache(cache_key, docling_dict, metadata_dict)

        return docling_dict, metadata_dict
    except Exception as e:
        logger.warning(f"GCS docling cache check failed ({e}), no local cache found")
        return None, None


def upload_docling_cache(
    bucket_name: str,
    cache_key: str,
    docling_dict: dict,
    metadata: dict,
) -> None:
    """Upload Docling output and metadata sidecar to GCS + local mirror."""
    # Always write local mirror
    _write_local_docling_cache(cache_key, docling_dict, metadata)

    try:
        client = _get_client()
        bucket = client.bucket(bucket_name)

        output_blob = bucket.blob(f"docling-cache/{cache_key}/docling_output.json")
        output_blob.upload_from_string(
            json.dumps(docling_dict, ensure_ascii=False),
            content_type="application/json",
        )

        metadata_blob = bucket.blob(f"docling-cache/{cache_key}/metadata.json")
        metadata_blob.upload_from_string(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

        logger.info(f"Docling output cached at gs://{bucket_name}/docling-cache/{cache_key}/")
    except Exception as e:
        logger.warning(f"GCS docling upload failed ({e}), using local mirror only")


def _write_local_docling_cache(
    cache_key: str, docling_dict: dict, metadata: dict | None
) -> None:
    local_dir = LOCAL_MIRROR_DIR / "docling-cache" / cache_key
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "docling_output.json").write_text(
        json.dumps(docling_dict, ensure_ascii=False)
    )
    if metadata:
        (local_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )


# ---------------------------------------------------------------------------
# Tree artifact operations
# ---------------------------------------------------------------------------


def upload_tree_artifact(
    bucket_name: str,
    cache_key: str,
    run_id: str,
    ablation_label: str,
    tree_bytes: bytes,
    manifest: dict,
) -> str:
    """Upload tree pickle and manifest to GCS. Returns the GCS prefix path.

    Falls back to local-only storage if GCS upload fails.
    """
    prefix = f"trees/{cache_key}/{run_id}/{ablation_label}"
    gcs_path = f"gs://{bucket_name}/{prefix}"

    try:
        client = _get_client()
        bucket = client.bucket(bucket_name)

        tree_blob = bucket.blob(f"{prefix}/tree.pkl")
        tree_blob.upload_from_string(tree_bytes, content_type="application/octet-stream")

        manifest_blob = bucket.blob(f"{prefix}/tree_manifest.json")
        manifest_blob.upload_from_string(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

        logger.info(f"Tree artifact uploaded to {gcs_path}")
    except Exception as e:
        logger.warning(f"GCS upload failed ({e}), using local mirror only")
        gcs_path = f"local://{prefix}"

    # Always populate local mirror
    _write_local_mirror(cache_key, run_id, ablation_label, tree_bytes, manifest)

    return gcs_path


def download_tree_artifact(
    bucket_name: str,
    cache_key: str,
    run_id: str,
    ablation_label: str,
) -> tuple[bytes, dict] | None:
    """Download tree pickle and manifest from GCS.

    Checks local mirror first. Populates mirror on GCS download.
    Returns (tree_bytes, manifest_dict) or None if not found.
    """
    # Check local mirror first
    local = _read_local_mirror(cache_key, run_id, ablation_label)
    if local is not None:
        logger.debug(f"Local mirror hit: {cache_key}/{run_id}/{ablation_label}")
        return local

    # Fall back to GCS
    client = _get_client()
    bucket = client.bucket(bucket_name)
    prefix = f"trees/{cache_key}/{run_id}/{ablation_label}"

    tree_blob = bucket.blob(f"{prefix}/tree.pkl")
    manifest_blob = bucket.blob(f"{prefix}/tree_manifest.json")

    if not tree_blob.exists():
        return None

    tree_bytes = tree_blob.download_as_bytes()
    manifest = json.loads(manifest_blob.download_as_bytes())

    # Populate local mirror
    _write_local_mirror(cache_key, run_id, ablation_label, tree_bytes, manifest)

    return tree_bytes, manifest


def list_run_ids_for_ablation(
    bucket_name: str,
    cache_key: str,
    ablation_label: str,
) -> list[tuple[str, dict]]:
    """List all (run_id, manifest) pairs for a given cache_key + ablation_label.

    Returns list sorted by created_at descending (most recent first).
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)
    prefix = f"trees/{cache_key}/"

    results: list[tuple[str, dict]] = []
    # List all blobs under trees/{cache_key}/ to find run_ids with this ablation_label
    blobs = client.list_blobs(bucket, prefix=prefix, delimiter="/")
    # We need to iterate subdirectories (run_ids)
    # list_blobs with delimiter gives us prefixes for subdirectories
    # but we need a different approach: list all manifests matching the pattern
    manifest_prefix = f"trees/{cache_key}/"
    for blob in client.list_blobs(bucket, prefix=manifest_prefix):
        # Match pattern: trees/{cache_key}/{run_id}/{ablation_label}/tree_manifest.json
        parts = blob.name.split("/")
        if (
            len(parts) == 5
            and parts[3] == ablation_label
            and parts[4] == "tree_manifest.json"
        ):
            run_id = parts[2]
            manifest = json.loads(blob.download_as_bytes())
            results.append((run_id, manifest))

    # Sort by created_at descending
    results.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Local mirror operations
# ---------------------------------------------------------------------------


def _local_mirror_path(cache_key: str, run_id: str, ablation_label: str) -> Path:
    return LOCAL_MIRROR_DIR / cache_key / run_id / ablation_label


def _write_local_mirror(
    cache_key: str,
    run_id: str,
    ablation_label: str,
    tree_bytes: bytes,
    manifest: dict,
) -> None:
    path = _local_mirror_path(cache_key, run_id, ablation_label)
    path.mkdir(parents=True, exist_ok=True)
    (path / "tree.pkl").write_bytes(tree_bytes)
    (path / "tree_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )


def _read_local_mirror(
    cache_key: str, run_id: str, ablation_label: str
) -> tuple[bytes, dict] | None:
    path = _local_mirror_path(cache_key, run_id, ablation_label)
    tree_file = path / "tree.pkl"
    manifest_file = path / "tree_manifest.json"
    if not tree_file.exists() or not manifest_file.exists():
        return None
    tree_bytes = tree_file.read_bytes()
    manifest = json.loads(manifest_file.read_text())
    return tree_bytes, manifest
