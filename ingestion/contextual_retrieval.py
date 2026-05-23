"""Contextual Retrieval (CR) chunk context generation.

Implements the Anthropic Contextual Retrieval method: for each naive chunk,
an LLM generates a short context snippet that situates the chunk within the
full document. The context is prepended to the chunk text before embedding.

Reference: https://www.anthropic.com/news/contextual-retrieval (2024)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT_DIR = _PROJECT_ROOT / ".cr-checkpoints"

# Approximate chars-per-token ratio for Gemini models
_CHARS_PER_TOKEN = 4
_MAX_DOCUMENT_TOKENS = 900_000
_MAX_DOCUMENT_CHARS = _MAX_DOCUMENT_TOKENS * _CHARS_PER_TOKEN
_WINDOW_PAGES = 15  # ±pages around chunk source when document exceeds limit

_CR_PROMPT_TEMPLATE = """\
<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def _build_full_document_text(pages_dict: dict) -> tuple[str, dict[int, tuple[int, int]]]:
    """Reconstruct full document text from pages_dict.

    Mirrors the concatenation logic in tree_builder._chunk_naive() so the
    document text used for CR context matches the text the naive chunks
    were derived from.

    Returns:
        (full_text, page_char_ranges) where page_char_ranges maps
        page_no → (start_char, end_char) within full_text.
    """
    parts: list[tuple[str, int]] = []
    for page_no in sorted(pages_dict.keys(), key=int):
        content = pages_dict[page_no]
        text_blocks = content.get("text_blocks", []) or []
        tables = content.get("tables", []) or []
        for block in text_blocks:
            if block:
                parts.append((block, int(page_no)))
        for table_entry in tables:
            parts.append((table_entry["markdown"], int(page_no)))

    if not parts:
        return "(no extractable content)", {}

    # Build text and track page character ranges
    page_ranges: dict[int, tuple[int, int]] = {}
    segments: list[str] = []
    offset = 0
    for text, page_no in parts:
        start = offset
        segments.append(text)
        offset += len(text) + 2  # +2 for "\n\n" separator
        # Expand range to cover all parts from this page
        if page_no in page_ranges:
            page_ranges[page_no] = (page_ranges[page_no][0], offset - 2)
        else:
            page_ranges[page_no] = (start, offset - 2)

    return "\n\n".join(segments), page_ranges


def _get_windowed_text(
    full_text: str,
    page_ranges: dict[int, tuple[int, int]],
    chunk_pages: list[int],
) -> str:
    """Extract ±_WINDOW_PAGES around the chunk's source pages."""
    if not chunk_pages:
        # No page info — use first/last 50K chars as context
        half = 50_000
        if len(full_text) <= half * 2:
            return full_text
        return full_text[:half] + "\n\n[...]\n\n" + full_text[-half:]

    all_pages = sorted(page_ranges.keys())
    if not all_pages:
        return full_text

    center = chunk_pages[0]
    min_page = max(all_pages[0], center - _WINDOW_PAGES)
    max_page = min(all_pages[-1], center + _WINDOW_PAGES)

    window_pages = [p for p in all_pages if min_page <= p <= max_page]
    if not window_pages:
        return full_text

    start = page_ranges[window_pages[0]][0]
    end = page_ranges[window_pages[-1]][1]
    return full_text[start:end]


def generate_cr_context(
    chunk_text: str,
    document_text: str,
    chunk_idx: int = 0,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Generate Contextual Retrieval context for a single chunk.

    Uses Gemini Flash via the same API client pattern as the RAPTOR
    tree builder (core.config.generate_with_retry).

    Returns a dict with keys: context, input_tokens, output_tokens, latency_ms, error.
    """
    from core.config import generate_with_retry, MODEL_FLASH
    from google.genai import types

    prompt = _CR_PROMPT_TEMPLATE.format(
        document_text=document_text,
        chunk_text=chunk_text,
    )

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=200,
    )

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            response = generate_with_retry(
                model=MODEL_FLASH,
                contents=prompt,
                config=config,
                max_retries=3,
                base_delay=2.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            context = (response.text or "").strip()
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

            return {
                "context": context,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": round(latency_ms, 1),
                "error": None,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            error_str = str(exc)
            if attempt < max_retries - 1:
                delay = (2 ** attempt) * 3
                logger.warning(
                    f"[CR] chunk {chunk_idx} attempt {attempt + 1}/{max_retries} "
                    f"failed: {error_str[:100]}, retrying in {delay:.0f}s"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"[CR] chunk {chunk_idx} all {max_retries} retries exhausted: "
                    f"{error_str[:200]}"
                )
                return {
                    "context": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": round(latency_ms, 1),
                    "error": error_str[:500],
                }

    # Should not reach here, but satisfy type checker
    return {"context": None, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "error": "unreachable"}


def _gcs_checkpoint_key(cache_key: str) -> str:
    """GCS blob key for the incremental CR checkpoint."""
    return f"cr-checkpoints/{cache_key}/cr_progress.pkl"


def _save_checkpoint_to_gcs(
    cache_key: str,
    chunks: list[dict],
    cr_log_chunks: list[dict],
    completed_idx: int,
    bucket_name: str = "raptor-assets",
) -> None:
    """Save incremental CR checkpoint to GCS (every 25 chunks)."""
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_gcs_checkpoint_key(cache_key))
        data = {
            "chunks": chunks,
            "cr_log_chunks": cr_log_chunks,
            "completed_idx": completed_idx,
        }
        blob.upload_from_string(
            pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
            content_type="application/octet-stream",
        )
        logger.info(f"[CR] GCS checkpoint saved: {completed_idx + 1} chunks done")
    except Exception as e:
        logger.warning(f"[CR] GCS checkpoint save failed: {e}")


def _load_checkpoint_from_gcs(
    cache_key: str,
    bucket_name: str = "raptor-assets",
) -> dict | None:
    """Load incremental CR checkpoint from GCS if available."""
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_gcs_checkpoint_key(cache_key))
        if not blob.exists():
            return None
        data = pickle.loads(blob.download_as_bytes())
        completed = data["completed_idx"] + 1
        total = len(data["chunks"])
        logger.info(f"[CR] GCS checkpoint found: {completed}/{total} chunks already done")
        return data
    except Exception as e:
        logger.warning(f"[CR] GCS checkpoint load failed: {e}")
        return None


def _delete_checkpoint_from_gcs(
    cache_key: str,
    bucket_name: str = "raptor-assets",
) -> None:
    """Delete checkpoint from GCS after successful completion."""
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_gcs_checkpoint_key(cache_key))
        if blob.exists():
            blob.delete()
            logger.info("[CR] GCS checkpoint cleaned up")
    except Exception as e:
        logger.warning(f"[CR] GCS checkpoint cleanup failed: {e}")


def build_cr_chunks(
    pages_dict: dict, cache_key: str | None = None,
) -> tuple[list[dict], dict]:
    """Build Contextual Retrieval chunks.

    1. Produces naive chunks via tree_builder._chunk_naive() (identical boundaries)
    2. Reconstructs full document text
    3. For each chunk, generates CR context via Gemini Flash and prepends it
    4. Saves incremental checkpoint to GCS every 25 chunks for resume

    When a GCS checkpoint exists for cache_key, resumes from where it left off.

    Returns:
        (cr_chunks, cr_log) where cr_log contains per-chunk generation details
        and aggregate statistics.
    """
    from ingestion.tree_builder import _chunk_naive

    # Step 1: Get naive chunks (same function, same boundaries)
    naive_chunks = _chunk_naive(pages_dict)
    logger.info(f"[CR] Naive chunking produced {len(naive_chunks)} chunks")

    # Step 2: Build full document text
    full_text, page_ranges = _build_full_document_text(pages_dict)
    use_windowed = len(full_text) > _MAX_DOCUMENT_CHARS
    if use_windowed:
        logger.info(
            f"[CR] Document exceeds {_MAX_DOCUMENT_TOKENS:,} token limit "
            f"({len(full_text):,} chars). Using windowed context."
        )

    # Step 3: Check for GCS checkpoint to resume from
    start_idx = 0
    cr_log_chunks: list[dict] = []

    if cache_key:
        checkpoint = _load_checkpoint_from_gcs(cache_key)
        if checkpoint is not None:
            start_idx = checkpoint["completed_idx"] + 1
            cr_log_chunks = checkpoint["cr_log_chunks"]
            # Restore chunk text from checkpoint (already has context prepended)
            for j in range(start_idx):
                naive_chunks[j] = checkpoint["chunks"][j]
            logger.info(f"[CR] Resuming from chunk {start_idx}/{len(naive_chunks)}")

    # Step 4: Generate and prepend context for remaining chunks
    total_input_tokens = sum(e.get("input_tokens", 0) for e in cr_log_chunks)
    total_output_tokens = sum(e.get("output_tokens", 0) for e in cr_log_chunks)
    total_latency_ms = sum(e.get("latency_ms", 0) for e in cr_log_chunks)
    fallback_count = sum(1 for e in cr_log_chunks if e.get("error"))
    t_start = time.perf_counter()

    for i in range(start_idx, len(naive_chunks)):
        chunk = naive_chunks[i]

        # Determine document text for this chunk
        if use_windowed:
            doc_text = _get_windowed_text(full_text, page_ranges, chunk.get("pages", []))
        else:
            doc_text = full_text

        result = generate_cr_context(
            chunk_text=chunk["text"],
            document_text=doc_text,
            chunk_idx=i,
        )

        log_entry = {
            "chunk_idx": i,
            "chunk_id": chunk["id"],
            "original_text_preview": chunk["text"][:150],
            "generated_context": result["context"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_ms": result["latency_ms"],
            "error": result["error"],
            "pages": chunk.get("pages", []),
            "windowed": use_windowed,
        }
        cr_log_chunks.append(log_entry)

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_latency_ms += result["latency_ms"]

        # Prepend context to chunk text (or leave unchanged on failure)
        if result["context"]:
            chunk["text"] = f"{result['context']}\n\n{chunk['text']}"
        else:
            fallback_count += 1
            logger.warning(
                f"[CR] chunk {i} context generation failed, using original text"
            )

        if (i + 1) % 25 == 0 or (i + 1) == len(naive_chunks):
            elapsed = time.perf_counter() - t_start
            logger.info(
                f"[CR] Progress: {i + 1}/{len(naive_chunks)} chunks "
                f"({elapsed:.0f}s elapsed, {fallback_count} fallbacks)"
            )
            # Incremental checkpoint to GCS every 25 chunks
            if cache_key:
                _save_checkpoint_to_gcs(
                    cache_key, naive_chunks, cr_log_chunks, i,
                )

    # Build aggregate log
    generation_time_s = time.perf_counter() - t_start

    # Estimate cost (Gemini 2.5 Flash pricing: $0.15/M input, $0.60/M output)
    estimated_cost = (total_input_tokens * 0.15 / 1_000_000) + (total_output_tokens * 0.60 / 1_000_000)

    cr_log = {
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "total_chunks": len(naive_chunks),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": round(estimated_cost, 4),
        "avg_context_length_tokens": (
            round(total_output_tokens / len(naive_chunks), 1) if naive_chunks else 0
        ),
        "generation_time_seconds": round(generation_time_s, 1),
        "fallback_count": fallback_count,
        "windowed_mode": use_windowed,
        "chunks": cr_log_chunks,
    }

    logger.info(
        f"[CR] Complete: {len(naive_chunks)} chunks, "
        f"{total_input_tokens:,} input tokens, {total_output_tokens:,} output tokens, "
        f"${estimated_cost:.4f} estimated cost, {generation_time_s:.0f}s, "
        f"{fallback_count} fallbacks"
    )

    # Clean up GCS checkpoint after successful completion
    if cache_key:
        _delete_checkpoint_from_gcs(cache_key)

    return naive_chunks, cr_log
