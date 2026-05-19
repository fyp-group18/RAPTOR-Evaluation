"""LLM reranker for evaluation — duplicates production logic from modules/reranker.py.

Text-only variant (no GCS image loading). Uses gemini-2.5-flash with 0-10
relevance scoring, identical prompt and schema to production.
"""

from __future__ import annotations

import json
import logging
import os
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_FLASH = "gemini-2.5-flash"

_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")


class ChunkScore(BaseModel):
    chunk_id: str
    score: int = Field(ge=0, le=10, description="Relevance score 0-10")


class RerankerResponse(BaseModel):
    scores: list[ChunkScore]


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: types.GenerateContentConfig,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> types.GenerateContentResponse:
    """Retry wrapper with exponential backoff on 429."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"[EvalReranker] 429 (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                raise


class EvalReranker:
    """LLM reranker for evaluation using Gemini Flash."""

    def __init__(self, min_score: int = 3, top_k: int = 10):
        self.min_score = min_score
        self.top_k = top_k
        self.client = genai.Client(
            vertexai=True,
            project=_PROJECT_ID,
            location=_LOCATION,
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._call_count = 0

    def rerank(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, int]]:
        """Score and rerank candidates.

        Args:
            query: User query text.
            candidates: List of (chunk_id, chunk_text) tuples.

        Returns:
            Sorted list of (chunk_id, score) tuples, filtered by min_score,
            capped at top_k.
        """
        if not candidates:
            return []

        if len(candidates) <= 2:
            return [(cid, 10) for cid, _ in candidates]

        prompt_header = (
            "You are a relevance reranker for industrial equipment technical manuals.\n"
            "Score each CHUNK 0-10 based on how well it answers the user query.\n\n"
            "SCORING GUIDE:\n"
            "  9-10: Chunk directly contains the fault code, troubleshooting step, or exact specification asked about\n"
            "  7-8:  Chunk covers the correct system/component and contains related diagnostic info\n"
            "  4-6:  Chunk is from the correct manual section but only partially relevant\n"
            "  1-3:  Chunk is from the same manual but wrong section/topic\n"
            "  0:    Completely irrelevant\n\n"
            f'User Query: "{query}"\n\nChunks:\n'
        )

        contents: list = [prompt_header]
        for cid, text in candidates:
            contents.append(
                f"\n--- CHUNK ID: {cid} ---\n"
                f"{text[:2000]}\n"
            )

        try:
            t0 = time.time()
            res = _generate_with_retry(
                client=self.client,
                model=MODEL_FLASH,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RerankerResponse,
                    temperature=0.0,
                ),
            )
            latency_ms = int((time.time() - t0) * 1000)

            usage = getattr(res, "usage_metadata", None)
            if usage:
                inp = getattr(usage, "prompt_token_count", 0) or 0
                out = getattr(usage, "candidates_token_count", 0) or 0
                self.total_input_tokens += inp
                self.total_output_tokens += out

            scores_data = json.loads(res.text).get("scores", [])
            scores_map = {s["chunk_id"]: s["score"] for s in scores_data}

            scored = [
                (cid, scores_map.get(cid, 0)) for cid, _ in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            if self.min_score > 0:
                filtered = [(cid, s) for cid, s in scored if s >= self.min_score]
                if len(filtered) < 2:
                    filtered = scored[:2]
                result = filtered[: self.top_k]
            else:
                result = scored[: self.top_k]

            self._call_count += 1
            if self._call_count % 50 == 0:
                logger.info(
                    f"[EvalReranker] {self._call_count} queries reranked "
                    f"({self.total_input_tokens} input tokens, "
                    f"{self.total_output_tokens} output tokens, "
                    f"last call {latency_ms}ms)"
                )

            return result

        except Exception:
            logger.exception("[EvalReranker] LLM rerank failed; returning original order")
            return [(cid, 0) for cid, _ in candidates[: self.top_k]]
