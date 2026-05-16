"""
Standalone RAPTOR tree builder for the QASPER benchmark.

Replicates the algorithm from backend/modules/ingestion.py without any DB,
GCS, or document-ingestion dependencies. All computation is in-memory.

Algorithm (from RAPTOR paper, matching ingestion._build_raptor_tree):
  1. Take all level-N nodes (embeddings + texts)
  2. UMAP reduce to lower dimension
  3. GMM clustering with BIC-optimised k
  4. Per cluster: concatenate texts → Gemini Flash summary → embed summary
  5. Summary nodes become level N+1
  6. Recurse until single node or max_levels reached
  7. Return all nodes (all levels) flat list
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import numpy as np
import umap
from pydantic import BaseModel, Field
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

# Limit parallel summarisation calls to avoid quota spikes
_MAX_SUMMARY_THREADS = 4


class _ClusterSummary(BaseModel):
    summary_text: str = Field(description="A comprehensive summary of the concepts.")
    retained_specs: list[str] = Field(
        description="Array of ALL exact numerical values and specs."
    )


def _cluster_summary(cluster_texts: list[str], level: int = 0) -> str | None:
    """Summarise a cluster of texts using Gemini Flash.

    Mirrors ingestion._cluster_summary exactly (same prompt, same retry
    logic, same JSON schema) without the telemetry/threading wrappers.

    Returns formatted summary string or None when all retries fail.
    """
    from core.config import genai_client, MODEL_FLASH
    from google.genai import types

    combined = "\n\n--- NEXT EXCERPT ---\n\n".join(cluster_texts)[:120000]
    prompt = f"Synthesize these manual excerpts. EXCERPTS:\n{combined}"
    permissive_decoder = json.JSONDecoder(strict=False)

    for attempt in range(6):
        current_temp = 0.1 + (attempt * 0.1)
        try:
            response = genai_client.models.generate_content(
                model=MODEL_FLASH,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_ClusterSummary,
                    temperature=current_temp,
                ),
            )
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "quota" in error_str.lower():
                sleep_secs = (2**attempt) * 5
                logger.warning(
                    f"[RAPTOR Throttled] attempt {attempt + 1}/6, sleeping {sleep_secs}s..."
                )
                time.sleep(sleep_secs)
            else:
                logger.warning(f"[RAPTOR] attempt {attempt + 1}/6 LLM call failed: {exc}")
                time.sleep(2)
            continue

        # Parse JSON response
        try:
            clean = response.text.strip().lstrip("\ufeff")
            clean = re.sub(r"^```json\s*", "", clean, flags=re.MULTILINE)
            clean = re.sub(r"^```\s*", "", clean, flags=re.MULTILINE)
            clean = re.sub(r"```$", "", clean, flags=re.MULTILINE).strip()
            result = permissive_decoder.decode(clean)
            summary = f"SUMMARY:\n{result['summary_text']}\n\nCRITICAL SPECS RETAINED:\n"
            if isinstance(result.get("retained_specs"), list):
                summary += "\n".join(f"- {s}" for s in result["retained_specs"])
            return summary
        except (json.JSONDecodeError, KeyError, AttributeError) as exc:
            logger.warning(f"[RAPTOR JSON parse] attempt {attempt + 1}/6: {exc}")
            time.sleep(2)

    logger.error(
        f"[RAPTOR] Level {level}: all 6 summary retries failed — dropping cluster node."
    )
    return None


def build_raptor_tree(
    chunks: list[dict],
    max_levels: int = 3,
) -> list[dict]:
    """Build a RAPTOR tree in-memory from pre-embedded leaf chunks.

    Args:
        chunks: List of dicts with keys:
            - "text" (str): chunk text
            - "embedding" (np.ndarray): (3072,) float32 embedding
            - "level" (int): must be 0 for leaf nodes
        max_levels: Maximum number of summary hierarchy levels to build.

    Returns:
        Flat list of all nodes — original leaf chunks plus summary nodes.
        Each node has the same dict shape as the input plus "level" ≥ 1 for
        summary nodes.

    UMAP / GMM parameters mirror ingestion._build_raptor_tree exactly:
        n_neighbors = min(15, max(2, n_samples - 1))
        n_components = min(10, max(2, n_samples - 2))
        metric      = "cosine"
        GMM max_k   = max(2, min(50, n_samples // 3))
        covariance  = "diag", selection by BIC
    """
    from modules.embeddings import embed_batch

    all_nodes: list[dict] = list(chunks)
    current_nodes: list[dict] = list(chunks)
    current_level = 0

    while len(current_nodes) > 1 and current_level < max_levels:
        current_level += 1
        n_samples = len(current_nodes)
        logger.info(
            f"[RAPTOR] Building hierarchy level {current_level} "
            f"({n_samples} input nodes)..."
        )

        embeds = np.array([n["embedding"] for n in current_nodes], dtype=np.float32)

        # --- UMAP + GMM clustering ---
        if n_samples <= 5:
            cluster_labels: list[int] = [0] * n_samples
        else:
            n_neighbors = min(15, max(2, n_samples - 1))
            n_components = min(10, max(2, n_samples - 2))
            reduced = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric="cosine",
                init="random",
            ).fit_transform(embeds)

            max_k = max(2, min(50, int(n_samples / 3)))
            best_gmm, best_bic = None, np.inf
            for k in range(1, max_k + 1):
                try:
                    gmm = GaussianMixture(
                        n_components=k, covariance_type="diag"
                    ).fit(reduced)
                except ValueError:
                    continue
                bic = gmm.bic(reduced)
                if bic < best_bic:
                    best_bic, best_gmm = bic, gmm

            if best_gmm is None:
                cluster_labels = [0] * n_samples
            else:
                cluster_labels = best_gmm.predict(reduced).tolist()

        # Group nodes by cluster
        clusters: dict[int, list[dict]] = {}
        for i, label in enumerate(cluster_labels):
            clusters.setdefault(int(label), []).append(current_nodes[i])

        logger.info(
            f"[RAPTOR] Level {current_level}: {len(clusters)} clusters "
            f"(avg {n_samples / max(len(clusters), 1):.1f} nodes/cluster)"
        )

        # --- Parallel summarisation ---
        def _process_cluster(nodes: list[dict]) -> dict | None:
            texts = [n["text"] for n in nodes]
            summary = _cluster_summary(texts, level=current_level)
            if summary is None:
                return None
            return {"text": summary, "embedding": None, "level": current_level}

        next_level_nodes: list[dict] = []
        # Per-cluster timeout: 180s covers worst-case large clusters.
        # as_completed timeout is total wall time for all clusters at this level.
        _CLUSTER_TIMEOUT = 180
        _LEVEL_TIMEOUT = _CLUSTER_TIMEOUT * len(clusters) + 30
        with ThreadPoolExecutor(max_workers=_MAX_SUMMARY_THREADS) as executor:
            futures = {
                executor.submit(_process_cluster, nodes): cid
                for cid, nodes in clusters.items()
            }
            try:
                for fut in as_completed(futures, timeout=_LEVEL_TIMEOUT):
                    try:
                        node = fut.result(timeout=_CLUSTER_TIMEOUT)
                    except Exception as exc:
                        logger.warning(
                            f"[RAPTOR] Level {current_level} cluster failed: {exc} — dropping node"
                        )
                        continue
                    if node is not None:
                        next_level_nodes.append(node)
            except FuturesTimeoutError:
                logger.warning(
                    f"[RAPTOR] Level {current_level}: global timeout hit — "
                    f"proceeding with {len(next_level_nodes)} nodes collected so far"
                )
                for fut in futures:
                    fut.cancel()

        if not next_level_nodes:
            logger.warning(
                f"[RAPTOR] Level {current_level}: all clusters failed. "
                "Stopping hierarchy build."
            )
            break

        # --- Embed summary nodes ---
        texts_to_embed = [n["text"] for n in next_level_nodes]
        embeddings_list = embed_batch(texts_to_embed)

        valid_nodes: list[dict] = []
        for node, emb in zip(next_level_nodes, embeddings_list):
            if emb is not None:
                node["embedding"] = np.array(emb, dtype=np.float32)
                valid_nodes.append(node)
            else:
                logger.warning(
                    f"[RAPTOR] Level {current_level}: embedding failed for a summary node — skipping."
                )

        if not valid_nodes:
            logger.warning(
                f"[RAPTOR] Level {current_level}: all embeddings failed. Stopping."
            )
            break

        logger.info(
            f"[RAPTOR] Level {current_level}: {len(valid_nodes)} summary nodes embedded."
        )
        all_nodes.extend(valid_nodes)
        current_nodes = valid_nodes

    return all_nodes
