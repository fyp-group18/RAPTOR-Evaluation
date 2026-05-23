"""Core retrieval evaluator: load tree, run queries, compute metrics.

Loads pre-built RAPTOR trees via ingestion.tree_resolver, runs retrieval
against an annotated query dataset, and produces per-sequence metric results.
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field

import numpy as np

from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from evaluation.statistical_tests import paired_bootstrap_test
from ingestion.ablation_configs import ABLATION_CASES
from ingestion.tree_resolver import resolve_tree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedNode:
    chunk_id: str
    text: str
    score: float
    level: int
    source_pages: list[int]


@dataclass
class QueryResult:
    query_id: str
    query_type: str
    difficulty: str
    retrieved_ids: list[str]
    match_mode: str  # "exact" or "page_fallback"
    gt_chunk_id: str | None = None
    per_k_metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cosine similarity (matches benchmarks/qasper_benchmark.py)
# ---------------------------------------------------------------------------


def _cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """(D,) query vs (N, D) matrix → (N,) cosine similarities."""
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-10:
        return np.zeros(matrix.shape[0])
    q_hat = query / q_norm

    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    m_norms = np.maximum(m_norms, 1e-10)
    m_hat = matrix / m_norms

    return m_hat @ q_hat


# ---------------------------------------------------------------------------
# Parent-swap aware matching
# ---------------------------------------------------------------------------


def build_parent_child_map(tree_nodes: list[dict]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Build parent↔child lookups from tree nodes.

    Returns:
        parent_to_children: {parent_id: set of child_ids}
        child_to_parent: {child_id: parent_id}
    """
    parent_to_children: dict[str, set[str]] = {}
    child_to_parent: dict[str, str] = {}
    for node in tree_nodes:
        pid = node.get("parent_id")
        nid = str(node.get("id", ""))
        if pid:
            pid = str(pid)
            child_to_parent[nid] = pid
            parent_to_children.setdefault(pid, set()).add(nid)
    return parent_to_children, child_to_parent


def is_hit(
    retrieved_ids: list[str],
    gt_chunk_id: str,
    parent_to_children: dict[str, set[str]],
    child_to_parent: dict[str, str],
    k: int,
) -> bool:
    """Check if ground truth is "hit" in top-k retrieved results.

    A hit occurs if:
    1. gt_chunk_id is directly in top-k (exact match), OR
    2. gt_chunk_id is a parent AND any of its children are in top-k
       (parent-swap: retrieving any child delivers the parent context)
    """
    top_k_set = set(retrieved_ids[:k])

    if gt_chunk_id in top_k_set:
        return True

    children = parent_to_children.get(gt_chunk_id, set())
    if children and (children & top_k_set):
        return True

    return False


def find_best_rank(
    retrieved_ids: list[str],
    gt_chunk_id: str,
    parent_to_children: dict[str, set[str]],
) -> int | None:
    """Find the rank of the best hit (for MRR/NDCG).

    Returns 0-indexed rank, or None if no hit.
    """
    children = parent_to_children.get(gt_chunk_id, set())

    for i, rid in enumerate(retrieved_ids):
        if rid == gt_chunk_id:
            return i
        if rid in children:
            return i

    return None


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------


def _embed_query(text: str, multimodal: bool) -> np.ndarray | None:
    """Embed a query using the appropriate model.

    Uses modules.embeddings.embed which dispatches to Gemini Embedding 2
    (multimodal) or text-embedding-004 (text-only) depending on the config
    set up in core.config.
    """
    from modules.embeddings import embed

    emb = embed(text=text)
    if emb is not None:
        return np.array(emb, dtype=np.float32)
    return None


# ---------------------------------------------------------------------------
# Tree loading and retrieval
# ---------------------------------------------------------------------------


class TreeIndex:
    """Wraps a loaded tree for efficient retrieval."""

    def __init__(self, nodes: list[dict], retrieval_mode: str):
        self.all_nodes = nodes
        self.retrieval_mode = retrieval_mode

        # Filter searchable nodes: exclude is_parent (no embedding) and
        # respect retrieval_mode
        searchable = []
        for node in nodes:
            if node.get("is_parent"):
                continue
            if node.get("embedding") is None:
                continue
            if retrieval_mode == "flat" and node.get("level", 0) != 0:
                continue
            searchable.append(node)

        self.searchable_nodes = searchable
        if searchable:
            self.embedding_matrix = np.array(
                [n["embedding"] for n in searchable], dtype=np.float32
            )
        else:
            self.embedding_matrix = np.empty((0, 0), dtype=np.float32)

        # Build parent lookup for retrieval expansion
        self._parent_map: dict[str, dict] = {}
        for i, node in enumerate(nodes):
            if node.get("is_parent"):
                self._parent_map[node.get("id", f"node_{i}")] = node

        # Parent-swap maps for matching
        self.parent_to_children, self.child_to_parent = build_parent_child_map(nodes)

    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedNode]:
        """Return top-k nodes ranked by cosine similarity to query."""
        if len(self.searchable_nodes) == 0:
            return []

        scores = _cosine_similarity_batch(query_embedding, self.embedding_matrix)
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            node = self.searchable_nodes[idx]
            pages = node.get("pages", [])
            # Coerce malformed pages (string "93" → [93], int 93 → [93])
            if isinstance(pages, str):
                pages = [int(p) for p in pages.split(",") if p.strip().isdigit()]
            elif isinstance(pages, (int, float)):
                pages = [int(pages)]
            if not pages:
                pages = list(range(
                    node.get("page_start", 0),
                    node.get("page_end", 0) + 1
                )) if node.get("page_start") else []

            results.append(RetrievedNode(
                chunk_id=node.get("id", f"node_{idx}"),
                text=node["text"],
                score=float(scores[idx]),
                level=node.get("level", 0),
                source_pages=pages,
            ))
        return results


def load_tree_index(
    cache_key: str,
    ablation_name: str,
    run_id: str | None = None,
    bucket: str | None = None,
) -> tuple[TreeIndex, dict]:
    """Load a tree from GCS/local mirror and construct a TreeIndex.

    Returns (TreeIndex, manifest_dict).
    """
    config = ABLATION_CASES[ablation_name]
    ablation_label = config["label"]

    tree_bytes, manifest = resolve_tree(
        cache_key=cache_key,
        ablation_label=ablation_label,
        run_id=run_id,
        bucket=bucket,
    )

    nodes: list[dict] = pickle.loads(tree_bytes)
    retrieval_mode = manifest.get("ablation_config", {}).get(
        "retrieval_mode", config.get("retrieval_mode", "collapsed")
    )

    index = TreeIndex(nodes, retrieval_mode)
    logger.info(
        f"Loaded tree [{ablation_label}]: "
        f"{len(index.searchable_nodes)} searchable nodes, "
        f"mode={retrieval_mode}"
    )
    return index, manifest


# ---------------------------------------------------------------------------
# Chunk ID matching (exact + page-level fallback)
# ---------------------------------------------------------------------------


def _resolve_match_ids(
    retrieved: list[RetrievedNode],
    relevant_chunks: list[dict],
    relevant_pages: list[int],
) -> tuple[list[str], set[str], dict[str, int], str]:
    """Determine matching strategy and return resolved IDs.

    Returns:
        retrieved_ids: Ordered list of retrieved IDs (may be remapped in fallback mode).
        relevant_ids: Set of relevant IDs (binary).
        relevance_map: {id: relevance_score} for graded metrics.
        match_mode: "exact" or "page_fallback".
    """
    # Ground-truth chunk IDs and relevance scores
    gt_chunk_ids = {rc["chunk_id"] for rc in relevant_chunks}
    gt_relevance_map = {rc["chunk_id"]: rc.get("relevance", 1) for rc in relevant_chunks}

    # Retrieved IDs in rank order
    retrieved_ids = [r.chunk_id for r in retrieved]

    # Check if any exact chunk_id match exists
    retrieved_set = set(retrieved_ids)
    has_exact_match = bool(retrieved_set & gt_chunk_ids)

    if has_exact_match or not relevant_pages:
        # Use exact matching
        return retrieved_ids, gt_chunk_ids, gt_relevance_map, "exact"

    # Page-level fallback: a retrieved chunk is relevant if its pages overlap
    # with relevant_pages
    logger.debug("Using page-level fallback matching")
    relevant_page_set = set(relevant_pages)

    # Remap: create synthetic IDs based on page overlap
    # A retrieved node is "relevant" if it has page overlap
    page_relevant_retrieved: set[str] = set()
    for r in retrieved:
        if set(r.source_pages) & relevant_page_set:
            page_relevant_retrieved.add(r.chunk_id)

    if not page_relevant_retrieved:
        # No page info on nodes — cannot match
        return retrieved_ids, gt_chunk_ids, gt_relevance_map, "page_fallback"

    # For page fallback, the "relevant set" is the set of retrieved node IDs
    # that overlap with ground-truth pages. The relevance score is 1 (binary)
    # since we can't determine graded relevance via page overlap alone.
    fallback_relevant_ids = page_relevant_retrieved
    fallback_relevance_map = {rid: 1 for rid in fallback_relevant_ids}

    return retrieved_ids, fallback_relevant_ids, fallback_relevance_map, "page_fallback"


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------


def evaluate_single_query(
    tree_index: TreeIndex,
    query: dict,
    query_embedding: np.ndarray,
    k_prime: int,
) -> tuple[QueryResult, list[RetrievedNode]]:
    """Run retrieval for a single query. Returns (QueryResult, retrieved_nodes)."""
    retrieved = tree_index.retrieve(query_embedding, top_k=k_prime)

    relevant_chunks = query.get("relevant_chunks", [])
    relevant_pages = query.get("relevant_pages", [])

    retrieved_ids, relevant_ids, relevance_map, match_mode = _resolve_match_ids(
        retrieved, relevant_chunks, relevant_pages
    )

    gt_chunk_id = relevant_chunks[0]["chunk_id"] if relevant_chunks else None

    return QueryResult(
        query_id=query["query_id"],
        query_type=query["query_type"],
        difficulty=query.get("difficulty", "unknown"),
        retrieved_ids=retrieved_ids,
        match_mode=match_mode,
        gt_chunk_id=gt_chunk_id,
        per_k_metrics={
            "relevant_ids": relevant_ids,
            "relevance_map": relevance_map,
        },
    ), retrieved


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def compute_metrics_for_results(
    results: list[QueryResult],
    k_values: list[int],
    parent_to_children: dict[str, set[str]] | None = None,
    child_to_parent: dict[str, str] | None = None,
) -> dict:
    """Compute aggregate metrics over a list of QueryResult objects.

    When parent maps are provided, exact-match queries use parent-swap aware
    matching (is_hit/find_best_rank). Page-fallback queries use set-based metrics.

    Returns dict with per-metric averages and per-query score lists (for stat tests).
    """
    max_k = max(k_values) if k_values else 10
    p2c = parent_to_children or {}
    c2p = child_to_parent or {}

    per_query_scores: dict[str, list[float]] = {
        f"precision@{k}": [] for k in k_values
    }
    for k in k_values:
        per_query_scores[f"recall@{k}"] = []
    per_query_scores["mrr"] = []
    per_query_scores[f"ndcg@{max_k}"] = []

    for qr in results:
        retrieved_ids = qr.retrieved_ids

        if qr.gt_chunk_id is not None:
            # Exact match mode: use parent-swap aware matching
            gt = qr.gt_chunk_id
            relevance = qr.per_k_metrics["relevance_map"].get(gt, 2)

            for k in k_values:
                hit = is_hit(retrieved_ids, gt, p2c, c2p, k)
                per_query_scores[f"precision@{k}"].append(1.0 / k if hit else 0.0)
                per_query_scores[f"recall@{k}"].append(1.0 if hit else 0.0)

            rank = find_best_rank(retrieved_ids, gt, p2c)
            per_query_scores["mrr"].append(1.0 / (rank + 1) if rank is not None else 0.0)

            if rank is not None and rank < max_k:
                dcg = relevance / math.log2(rank + 2)
                idcg = relevance / math.log2(2)
                per_query_scores[f"ndcg@{max_k}"].append(dcg / idcg)
            else:
                per_query_scores[f"ndcg@{max_k}"].append(0.0)
        else:
            # Page-fallback mode: use set-based metrics
            relevant_ids = qr.per_k_metrics["relevant_ids"]
            relevance_map = qr.per_k_metrics["relevance_map"]

            for k in k_values:
                per_query_scores[f"precision@{k}"].append(
                    precision_at_k(retrieved_ids, relevant_ids, k)
                )
                per_query_scores[f"recall@{k}"].append(
                    recall_at_k(retrieved_ids, relevant_ids, k)
                )

            per_query_scores["mrr"].append(mrr(retrieved_ids, relevant_ids))
            per_query_scores[f"ndcg@{max_k}"].append(
                ndcg_at_k(retrieved_ids, relevance_map, max_k)
            )

    # Compute means
    metrics = {name: float(np.mean(scores)) for name, scores in per_query_scores.items()}
    metrics["num_queries"] = len(results)

    return metrics, per_query_scores


def evaluate_ablation(
    cache_key: str,
    ablation_name: str,
    dataset: dict,
    k_values: list[int],
    run_id: str | None = None,
    bucket: str | None = None,
    k_prime: int = 30,
    retrieval_mode: str = "embedding_only",
    reranker=None,
) -> tuple[dict, dict[str, list[float]]]:
    """Run full evaluation for one ablation configuration.

    Args:
        k_prime: Initial cosine candidate pool size.
        retrieval_mode: "embedding_only" or "embedding_reranked".
        reranker: EvalReranker instance (required when retrieval_mode is "embedding_reranked").

    Returns:
        (metrics_dict, per_query_scores_dict) where metrics_dict has "overall"
        and "by_query_type" breakdowns.
    """
    tree_index, manifest = load_tree_index(
        cache_key, ablation_name, run_id=run_id, bucket=bucket
    )

    multimodal = ABLATION_CASES[ablation_name].get("multimodal_embedding", True)
    queries = dataset["queries"]

    logger.info(
        f"Evaluating {len(queries)} queries (mode={retrieval_mode}, k'={k_prime})"
    )
    query_results: list[QueryResult] = []
    fallback_count = 0

    for i, query in enumerate(queries):
        query_emb = _embed_query(query["query_text"], multimodal=multimodal)
        if query_emb is None:
            logger.warning(f"Failed to embed query {query['query_id']}, skipping")
            continue

        qr, retrieved_nodes = evaluate_single_query(
            tree_index, query, query_emb, k_prime
        )

        if retrieval_mode == "embedding_reranked" and reranker is not None:
            candidates = [(r.chunk_id, r.text) for r in retrieved_nodes]
            reranked = reranker.rerank(query["query_text"], candidates)
            qr.retrieved_ids = [cid for cid, _score in reranked]

        query_results.append(qr)

        if qr.match_mode == "page_fallback":
            fallback_count += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(queries)} queries")

    if fallback_count > 0:
        logger.warning(
            f"Page-level fallback used for {fallback_count}/{len(query_results)} queries "
            f"(chunk IDs from different chunking strategy)"
        )

    p2c = tree_index.parent_to_children
    c2p = tree_index.child_to_parent

    # Overall metrics
    overall_metrics, overall_scores = compute_metrics_for_results(
        query_results, k_values, p2c, c2p
    )

    # By query_type
    by_type: dict[str, dict] = {}
    by_type_scores: dict[str, dict[str, list[float]]] = {}
    query_types = sorted(set(qr.query_type for qr in query_results))

    for qt in query_types:
        type_results = [qr for qr in query_results if qr.query_type == qt]
        type_metrics, type_scores = compute_metrics_for_results(
            type_results, k_values, p2c, c2p
        )
        by_type[qt] = type_metrics
        by_type_scores[qt] = type_scores

    result = {
        "overall": overall_metrics,
        "by_query_type": by_type,
    }

    # Combine all per-query score dicts for stat tests
    all_scores = {"overall": overall_scores}
    all_scores.update(by_type_scores)

    if reranker is not None:
        logger.info(
            f"Reranker token usage: {reranker.total_input_tokens} input, "
            f"{reranker.total_output_tokens} output"
        )

    return result, all_scores


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


def run_statistical_comparisons(
    scores_a: dict[str, dict[str, list[float]]],
    scores_b: dict[str, dict[str, list[float]]],
    label_a: str,
    label_b: str,
    metrics_to_test: list[str],
) -> dict:
    """Run paired bootstrap tests between two systems.

    Args:
        scores_a/b: {"overall": {"recall@5": [...], ...}, "table": {...}, ...}
        label_a/b: System labels for output keys.
        metrics_to_test: Which metrics to run significance tests on.

    Returns:
        Dict of test results keyed by "{label_b}_vs_{label_a}_{scope}_{metric}".
    """
    tests = {}
    for scope in scores_a:
        for metric in metrics_to_test:
            if metric not in scores_a.get(scope, {}):
                continue
            if metric not in scores_b.get(scope, {}):
                continue

            sa = scores_a[scope][metric]
            sb = scores_b[scope][metric]
            if len(sa) != len(sb):
                logger.warning(
                    f"Score length mismatch for {scope}/{metric}: "
                    f"{len(sa)} vs {len(sb)}, skipping"
                )
                continue

            key = f"{label_b}_vs_{label_a}_{scope}_{metric}"
            tests[key] = paired_bootstrap_test(sa, sb)

    return tests
