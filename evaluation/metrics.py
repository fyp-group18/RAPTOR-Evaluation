"""Retrieval metrics: precision@k, recall@k, MRR, nDCG@k.

All functions operate on a single query's ranked results vs ground-truth relevance.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k that are relevant."""
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_ids) / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of first relevant result."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevance_map: {chunk_id: relevance_score} where score in {0, 1, 2}.
        k: Cutoff rank.
    """
    def dcg(scores: list[float], cutoff: int) -> float:
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:cutoff]))

    actual_scores = [relevance_map.get(rid, 0) for rid in retrieved_ids[:k]]
    ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]

    ideal = dcg(ideal_scores, k)
    if ideal == 0:
        return 0.0
    return dcg(actual_scores, k) / ideal
