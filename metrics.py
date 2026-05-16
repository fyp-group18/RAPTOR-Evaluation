"""
Standard evaluation metrics for RAG benchmarking.

All functions accept single prediction/ground_truth pairs.
Dataset-level scores are computed by averaging over all pairs.
When multiple ground truths exist, take max score across all ground truths.
"""

import re
import string

from Levenshtein import distance as levenshtein_distance


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace, remove punctuation and articles."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def anls_score(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity.

    Standard metric for DocVQA. Returns (1 - NLS) if NLS < threshold, else 0.0.
    """
    pred = _normalize(prediction)
    gt = _normalize(ground_truth)
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    max_len = max(len(pred), len(gt))
    dist = levenshtein_distance(pred, gt)
    nls = dist / max_len
    return 1.0 - nls if nls < threshold else 0.0


def f1_token_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1. Standard metric for QASPER and extractive QA."""
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    num_common = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization. Standard metric for open-domain QA."""
    return 1.0 if _normalize(prediction) == _normalize(ground_truth) else 0.0


def score_with_multiple_gts(
    prediction: str,
    ground_truths: list[str],
    metric_fn,
) -> float:
    """Take max score across multiple valid ground truth answers."""
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)
