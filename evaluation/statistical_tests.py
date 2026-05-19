"""Paired bootstrap resampling for significance testing."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Two-sided paired bootstrap test for difference in means.

    Args:
        scores_a: Per-query metric values for system A (baseline).
        scores_b: Per-query metric values for system B (proposed).
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with delta, p_value, ci_95_lower, ci_95_upper, method.
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert n == len(scores_b), "Score lists must have equal length"

    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    observed_delta = float(np.mean(arr_b) - np.mean(arr_a))

    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = np.mean(arr_b[idx]) - np.mean(arr_a[idx])

    # Two-sided p-value: fraction of bootstrap deltas at least as extreme
    # as observed, centered around the bootstrap mean
    p_value = float(np.mean(np.abs(deltas - np.mean(deltas)) >= np.abs(observed_delta)))
    ci_lower, ci_upper = np.percentile(deltas, [2.5, 97.5]).tolist()

    return {
        "delta": observed_delta,
        "p_value": p_value,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "method": f"paired_bootstrap_{n_bootstrap}",
    }
