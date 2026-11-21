"""Evaluation metrics.

Ranking metrics take a ranked list of 1-based movie ids that were NOT rated in
the training set. An item is *relevant* if its test rating is >= 4.
"""
from __future__ import annotations

from typing import Collection, Sequence

import numpy as np

RELEVANT_THRESHOLD = 4


def precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Fraction of the top-``k`` recommendations that are relevant.

    Args:
        recommended: Ranked movie ids, best first.
        relevant: Ids of relevant (test rating >= 4) movies.
        k: Cut-off.
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k).intersection(relevant))
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Fraction of relevant movies found in the top-``k``.

    Returns 0.0 when ``relevant`` is empty.
    """
    if k <= 0 or len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k).intersection(relevant))
    return hits / len(relevant)


def rmse(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Root mean squared error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Mean absolute error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))
