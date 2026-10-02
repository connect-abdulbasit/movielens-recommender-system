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
    raise NotImplementedError("TODO: |top-k ∩ relevant| / k")


def recall_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Fraction of relevant movies found in the top-``k``.

    Returns 0.0 when ``relevant`` is empty.
    """
    raise NotImplementedError("TODO: |top-k ∩ relevant| / |relevant|")


def rmse(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Root mean squared error."""
    raise NotImplementedError("TODO")


def mae(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Mean absolute error."""
    raise NotImplementedError("TODO")
