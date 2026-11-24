from __future__ import annotations

from typing import Collection, Sequence

import numpy as np

RELEVANT_THRESHOLD = 4

def precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k).intersection(relevant))
    return hits / k

def recall_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    if k <= 0 or len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k).intersection(relevant))
    return hits / len(relevant)

def rmse(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

def mae(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))
