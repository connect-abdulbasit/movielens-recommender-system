"""Run all models on the official u1..u5 splits and save a results table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "results.csv"
TOP_K = 10


def evaluate_fold(fold: int, k: int = TOP_K) -> pd.DataFrame:
    """Fit every model on ``u{fold}.base`` and score it on ``u{fold}.test``.

    Ranking metrics use only movies not rated in train; relevant = test
    rating >= 4. RMSE/MAE are computed on all test ratings (not applicable to
    models that do not predict ratings, e.g. Naive Bayes -> NaN).

    Returns:
        DataFrame with one row per model and columns
        ``model, precision@k, recall@k, rmse, mae``.
    """
    raise NotImplementedError("TODO")


def run_all(k: int = TOP_K, out_path: Path = RESULTS_PATH) -> pd.DataFrame:
    """Evaluate all five folds, average per model, and write the CSV.

    Returns:
        The averaged results table (also saved to ``out_path``).
    """
    raise NotImplementedError("TODO")
