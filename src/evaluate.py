"""Run all models on the official u1..u5 splits and save a results table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics
from src.baselines import PopularityRecommender, RandomRecommender
from src.data_loader import (
    GENRES, N_FOLDS, build_matrix, item_feature_matrix, load_movies, load_ratings,
)
from src.naive_bayes import NaiveBayesRecommender
from src.neural_recommender import NeuralRecommender

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "results.csv"
TOP_K = 10


def build_models(item_features: np.ndarray, feature_names: list[str]) -> dict:
    """Every model to compare, as ``name -> (model, fit function)``.

    The fit functions differ because the models need different inputs.
    """
    random, popularity = RandomRecommender(), PopularityRecommender()
    bayes, network = NaiveBayesRecommender(), NeuralRecommender()
    genres = item_features[:, : len(GENRES)]
    return {
        "Random": (random, random.fit),
        "Popularity": (popularity, popularity.fit),
        "Naive Bayes": (bayes, lambda matrix: bayes.fit(matrix, genres)),
        "Neural Network": (network, lambda matrix: network.fit(matrix, item_features, feature_names)),
    }


def ranking_scores(model, train_matrix: np.ndarray, test, k: int) -> tuple[float, float]:
    """Mean Precision@k and Recall@k over users that have a relevant test movie.

    Relevant = test rating >= 4. ``model.recommend`` only offers movies the user
    did not rate in train, so those are the candidates.
    """
    relevant = test[test["rating"] >= metrics.RELEVANT_THRESHOLD].groupby("user_id")["item_id"].apply(set)
    precisions, recalls = [], []
    for user, liked in relevant.items():  # users without a relevant movie never appear here
        ranked = [movie for movie, _ in model.recommend(int(user), k)]
        precisions.append(metrics.precision_at_k(ranked, liked, k))
        recalls.append(metrics.recall_at_k(ranked, liked, k))
    return float(np.mean(precisions)), float(np.mean(recalls))


def rating_errors(model, test) -> tuple[float, float]:
    """RMSE and MAE on all test ratings, or NaN for models that do not predict ratings."""
    if hasattr(model, "predict_many"):  # fast path: one call for all pairs
        predicted = model.predict_many(test["user_id"].to_numpy(), test["item_id"].to_numpy())
    elif hasattr(model, "predict"):
        predicted = np.array([model.predict(int(u), int(i)) for u, i in zip(test["user_id"], test["item_id"])])
    else:
        return float("nan"), float("nan")
    actual = test["rating"].to_numpy()
    return metrics.rmse(actual, predicted), metrics.mae(actual, predicted)


def evaluate_fold(fold: int, k: int = TOP_K) -> pd.DataFrame:
    """Fit every model on ``u{fold}.base`` and score it on ``u{fold}.test``.

    Ranking metrics use only movies not rated in train; relevant = test
    rating >= 4. RMSE/MAE are computed on all test ratings (not applicable to
    models that do not predict ratings, e.g. Naive Bayes -> NaN). A model whose
    ``fit`` is not implemented yet is skipped with a message.

    Returns:
        DataFrame with one row per model and columns
        ``model, precision@k, recall@k, rmse, mae``.
    """
    train, test = load_ratings(fold)
    train_matrix = build_matrix(train)
    item_features, feature_names = item_feature_matrix(load_movies())
    rows = []
    for name, (model, fit) in build_models(item_features, feature_names).items():
        try:
            fit(train_matrix)
        except NotImplementedError:  # a teammate's model that is still a stub
            print(f"  skipping {name}: not implemented yet")
            continue
        precision, recall = ranking_scores(model, train_matrix, test, k)
        rmse, mae = rating_errors(model, test)
        rows.append({"model": name, f"precision@{k}": precision, f"recall@{k}": recall, "rmse": rmse, "mae": mae})
    return pd.DataFrame(rows)


def summarise(folds: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean and standard deviation over folds, one row per model."""
    grouped = pd.concat(folds).groupby("model", sort=False)
    mean, std = grouped.mean(), grouped.std(ddof=0)
    table = mean.join(std, lsuffix="_mean", rsuffix="_std")
    # Put each metric's mean next to its std: precision_mean, precision_std, recall_mean, ...
    ordered = [f"{metric}{suffix}" for metric in mean.columns for suffix in ("_mean", "_std")]
    return table[ordered].reset_index()


def format_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Human-readable table with ``mean ± std`` strings."""
    metrics_names = [c[: -len("_mean")] for c in summary.columns if c.endswith("_mean")]
    shown = pd.DataFrame({"model": summary["model"]})
    for metric in metrics_names:
        shown[metric] = [
            "n/a" if np.isnan(m) else f"{m:.3f} ± {s:.3f}"
            for m, s in zip(summary[f"{metric}_mean"], summary[f"{metric}_std"])
        ]
    return shown


def run_all(k: int = TOP_K, out_path: Path = RESULTS_PATH) -> pd.DataFrame:
    """Evaluate all five folds, average per model, and write the CSV.

    Returns:
        The averaged results table (mean and std per metric; also saved to ``out_path``).
    """
    folds = []
    for fold in range(1, N_FOLDS + 1):
        print(f"Fold {fold}/{N_FOLDS} ...")
        folds.append(evaluate_fold(fold, k))
    summary = summarise(folds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(format_table(summary).to_string(index=False))
    return summary
