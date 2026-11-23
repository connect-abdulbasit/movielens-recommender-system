"""Sensitivity analyses of fixed settings on the official folds u1..u5.

Nothing here is tuned or selected on test data. Every setting below was fixed
in advance, every model is trained on ``u{fold}.base`` only and is scored with
exactly the protocol of :mod:`src.evaluate` (same candidates, same relevance
threshold, same metrics). The model defaults stay the defaults; these runs only
show how much the results move when one setting changes.

Runs are cached in ``results/experiments_by_fold.csv`` after each one, so an
interrupted run resumes where it stopped (``--fresh`` recomputes everything).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.baselines import PopularityRecommender, RandomRecommender
from src.data_loader import GENRES, N_FOLDS, build_matrix, item_feature_matrix, load_movies, load_ratings
from src.evaluate import TOP_K, ranking_scores, rating_errors
from src.naive_bayes import NaiveBayesRecommender
from src.neural_recommender import NeuralRecommender

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SUMMARY_NAME, BY_FOLD_NAME = "experiments.csv", "experiments_by_fold.csv"
METRICS = [f"precision@{TOP_K}", f"recall@{TOP_K}", "rmse", "mae"]
LABEL_COLUMNS = ["experiment", "setting", "default", "model", "config"]


@dataclass(frozen=True)
class Setting:
    """One row of an experiment: a model and the parameters that differ from its defaults."""

    experiment: str
    label: str
    model: str  # "nn", "nb", "popularity" or "random"
    params: tuple = ()  # sorted (name, value) pairs; empty = the model's defaults
    default: bool = False

    @property
    def config(self) -> str:
        """Identifies the run, e.g. ``"nn hidden=(8,)"``; equal configs are computed once."""
        return " ".join([self.model] + [f"{name}={value!r}" for name, value in self.params])


def setting(experiment: str, label: str, model: str, default: bool = False, **params) -> Setting:
    """Shorthand for :class:`Setting` with keyword parameters."""
    return Setting(experiment, label, model, tuple(sorted(params.items())), default)


SETTINGS: list[Setting] = [
    setting("NN architecture", "hidden (8,)", "nn", hidden=(8,)),
    setting("NN architecture", "hidden (16, 8)", "nn", hidden=(16, 8)),
    setting("NN architecture", "hidden (32, 16)", "nn", default=True),
    setting("NN architecture", "hidden (64, 32)", "nn", hidden=(64, 32)),
    setting("NN feature set", "without movie stats", "nn", use_item_stats=False),
    setting("NN feature set", "with movie stats", "nn", default=True),
    setting("NN taste prior", "m = 1", "nn", taste_prior=1.0),
    setting("NN taste prior", "m = 5", "nn", default=True),
    setting("NN taste prior", "m = 20", "nn", taste_prior=20.0),
    setting("NN seed", "seed 0", "nn", default=True),
    *[setting("NN seed", f"seed {seed}", "nn", seed=seed) for seed in range(1, 5)],
    setting("NB like threshold", "like >= 3", "nb", like_threshold=3),
    setting("NB like threshold", "like >= 4", "nb", default=True),
    setting("NB alpha", "alpha 0.1", "nb", alpha=0.1),
    setting("NB alpha", "alpha 1", "nb", default=True),
    setting("NB alpha", "alpha 5", "nb", alpha=5.0),
    setting("Reference", "Popularity", "popularity", default=True),
    setting("Reference", "Random", "random", default=True),
]

FoldData = tuple  # (train_matrix, test DataFrame, item_features, feature_names)


@lru_cache(maxsize=1)
def _movie_features() -> tuple[np.ndarray, list[str]]:
    """Movie features, loaded once per process."""
    return item_feature_matrix(load_movies())


def load_fold(fold: int) -> FoldData:
    """Training matrix, test ratings and movie features of an official fold."""
    train, test = load_ratings(fold)
    features, names = _movie_features()
    return build_matrix(train), test, features, names


def fit_model(spec: Setting, train_matrix: np.ndarray, item_features: np.ndarray, names: list[str]):
    """Build the model described by ``spec`` and fit it on the training matrix."""
    params = dict(spec.params)
    if spec.model == "nn":
        return NeuralRecommender(**params).fit(train_matrix, item_features, names)
    if spec.model == "nb":
        return NaiveBayesRecommender(**params).fit(train_matrix, item_features[:, : len(GENRES)])
    if spec.model == "popularity":
        return PopularityRecommender(**params).fit(train_matrix)
    if spec.model == "random":
        return RandomRecommender(**params).fit(train_matrix)
    raise ValueError(f"unknown model {spec.model!r}")


def score_setting(spec: Setting, data: FoldData, k: int = TOP_K) -> dict[str, float]:
    """Fit on the fold's training data and score with the protocol of ``src.evaluate``."""
    train_matrix, test, item_features, names = data
    model = fit_model(spec, train_matrix, item_features, names)
    precision, recall = ranking_scores(model, train_matrix, test, k)
    rmse, mae = rating_errors(model, test)
    return {f"precision@{k}": precision, f"recall@{k}": recall, "rmse": rmse, "mae": mae}


def _read_cache(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    """Metrics of runs already stored in a by-fold CSV, keyed by ``(config, fold)``."""
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    metric_columns = [c for c in frame.columns if c not in LABEL_COLUMNS + ["fold"]]
    return {(row["config"], int(row["fold"])): {m: float(row[m]) for m in metric_columns}
            for row in frame.to_dict("records")}


def by_fold_frame(settings: list[Setting], runs: dict[tuple[str, int], dict[str, float]], folds) -> pd.DataFrame:
    """One row per (setting, fold) that has been run."""
    rows = []
    for spec in settings:
        for fold in folds:
            if (spec.config, fold) in runs:
                rows.append({"experiment": spec.experiment, "setting": spec.label, "default": spec.default,
                             "model": spec.model, "config": spec.config, "fold": fold, **runs[(spec.config, fold)]})
    return pd.DataFrame(rows)


def summarise(by_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean and population standard deviation over folds, one row per setting (input order kept)."""
    grouped = by_fold.drop(columns="fold").groupby(LABEL_COLUMNS, sort=False)
    mean, std = grouped.mean(), grouped.std(ddof=0)
    table = mean.join(std, lsuffix="_mean", rsuffix="_std")
    ordered = [f"{metric}{suffix}" for metric in mean.columns for suffix in ("_mean", "_std")]
    return table[ordered].reset_index()


def format_summary(summary: pd.DataFrame) -> str:
    """Compact text table: ``mean ± std`` per metric, ``*`` marks the default setting."""
    lines = [f"{'experiment':<18} {'setting':<22} " + " ".join(f"{m:>15}" for m in METRICS)]
    for row in summary.to_dict("records"):
        cells = []
        for metric in METRICS:
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            cells.append(f"{'n/a':>15}" if np.isnan(mean) else f"{mean:>7.4f} ± {std:.4f}")
        label = ("* " if row["default"] else "  ") + row["setting"]
        lines.append(f"{row['experiment']:<18} {label:<22} " + " ".join(cells))
    return "\n".join(lines)


def run_experiments(
    settings: list[Setting] = SETTINGS,
    folds=range(1, N_FOLDS + 1),
    data_for_fold: Callable[[int], FoldData] = load_fold,
    out_dir: Path = RESULTS_DIR,
    resume: bool = True,
    k: int = TOP_K,
) -> pd.DataFrame:
    """Run every setting on every fold, write both CSVs, print a summary.

    Settings with the same config (e.g. the default network, which appears in
    several experiments) are computed once per fold and reused.

    Returns:
        The summary table (also written to ``out_dir / "experiments.csv"``).
    """
    start = time.time()
    folds = list(folds)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_fold_path = out_dir / BY_FOLD_NAME
    runs = _read_cache(by_fold_path) if resume else {}
    configs = list(dict.fromkeys(spec.config for spec in settings))
    first = {config: next(s for s in settings if s.config == config) for config in configs}
    total, step, computed = len(configs) * len(folds), 0, 0
    for fold in folds:
        data = None
        for config in configs:
            step += 1
            if (config, fold) in runs:
                print(f"[{step:>3}/{total}] fold {fold}  {config:<32} cached")
                continue
            if data is None:
                data = data_for_fold(fold)
            began = time.time()
            runs[(config, fold)] = score_setting(first[config], data, k)
            computed += 1
            print(f"[{step:>3}/{total}] fold {fold}  {config:<32} {time.time() - began:5.1f}s")
            by_fold_frame(settings, runs, folds).to_csv(by_fold_path, index=False)  # resumable
    by_fold = by_fold_frame(settings, runs, folds)
    by_fold.to_csv(by_fold_path, index=False)
    summary = summarise(by_fold)
    summary.to_csv(out_dir / SUMMARY_NAME, index=False)
    print(format_summary(summary))
    print(f"{total} runs ({computed} computed, {total - computed} cached) in {time.time() - start:.1f}s; "
          f"wrote {out_dir / SUMMARY_NAME} and {by_fold_path}")
    return summary
