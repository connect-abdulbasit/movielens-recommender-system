from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd

from src.data_loader import build_matrix, item_feature_matrix, load_movies, load_ratings
from src.evaluate import RESULTS_PATH, TOP_K
from src.neural_recommender import NeuralRecommender

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
CURVE_PATH = RESULTS_DIR / "training_curve.csv"
DPI = 200
BAR = dict(color="0.75", edgecolor="black", linewidth=0.8)
ERR = dict(ecolor="black", capsize=3, elinewidth=0.8)
STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.spines.top": False, "axes.spines.right": False,
}
METRIC_LABEL = {f"precision@{TOP_K}": f"Precision@{TOP_K}", f"recall@{TOP_K}": f"Recall@{TOP_K}",
                "rmse": "RMSE (stars)", "mae": "MAE (stars)"}

def _bar_panel(ax, labels: list[str], means, stds, metric: str, decimals: int = 3) -> None:
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, error_kw=ERR, **BAR)
    for xi, mean, std in zip(x, means, stds):
        ax.text(xi, mean + std, f"{mean:.{decimals}f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_title(METRIC_LABEL[metric])
    ax.set_ylim(0, max(np.asarray(means) + np.asarray(stds)) * 1.18)

def plot_metric_bars(summary: pd.DataFrame, metrics: list[str], title: str, path: Path) -> Path:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
        for ax, metric in zip(np.atleast_1d(axes), metrics):
            rows = summary[summary[f"{metric}_mean"].notna()]
            _bar_panel(ax, list(rows["model"]), rows[f"{metric}_mean"], rows[f"{metric}_std"], metric)
            ax.set_xlabel("Model")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
    return path

def write_training_curve(path: Path = CURVE_PATH, fold: int = 1) -> Path:
    train, _ = load_ratings(fold)
    features, names = item_feature_matrix(load_movies())
    model = NeuralRecommender().fit(build_matrix(train), features, names)
    history = model.history
    frame = pd.DataFrame({"epoch": np.arange(1, len(history["train_loss"]) + 1),
                          "train_mse": history["train_loss"], "validation_mse": history["val_loss"]})
    frame.to_csv(path, index=False)
    return path

def plot_training_curve(curve: pd.DataFrame, path: Path, fold: int = 1) -> Path:
    best = int(curve.loc[curve["validation_mse"].idxmin(), "epoch"])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(curve["epoch"], curve["train_mse"], color="black", linewidth=1.4)
        ax.plot(curve["epoch"], curve["validation_mse"], color="0.55", linewidth=1.4)
        last = curve.iloc[-1]
        ax.text(last["epoch"] + 0.3, last["train_mse"], "training (95%)", va="center", fontsize=8)
        ax.text(last["epoch"] + 0.3, last["validation_mse"], "validation (5%)", va="center", fontsize=8, color="0.35")
        ax.axvline(best, color="black", linestyle=":", linewidth=0.9)
        ax.text(best, ax.get_ylim()[1], f" best epoch {best}\n (weights restored)", va="top", fontsize=8)
        ax.set_xlim(0.5, last["epoch"] + 4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE of the rating offset (stars²)")
        ax.set_title(f"Neural network training on fold {fold} (early stopping, patience 5)")
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
    return path

def make_all(results_dir: Path = RESULTS_DIR, fig_dir: Path = FIG_DIR) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    summary = pd.read_csv(results_dir / RESULTS_PATH.name)
    ranking = [f"precision@{TOP_K}", f"recall@{TOP_K}"]
    written.append(plot_metric_bars(summary, ranking, "Top-10 ranking quality (mean ± SD over 5 folds)",
                                    fig_dir / "fig1_ranking.png"))
    written.append(plot_metric_bars(summary, ["rmse", "mae"],
                                    "Rating error of the models that predict stars (mean ± SD over 5 folds)",
                                    fig_dir / "fig2_rating_error.png"))
    curve_path = write_training_curve(results_dir / CURVE_PATH.name)
    written.append(curve_path)
    written.append(plot_training_curve(pd.read_csv(curve_path), fig_dir / "fig4_training_curve.png"))
    for path in written:
        print(f"wrote {path}")
    return written
