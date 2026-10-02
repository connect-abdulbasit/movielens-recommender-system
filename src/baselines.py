"""Non-personalised baselines: random and popularity."""
from __future__ import annotations

import numpy as np


class RandomRecommender:
    """Recommends random unseen movies; predicts the global mean rating.

    Args:
        seed: Seed for reproducibility.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, train_matrix: np.ndarray) -> "RandomRecommender":
        """Store the training matrix and global mean."""
        raise NotImplementedError("TODO")

    def predict(self, user: int, item: int) -> float:
        """Global mean rating (1-based ids)."""
        raise NotImplementedError("TODO")

    def recommend(self, user: int, n: int = 10) -> list[tuple[int, float]]:
        """``n`` random unseen movies as ``(movie_id, score)`` (1-based ids)."""
        raise NotImplementedError("TODO")


class PopularityRecommender:
    """Recommends the most-rated / highest-mean unseen movies.

    Args:
        min_ratings: Minimum training ratings for a movie to be ranked.
    """

    def __init__(self, min_ratings: int = 1) -> None:
        self.min_ratings = min_ratings

    def fit(self, train_matrix: np.ndarray) -> "PopularityRecommender":
        """Compute per-movie rating counts and means."""
        raise NotImplementedError("TODO")

    def predict(self, user: int, item: int) -> float:
        """Movie mean rating, global mean if unseen (1-based ids)."""
        raise NotImplementedError("TODO")

    def recommend(self, user: int, n: int = 10) -> list[tuple[int, float]]:
        """Top-``n`` popular unseen movies as ``(movie_id, score)``."""
        raise NotImplementedError("TODO")
