"""Per-user Naive Bayes recommender over movie genres.

For each user, ratings >= ``like_threshold`` are "Like", the rest "Dislike".
The model learns P(Like), P(genre | Like) and P(genre | Dislike) with Laplace
smoothing, then ranks unseen movies by P(Like | genres of the movie).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class NaiveBayesRecommender:
    """Genre-based Naive Bayes recommender, one model per user.

    Args:
        like_threshold: Minimum rating counted as a "Like".
        alpha: Laplace smoothing constant.
    """

    def __init__(self, like_threshold: int = 4, alpha: float = 1.0) -> None:
        self.like_threshold = like_threshold
        self.alpha = alpha

    def fit(self, train_matrix: np.ndarray, item_genres: np.ndarray) -> "NaiveBayesRecommender":
        """Estimate per-user priors and genre likelihoods.

        Args:
            train_matrix: ``(n_users, n_items)`` ratings, 0 = unrated.
            item_genres: ``(n_items, 19)`` binary genre matrix.

        Returns:
            ``self``.
        """
        raise NotImplementedError("TODO: count likes/dislikes per user and genre")

    def predict_proba(self, user: int, item: int) -> float:
        """P(Like | genres of item) for a user (1-based ids)."""
        raise NotImplementedError("TODO: Bayes rule with smoothed likelihoods")

    def recommend(self, user: int, n: int = 10) -> list[tuple[int, float]]:
        """Top-n unseen movies ranked by P(Like | genres).

        Returns:
            ``[(movie_id, probability), ...]`` best first (1-based ids),
            excluding movies the user rated in the training set.
        """
        raise NotImplementedError("TODO: score all unseen items and sort")

    def explain(self, user: int, item: int) -> dict[str, Any]:
        """Return the internals behind a probability (1-based ids).

        Returns:
            Dict with e.g. ``prior_like``, ``probability`` and
            ``genre_contributions``: list of ``(genre, log_odds)`` for the
            movie's genres. Feed it to ``src.explain``.
        """
        raise NotImplementedError("TODO: per-genre log-likelihood ratios")
