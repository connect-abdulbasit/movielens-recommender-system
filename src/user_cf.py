"""User-based collaborative filtering (k-NN) implemented with numpy."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np


class UserBasedCF:
    """Neighbourhood recommender using user-user similarity.

    Args:
        k: Number of nearest neighbours used per prediction.
        similarity: ``"pearson"`` or ``"cosine"``.
        shrinkage: Shrinkage constant; similarity is scaled by
            ``n_common / (n_common + shrinkage)`` to damp small overlaps.
    """

    def __init__(
        self,
        k: int = 30,
        similarity: Literal["pearson", "cosine"] = "pearson",
        shrinkage: float = 0.0,
    ) -> None:
        self.k = k
        self.similarity = similarity
        self.shrinkage = shrinkage

    def fit(self, train_matrix: np.ndarray) -> "UserBasedCF":
        """Store the training matrix and compute user-user similarities.

        Args:
            train_matrix: ``(n_users, n_items)`` ratings, 0 = unrated.

        Returns:
            ``self``.
        """
        raise NotImplementedError("TODO: compute similarity matrix, user means")

    def predict(self, user: int, item: int) -> float:
        """Predict a rating (1-based ids).

        Args:
            user: 1-based user id.
            item: 1-based movie id.

        Returns:
            Predicted rating, clipped to [1, 5].
        """
        raise NotImplementedError("TODO: weighted neighbour average")

    def recommend(self, user: int, n: int = 10) -> list[tuple[int, float]]:
        """Top-n unseen movies for a user.

        Args:
            user: 1-based user id.
            n: Number of recommendations.

        Returns:
            ``[(movie_id, predicted_rating), ...]`` best first (1-based ids),
            excluding movies the user rated in the training set.
        """
        raise NotImplementedError("TODO: score all unseen items and sort")

    def explain(self, user: int, item: int) -> dict[str, Any]:
        """Return the internals behind a prediction (1-based ids).

        Returns:
            Dict with e.g. ``prediction``, ``user_mean`` and ``neighbours``:
            a list of ``(neighbour_id, similarity, neighbour_rating)`` for the
            neighbours who rated the item. Feed it to ``src.explain``.
        """
        raise NotImplementedError("TODO: expose contributing neighbours")
