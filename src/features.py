"""STEP 1 of the neural model: turn a (user, movie) pair into numbers.

Everything is computed from the TRAIN matrix only. For pairs that are part of
the training set we use *leave-one-out* statistics: the rating we want to
predict is removed from the user's and the movie's statistics first, so the
network can never "see the answer" through its inputs.

Columns of the feature vector (in this order):

1. movie description: genre flags, then release-era flags (``item_features``)
2. user level: mean rating minus 3.5, and log(1 + number of ratings) / 5
3. taste per genre (0 for genres the movie does not have) and their average
4. optional movie statistics: popularity and shrunk average rating
"""
from __future__ import annotations

import numpy as np

from src.data_loader import GENRES

USER_LEVEL_CENTRE = 3.5  # a typical average rating; centres the user-level feature
ITEM_PRIOR = 5.0  # "virtual ratings" that pull a movie's average towards the global mean

# Names of the explanation groups that are not genres (see FeatureBuilder.groups).
LABEL_ERA = "release era"
LABEL_USER_LEVEL = "your average rating level"
LABEL_POPULARITY = "movie popularity"
LABEL_MOVIE_AVG = "movie's average rating"


class FeatureBuilder:
    """Builds feature vectors for (user, movie) pairs from a training matrix.

    Args:
        train_matrix: ``(n_users, n_items)`` ratings, 0 = unrated.
        item_features: ``(n_items, n_item_features)``; the first ``n_genres``
            columns must be the genre flags, the rest are release-era flags.
        feature_names: Names of the ``item_features`` columns.
        n_genres: Number of genre columns at the start of ``item_features``.
        taste_prior: ``m`` in the taste formula; how strongly a user's taste for
            a genre is pulled towards their own average rating.
        use_item_stats: Add the movie's popularity and average rating.
    """

    def __init__(
        self,
        train_matrix: np.ndarray,
        item_features: np.ndarray,
        feature_names: list[str],
        n_genres: int = len(GENRES),
        taste_prior: float = 5.0,
        use_item_stats: bool = True,
    ) -> None:
        self.ratings = train_matrix
        self.item_features = item_features
        self.n_genres = n_genres
        self.taste_prior = taste_prior
        self.use_item_stats = use_item_stats
        self.genre_flags = item_features[:, :n_genres]
        self._compute_statistics()
        self.feature_names = self._build_names(feature_names)

    # ---- statistics gathered once from the training matrix -----------------

    def _compute_statistics(self) -> None:
        """Sum and count of ratings per user, per movie, per (user, genre) and overall."""
        rated = (self.ratings > 0).astype(float)
        self.user_sum = self.ratings.sum(axis=1)
        self.user_count = rated.sum(axis=1)
        self.item_sum = self.ratings.sum(axis=0)
        self.item_count = rated.sum(axis=0)
        # (users x items) @ (items x genres) = total / number of the user's ratings per genre
        self.user_genre_sum = self.ratings @ self.genre_flags
        self.user_genre_count = rated @ self.genre_flags
        self.total_sum = self.ratings.sum()
        self.total_count = rated.sum()

    def _build_names(self, item_feature_names: list[str]) -> list[str]:
        """Names of all columns, in the order produced by :meth:`build`."""
        genres = item_feature_names[: self.n_genres]
        names = list(item_feature_names)
        names += ["user average level", "user activity"]
        names += [f"taste: {genre}" for genre in genres]
        names += ["taste: average"]
        if self.use_item_stats:
            names += ["movie popularity", "movie average rating"]
        return names

    @property
    def n_features(self) -> int:
        """Length of a feature vector."""
        return len(self.feature_names)

    # ---- building features --------------------------------------------------

    def build(
        self, users: np.ndarray, items: np.ndarray, leave_one_out: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Features for many pairs at once (0-based indices).

        Args:
            users: 0-based user indices, shape ``(n,)``.
            items: 0-based movie indices, shape ``(n,)``.
            leave_one_out: Remove each pair's own training rating from the
                statistics. Use True for training pairs, False for prediction.

        Returns:
            ``(X, user_mean)``: the ``(n, n_features)`` matrix, and the user's
            mean rating (with the same leave-one-out treatment) for each pair.
        """
        users, items = np.asarray(users), np.asarray(items)
        removed = self._ratings_to_remove(users, items, leave_one_out)
        global_mean = self._global_mean(removed)
        user_block, user_mean = self._user_block(users, removed, global_mean)
        taste_block = self._taste_block(users, items, removed, user_mean)
        blocks = [self.item_features[items], user_block, taste_block]
        if self.use_item_stats:
            blocks.append(self._item_block(items, removed, global_mean))
        return np.hstack(blocks), user_mean

    def _ratings_to_remove(
        self, users: np.ndarray, items: np.ndarray, leave_one_out: bool
    ) -> np.ndarray:
        """The rating each pair must hide from its own statistics (0 = nothing to hide)."""
        if not leave_one_out:
            return np.zeros(len(users))
        return self.ratings[users, items]  # 0 already when the pair is unrated

    def _global_mean(self, removed: np.ndarray) -> np.ndarray:
        """Mean of all training ratings, without the removed rating (one value per pair)."""
        count = self.total_count - (removed > 0)
        return (self.total_sum - removed) / count

    def _user_block(
        self, users: np.ndarray, removed: np.ndarray, global_mean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """User-level features and the user's mean rating."""
        count = self.user_count[users] - (removed > 0)
        total = self.user_sum[users] - removed
        # A user with no other rating falls back to the global mean.
        mean = np.where(count > 0, total / np.maximum(count, 1), global_mean)
        level = mean - USER_LEVEL_CENTRE
        activity = np.log1p(count) / 5  # dividing keeps the value on a small scale
        return np.column_stack([level, activity]), mean

    def _taste_block(
        self, users: np.ndarray, items: np.ndarray, removed: np.ndarray, user_mean: np.ndarray
    ) -> np.ndarray:
        """How much the user rates each of the movie's genres above their own average.

        taste = (sum + m * user_mean) / (count + m) - user_mean. The ``m`` virtual
        ratings at the user's mean make a taste based on 1-2 ratings stay near 0.
        """
        flags = self.genre_flags[items]  # 1 for the genres of this movie
        hidden = (removed > 0)[:, None] * flags  # the hidden rating counted in these genres
        total = self.user_genre_sum[users] - removed[:, None] * flags
        count = self.user_genre_count[users] - hidden
        mean = user_mean[:, None]
        m = self.taste_prior
        taste = ((total + m * mean) / (count + m) - mean) * flags  # genres not in the movie -> 0
        average = taste.sum(axis=1) / np.maximum(flags.sum(axis=1), 1)
        return np.column_stack([taste, average])

    def _item_block(
        self, items: np.ndarray, removed: np.ndarray, global_mean: np.ndarray
    ) -> np.ndarray:
        """Movie popularity and its shrunk average rating (relative to the global mean)."""
        count = self.item_count[items] - (removed > 0)
        total = self.item_sum[items] - removed
        popularity = np.log1p(count) / 6
        # Shrinking: a movie with few ratings is pulled towards the global mean.
        average = (total + ITEM_PRIOR * global_mean) / (count + ITEM_PRIOR) - global_mean
        return np.column_stack([popularity, average])

    # ---- groups used by explain() -------------------------------------------

    def taste_column(self, genre: int) -> int:
        """Column holding the user's taste for ``genre`` (0-based genre index)."""
        return self.item_features.shape[1] + 2 + genre  # after the item features and 2 user-level columns

    def groups(self, item: int) -> dict[str, list[int]]:
        """Feature columns that belong together, for one movie (0-based index).

        The explanation switches one group at a time to "neutral", so groups are
        the human-readable reasons: each genre of the movie (its flag + the
        user's taste for it), the release era, the user's average level, the
        movie's popularity and the movie's average rating.
        """
        n_item_features = self.item_features.shape[1]
        taste_start = n_item_features + 2  # after the 2 user-level columns
        groups: dict[str, list[int]] = {}
        for genre in np.flatnonzero(self.genre_flags[item]):
            groups[self.feature_names[genre]] = [int(genre), self.taste_column(int(genre))]
        if n_item_features > self.n_genres:
            groups[LABEL_ERA] = list(range(self.n_genres, n_item_features))
        groups[LABEL_USER_LEVEL] = [n_item_features, n_item_features + 1]
        if self.use_item_stats:
            first = taste_start + self.n_genres + 1  # after the genre tastes and their average
            groups[LABEL_POPULARITY] = [first]
            groups[LABEL_MOVIE_AVG] = [first + 1]
        return groups
