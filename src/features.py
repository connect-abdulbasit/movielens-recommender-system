from __future__ import annotations

import numpy as np

from src.data_loader import GENRES

USER_LEVEL_CENTRE = 3.5
ITEM_PRIOR = 5.0

LABEL_ERA = "release era"
LABEL_USER_LEVEL = "your average rating level"
LABEL_POPULARITY = "movie popularity"
LABEL_MOVIE_AVG = "movie's average rating"

class FeatureBuilder:

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

    def _compute_statistics(self) -> None:
        rated = (self.ratings > 0).astype(float)
        self.user_sum = self.ratings.sum(axis=1)
        self.user_count = rated.sum(axis=1)
        self.item_sum = self.ratings.sum(axis=0)
        self.item_count = rated.sum(axis=0)
        self.user_genre_sum = self.ratings @ self.genre_flags
        self.user_genre_count = rated @ self.genre_flags
        self.total_sum = self.ratings.sum()
        self.total_count = rated.sum()

    def _build_names(self, item_feature_names: list[str]) -> list[str]:
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
        return len(self.feature_names)

    def build(
        self, users: np.ndarray, items: np.ndarray, leave_one_out: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
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
        if not leave_one_out:
            return np.zeros(len(users))
        return self.ratings[users, items]

    def _global_mean(self, removed: np.ndarray) -> np.ndarray:
        count = self.total_count - (removed > 0)
        return (self.total_sum - removed) / count

    def _user_block(
        self, users: np.ndarray, removed: np.ndarray, global_mean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        count = self.user_count[users] - (removed > 0)
        total = self.user_sum[users] - removed
        mean = np.where(count > 0, total / np.maximum(count, 1), global_mean)
        level = mean - USER_LEVEL_CENTRE
        activity = np.log1p(count) / 5
        return np.column_stack([level, activity]), mean

    def _taste_block(
        self, users: np.ndarray, items: np.ndarray, removed: np.ndarray, user_mean: np.ndarray
    ) -> np.ndarray:
        flags = self.genre_flags[items]
        hidden = (removed > 0)[:, None] * flags
        total = self.user_genre_sum[users] - removed[:, None] * flags
        count = self.user_genre_count[users] - hidden
        mean = user_mean[:, None]
        m = self.taste_prior
        taste = ((total + m * mean) / (count + m) - mean) * flags
        average = taste.sum(axis=1) / np.maximum(flags.sum(axis=1), 1)
        return np.column_stack([taste, average])

    def _item_block(
        self, items: np.ndarray, removed: np.ndarray, global_mean: np.ndarray
    ) -> np.ndarray:
        count = self.item_count[items] - (removed > 0)
        total = self.item_sum[items] - removed
        popularity = np.log1p(count) / 6
        average = (total + ITEM_PRIOR * global_mean) / (count + ITEM_PRIOR) - global_mean
        return np.column_stack([popularity, average])

    def taste_column(self, genre: int) -> int:
        return self.item_features.shape[1] + 2 + genre

    def groups(self, item: int) -> dict[str, list[int]]:
        n_item_features = self.item_features.shape[1]
        taste_start = n_item_features + 2
        groups: dict[str, list[int]] = {}
        for genre in np.flatnonzero(self.genre_flags[item]):
            groups[self.feature_names[genre]] = [int(genre), self.taste_column(int(genre))]
        if n_item_features > self.n_genres:
            groups[LABEL_ERA] = list(range(self.n_genres, n_item_features))
        groups[LABEL_USER_LEVEL] = [n_item_features, n_item_features + 1]
        if self.use_item_stats:
            first = taste_start + self.n_genres + 1
            groups[LABEL_POPULARITY] = [first]
            groups[LABEL_MOVIE_AVG] = [first + 1]
        return groups
