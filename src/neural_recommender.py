from __future__ import annotations

from typing import Any

import numpy as np

from src.features import FeatureBuilder
from src.neural_net import MLP

VALIDATION_FRACTION = 0.05
MIN_RATING, MAX_RATING = 1.0, 5.0

class NeuralRecommender:

    def __init__(
        self,
        hidden: tuple[int, ...] = (32, 16),
        lr: float = 2e-3,
        epochs: int = 40,
        batch_size: int = 256,
        l2: float = 1e-5,
        patience: int = 5,
        taste_prior: float = 5.0,
        use_item_stats: bool = True,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.patience = patience
        self.taste_prior = taste_prior
        self.use_item_stats = use_item_stats
        self.seed = seed
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def fit(
        self, train_matrix: np.ndarray, item_features: np.ndarray, feature_names: list[str]
    ) -> "NeuralRecommender":
        self.train_matrix = train_matrix
        self.builder = FeatureBuilder(
            train_matrix, item_features, feature_names,
            taste_prior=self.taste_prior, use_item_stats=self.use_item_stats,
        )
        X, y = self._training_pairs()
        rng = np.random.default_rng(self.seed)
        (X_train, y_train), (X_val, y_val) = self._split(X, y, rng)
        self._fit_scaler(X_train)
        X_train, X_val = self._scale(X_train), self._scale(X_val)
        sizes = [self.builder.n_features, *self.hidden, 1]
        self.net = MLP(sizes, seed=self.seed)
        self._train_loop(X_train, y_train, X_val, y_val, rng)
        return self

    def _training_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        users, items = np.nonzero(self.train_matrix)
        X, user_mean = self.builder.build(users, items, leave_one_out=True)
        return X, self.train_matrix[users, items] - user_mean

    def _split(self, X, y, rng):
        order = rng.permutation(len(y))
        n_val = int(round(VALIDATION_FRACTION * len(y)))
        val, train = order[:n_val], order[n_val:]
        return (X[train], y[train]), (X[val], y[val])

    def _fit_scaler(self, X_train: np.ndarray) -> None:
        self.feature_mean = X_train.mean(axis=0)
        spread = X_train.std(axis=0)
        self.feature_std = np.where(spread < 1e-8, 1.0, spread)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean) / self.feature_std

    def _train_loop(self, X_train, y_train, X_val, y_val, rng) -> None:
        best_loss, best_weights, bad_epochs = np.inf, self.net.get_weights(), 0
        self.history = {"train_loss": [], "val_loss": []}
        for _ in range(self.epochs):
            order = rng.permutation(len(y_train))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                grads = self.net.gradients(X_train[batch], y_train[batch], self.l2)
                self.net.step(grads, self.lr)
            self.history["train_loss"].append(self.net.loss(X_train, y_train))
            val_loss = self.net.loss(X_val, y_val)
            self.history["val_loss"].append(val_loss)
            if val_loss < best_loss:
                best_loss, best_weights, bad_epochs = val_loss, self.net.get_weights(), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.net.set_weights(best_weights)

    def _predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        X, user_mean = self.builder.build(users, items, leave_one_out=False)
        offset = self.net.forward(self._scale(X))
        return np.clip(user_mean + offset, MIN_RATING, MAX_RATING)

    def predict_many(self, users, items) -> np.ndarray:
        return self._predict_pairs(np.asarray(users) - 1, np.asarray(items) - 1)

    def predict(self, user: int, item: int) -> float:
        return float(self.predict_many([user], [item])[0])

    def recommend(self, user: int, n: int = 10) -> list[tuple[int, float]]:
        candidates = np.flatnonzero(self.train_matrix[user - 1] == 0)
        scores = self._predict_pairs(np.full(len(candidates), user - 1), candidates)
        best = np.argsort(-scores, kind="stable")[:n]
        return [(int(candidates[i]) + 1, float(scores[i])) for i in best]

    def explain(self, user: int, item: int) -> dict[str, Any]:
        u, i = user - 1, item - 1
        X, user_mean = self.builder.build(np.array([u]), np.array([i]), leave_one_out=False)
        genre_taste = {
            self.builder.feature_names[g]: float(X[0, self.builder.taste_column(int(g))])
            for g in np.flatnonzero(self.builder.genre_flags[i])
        }
        x = self._scale(X)
        base = self.net.forward(x)[0]
        contributions = []
        for label, columns in self.builder.groups(i).items():
            occluded = x.copy()
            occluded[0, columns] = 0.0
            contributions.append((label, float(base - self.net.forward(occluded)[0])))
        contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
        prediction = float(np.clip(user_mean[0] + base, MIN_RATING, MAX_RATING))
        return {
            "prediction": prediction,
            "user_mean": float(user_mean[0]),
            "contributions": contributions,
            "genre_taste": genre_taste,
        }
