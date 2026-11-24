import numpy as np

from src.data_loader import GENRES

class NaiveBayesRecommender:
    def __init__(self, like_threshold=4, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        self.like_threshold = like_threshold
        self.alpha = alpha

    def fit(self, train_matrix, item_genres):
        ratings = np.asarray(train_matrix)
        genres = np.asarray(item_genres, dtype=float)
        if ratings.ndim != 2 or genres.shape != (ratings.shape[1], len(GENRES)):
            raise ValueError("Supply one row of MovieLens genre flags per movie")
        if not np.isin(genres, [0, 1]).all():
            raise ValueError("Genre flags must be binary")
        self.seen = ratings != 0
        self.genres = genres
        liked = self.seen & (ratings >= self.like_threshold)
        disliked = self.seen & ~liked
        likes = liked.sum(axis=1)
        dislikes = disliked.sum(axis=1)
        self.prior = (likes + self.alpha) / (likes + dislikes + 2 * self.alpha)

        def likelihood(mask, counts):
            return (mask @ genres + self.alpha) / (counts[:, None] + 2 * self.alpha)

        pos = likelihood(liked, likes)
        neg = likelihood(disliked, dislikes)
        self.present = np.log(pos) - np.log(neg)
        self.absent = np.log1p(-pos) - np.log1p(-neg)
        base = np.log(self.prior) - np.log1p(-self.prior) + self.absent.sum(axis=1)
        odds = base[:, None] + (self.present - self.absent) @ genres.T
        self.scores = np.exp(-np.logaddexp(0, -odds))
        return self

    def _index(self, value, size):
        if not 1 <= value <= size:
            raise ValueError(f"ID must be in 1..{size}")
        return value - 1

    def predict_proba(self, user, item):
        u = self._index(user, len(self.seen))
        i = self._index(item, len(self.genres))
        return float(self.scores[u, i])

    def recommend(self, user, n=10):
        u = self._index(user, len(self.seen))
        unseen = np.flatnonzero(~self.seen[u])
        order = np.argsort(-self.scores[u, unseen], kind="stable")[:max(n, 0)]
        return [(int(i + 1), float(self.scores[u, i])) for i in unseen[order]]

    def explain(self, user, item):
        probability = self.predict_proba(user, item)
        u, i = user - 1, item - 1
        present = self.genres[i].astype(bool)
        evidence = [(GENRES[g], float(self.present[u, g])) for g in np.flatnonzero(present)]
        evidence.sort(key=lambda pair: -abs(pair[1]))
        return dict(prior_like=float(self.prior[u]), probability=probability,
                    genre_contributions=evidence,
                    absent_log_odds=float(self.absent[u, ~present].sum()))
