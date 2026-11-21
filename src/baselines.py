"""Reproducible random and popularity baselines using training ratings only."""
import numpy as np


class RandomRecommender:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, train_matrix):
        self.seen = train_matrix != 0
        self.mean = float(train_matrix[self.seen].mean())
        return self

    def predict(self, user, item):
        return self.mean

    def recommend(self, user, n=10):
        unseen = np.flatnonzero(~self.seen[user - 1])
        rng = np.random.default_rng([self.seed, user])
        ranked = rng.permutation(unseen)[:max(n, 0)]
        return [(int(i + 1), self.mean) for i in ranked]


class PopularityRecommender:
    def __init__(self, min_ratings=1):
        self.min_ratings = min_ratings

    def fit(self, train_matrix):
        self.seen = train_matrix != 0
        counts = self.seen.sum(axis=0)
        global_mean = train_matrix[self.seen].mean()
        self.means = np.divide(train_matrix.sum(axis=0), counts,
                               out=np.full(len(counts), global_mean), where=counts > 0)
        items = np.flatnonzero(counts >= self.min_ratings)
        self.order = items[np.lexsort((items, -self.means[items], -counts[items]))]
        self.counts = counts
        return self

    def predict(self, user, item):
        return float(self.means[item - 1])

    def recommend(self, user, n=10):
        unseen = self.order[~self.seen[user - 1, self.order]]
        return [(int(i + 1), float(self.counts[i])) for i in unseen[:max(n, 0)]]
