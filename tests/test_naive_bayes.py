import unittest

import numpy as np

from src.naive_bayes import NaiveBayesRecommender
from src.metrics import precision_at_k, recall_at_k


class BayesTests(unittest.TestCase):
    def setUp(self):
        flags = np.zeros((4, 19))
        flags[[0, 2], 1] = 1
        self.model = NaiveBayesRecommender().fit([[5, 2, 0, 0], [0, 0, 0, 0]], flags)

    def test_hand_computed_probability(self):
        self.assertAlmostEqual(self.model.predict_proba(1, 3), 2/3)
        self.assertAlmostEqual(self.model.predict_proba(1, 4), 1/3)

    def test_seen_filter_and_cold_user(self):
        self.assertEqual([i for i, _ in self.model.recommend(1)], [3, 4])
        self.assertEqual(self.model.recommend(2, 2), [(1, .5), (2, .5)])

    def test_explanation_reconstructs_odds(self):
        for item in (3, 4):
            e = self.model.explain(1, item)
            prior = e['prior_like']
            odds = np.log(prior/(1-prior)) + e['absent_log_odds']
            odds += sum(x for _, x in e['genre_contributions'])
            self.assertAlmostEqual(1/(1+np.exp(-odds)), e['probability'])

    def test_ranking_edges(self):
        self.assertEqual(precision_at_k([1, 1], {1}, 2), .5)
        self.assertEqual(recall_at_k([1, 2], {1}, -1), 0.)
