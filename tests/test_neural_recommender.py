"""NeuralRecommender: determinism, ranges, recommendations, explanations, and real-data quality."""
import numpy as np
import pytest

from src.data_loader import GENRES, build_matrix, item_feature_matrix, load_movies, load_ratings
from src.neural_recommender import NeuralRecommender

FAST = dict(epochs=5, batch_size=64)


def _fit(toy_data, **kwargs):
    matrix, item_features, names = toy_data
    return NeuralRecommender(**{**FAST, **kwargs}).fit(matrix, item_features, names)


def test_fit_is_deterministic_for_a_fixed_seed(toy_data):
    a, b = _fit(toy_data, seed=7), _fit(toy_data, seed=7)
    assert [a.predict(3, i) for i in range(1, 20)] == [b.predict(3, i) for i in range(1, 20)]


def test_different_seeds_differ(toy_data):
    assert _fit(toy_data, seed=1).predict(3, 5) != _fit(toy_data, seed=2).predict(3, 5)


def test_predictions_are_within_one_to_five(toy_data):
    model = _fit(toy_data)
    users, items = np.meshgrid(np.arange(1, 41), np.arange(1, 61))
    predictions = model.predict_many(users.ravel(), items.ravel())
    assert predictions.min() >= 1.0 and predictions.max() <= 5.0


def test_predict_matches_predict_many(toy_data):
    model = _fit(toy_data)
    assert model.predict(2, 9) == pytest.approx(model.predict_many([2], [9])[0])


def test_recommend_excludes_rated_is_sorted_and_one_based(toy_data):
    model = _fit(toy_data)
    matrix = toy_data[0]
    for user in (1, 17, 40):
        recs = model.recommend(user, n=10)
        ids = [movie for movie, _ in recs]
        scores = [score for _, score in recs]
        assert len(recs) == 10 and len(set(ids)) == 10
        assert all(matrix[user - 1, movie - 1] == 0 for movie in ids)  # not rated in train
        assert scores == sorted(scores, reverse=True)
        assert all(1 <= movie <= matrix.shape[1] for movie in ids)


def test_explain_prediction_equals_predict(toy_data):
    model = _fit(toy_data)
    for user, item in [(1, 1), (5, 30), (22, 59)]:
        assert model.explain(user, item)["prediction"] == pytest.approx(model.predict(user, item))


def test_explain_contributions_sorted_and_only_own_genres(toy_data):
    model = _fit(toy_data)
    item_features = toy_data[1]
    item = 12
    info = model.explain(4, item)
    deltas = [abs(delta) for _, delta in info["contributions"]]
    assert deltas == sorted(deltas, reverse=True)
    own_genres = {GENRES[j] for j in np.flatnonzero(item_features[item - 1, : len(GENRES)])}
    mentioned_genres = {label for label, _ in info["contributions"] if label in GENRES}
    assert mentioned_genres == own_genres


@pytest.fixture(scope="module")
def fold1():
    train, test = load_ratings(1)
    matrix = build_matrix(train)
    features, names = item_feature_matrix(load_movies())
    model = NeuralRecommender(epochs=15).fit(matrix, features, names)
    return matrix, test.sample(5000, random_state=0), model


def test_beats_the_user_mean_baseline_on_fold_1(fold1):
    matrix, sample, model = fold1
    actual = sample["rating"].to_numpy()
    user_means = matrix.sum(axis=1) / (matrix > 0).sum(axis=1)
    baseline = np.sqrt(np.mean((user_means[sample["user_id"].to_numpy() - 1] - actual) ** 2))
    predicted = model.predict_many(sample["user_id"].to_numpy(), sample["item_id"].to_numpy())
    assert np.sqrt(np.mean((predicted - actual) ** 2)) < baseline
