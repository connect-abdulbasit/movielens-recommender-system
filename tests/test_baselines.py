import numpy as np
import pytest

from src.baselines import PopularityRecommender, RandomRecommender

MATRIX = np.array([
    [5, 0, 3, 0, 0],
    [4, 2, 0, 0, 0],
    [0, 4, 0, 1, 0],
], dtype=float)

def test_random_excludes_seen_and_is_reproducible():
    model = RandomRecommender(seed=3).fit(MATRIX)
    recs = model.recommend(1, 10)
    assert sorted(movie for movie, _ in recs) == [2, 4, 5]
    assert recs == RandomRecommender(seed=3).fit(MATRIX).recommend(1, 10)

def test_random_predicts_the_global_mean():
    model = RandomRecommender().fit(MATRIX)
    assert model.predict(2, 3) == pytest.approx(19 / 6)
    assert all(score == pytest.approx(19 / 6) for _, score in model.recommend(3, 2))

def test_popularity_orders_by_count_then_mean_then_id():
    model = PopularityRecommender().fit(MATRIX)
    assert [movie for movie, _ in model.recommend(3, 10)] == [1, 3]
    assert [movie for movie, _ in model.recommend(1, 10)] == [2, 4]
    assert model.recommend(1, 1) == [(2, 2.0)]

def test_popularity_predicts_movie_mean_with_global_fallback():
    model = PopularityRecommender().fit(MATRIX)
    assert model.predict(1, 1) == pytest.approx(4.5)
    assert model.predict(1, 5) == pytest.approx(19 / 6)

def test_n_zero_returns_nothing():
    assert RandomRecommender().fit(MATRIX).recommend(1, 0) == []
    assert PopularityRecommender().fit(MATRIX).recommend(1, 0) == []
