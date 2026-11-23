"""Random and popularity baselines on a tiny hand-made matrix."""
import numpy as np
import pytest

from src.baselines import PopularityRecommender, RandomRecommender

# 3 users x 5 movies; 0 = unrated.
MATRIX = np.array([
    [5, 0, 3, 0, 0],
    [4, 2, 0, 0, 0],
    [0, 4, 0, 1, 0],
], dtype=float)


def test_random_excludes_seen_and_is_reproducible():
    model = RandomRecommender(seed=3).fit(MATRIX)
    recs = model.recommend(1, 10)
    assert sorted(movie for movie, _ in recs) == [2, 4, 5]  # all unseen movies, 1-based
    assert recs == RandomRecommender(seed=3).fit(MATRIX).recommend(1, 10)


def test_random_predicts_the_global_mean():
    model = RandomRecommender().fit(MATRIX)
    assert model.predict(2, 3) == pytest.approx(19 / 6)
    assert all(score == pytest.approx(19 / 6) for _, score in model.recommend(3, 2))


def test_popularity_orders_by_count_then_mean_then_id():
    model = PopularityRecommender().fit(MATRIX)
    # counts: movie1=2 (mean 4.5), movie2=2 (mean 3), movie3=1 (3), movie4=1 (1), movie5=0
    assert [movie for movie, _ in model.recommend(3, 10)] == [1, 3]  # unseen by user 3, most rated first
    assert [movie for movie, _ in model.recommend(1, 10)] == [2, 4]
    assert model.recommend(1, 1) == [(2, 2.0)]  # score = number of training ratings


def test_popularity_predicts_movie_mean_with_global_fallback():
    model = PopularityRecommender().fit(MATRIX)
    assert model.predict(1, 1) == pytest.approx(4.5)
    assert model.predict(1, 5) == pytest.approx(19 / 6)  # nobody rated movie 5


def test_n_zero_returns_nothing():
    assert RandomRecommender().fit(MATRIX).recommend(1, 0) == []
    assert PopularityRecommender().fit(MATRIX).recommend(1, 0) == []
