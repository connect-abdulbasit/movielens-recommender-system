"""Feature construction: item features, hand-computed taste, and no leakage."""
import numpy as np
import pytest

from src.data_loader import GENRES, item_feature_matrix, load_movies
from src.features import FeatureBuilder, ITEM_PRIOR


def test_item_feature_matrix_shape_and_toy_story():
    features, names = item_feature_matrix(load_movies())
    assert features.shape == (1682, 28) and len(names) == 28
    toy_story = [name for name, flag in zip(names, features[0]) if flag]
    assert toy_story == ["Animation", "Children's", "Comedy", "era 1995"]


def test_unknown_release_date_has_no_era():
    movies = load_movies()
    features, _ = item_feature_matrix(movies)
    unknown = movies.index[movies["release_date"].isna()]
    assert len(unknown) >= 1
    assert features[unknown, 19:].sum() == 0


def test_taste_matches_hand_computation():
    # One user rated three Sci-Fi movies 5, 5, 5 and one Drama movie 1 -> mean 4.0.
    # Sci-Fi taste = (15 + 5 * 4.0) / (3 + 5) - 4.0 = 0.375 (same worked example as docs).
    n_genres = len(GENRES)
    genres = np.zeros((5, n_genres))
    genres[[0, 1, 2, 4], GENRES.index("Sci-Fi")] = 1
    genres[3, GENRES.index("Drama")] = 1
    matrix = np.array([[5, 5, 5, 1, 0]], dtype=float)  # movie 4 (Sci-Fi) is unrated
    builder = FeatureBuilder(matrix, genres, GENRES, use_item_stats=False)
    X, user_mean = builder.build([0], [4])
    column = builder.feature_names.index("taste: Sci-Fi")
    assert user_mean[0] == pytest.approx(4.0)
    assert X[0, column] == pytest.approx(0.375)
    assert X[0, builder.feature_names.index("taste: Drama")] == 0.0  # not a genre of movie 4


def test_leave_one_out_equals_matrix_without_the_rating(toy_data):
    matrix, item_features, names = toy_data
    builder = FeatureBuilder(matrix, item_features, names)
    users, items = np.nonzero(matrix)
    pick = np.random.default_rng(0).choice(len(users), 25, replace=False)
    for u, i in zip(users[pick], items[pick]):
        stripped = matrix.copy()
        stripped[u, i] = 0  # the answer is simply not there
        expected, expected_mean = FeatureBuilder(stripped, item_features, names).build([u], [i])
        got, got_mean = builder.build([u], [i], leave_one_out=True)
        np.testing.assert_allclose(got, expected, atol=1e-12)
        np.testing.assert_allclose(got_mean, expected_mean, atol=1e-12)


def test_without_leave_one_out_the_rating_is_visible(toy_data):
    # Sanity check that the test above is meaningful: normal features DO contain the rating.
    matrix, item_features, names = toy_data
    builder = FeatureBuilder(matrix, item_features, names)
    u, i = map(int, np.argwhere(matrix > 0)[0])
    with_rating, _ = builder.build([u], [i], leave_one_out=False)
    without, _ = builder.build([u], [i], leave_one_out=True)
    assert not np.allclose(with_rating, without)


def test_shrunk_movie_average_is_zero_for_a_movie_nobody_rated(toy_data):
    matrix, item_features, names = toy_data
    matrix = matrix.copy()
    matrix[:, 0] = 0
    builder = FeatureBuilder(matrix, item_features, names)
    X, _ = builder.build([0], [0])
    assert X[0, -1] == pytest.approx(0.0)  # last column = shrunk average rating
    assert X[0, -2] == pytest.approx(0.0)  # log(1 + 0) popularity
    assert ITEM_PRIOR > 0
