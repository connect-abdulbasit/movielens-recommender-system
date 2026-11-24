import re

import numpy as np
import pandas as pd

from src.data_loader import GENRES
from src.explain import explain_nb, explain_nn, format_recommendations
from src.neural_recommender import NeuralRecommender

def _info(contributions, tastes, prediction=4.0):
    return {"prediction": prediction, "user_mean": 3.5, "contributions": contributions, "genre_taste": tastes}

def test_genre_effect_against_taste_is_stated_separately():
    text = explain_nn(_info([("Thriller", 0.17)], {"Thriller": -0.133}))
    assert "you rate Thriller above" not in text
    assert "Thriller raises the estimate here although you rate it below your average (+0.17)" in text

def test_genre_effect_matching_taste_keeps_the_short_wording():
    text = explain_nn(_info([("Sci-Fi", 0.2), ("Comedy", -0.1)], {"Sci-Fi": 0.3, "Comedy": -0.2}))
    assert "you rate Sci-Fi above your average (+0.20)" in text
    assert "you rate Comedy below your average (-0.10)" in text

def test_genre_without_taste_makes_no_taste_claim():
    text = explain_nn(_info([("Children's", 0.45)], {"Children's": 0.0}))
    assert "Children's raises the estimate here (+0.45)" in text
    assert "you rate" not in text

def test_small_effects_are_hidden_and_non_genre_phrases():
    assert "No single factor stands out" in explain_nn(_info([("Drama", 0.04)], {"Drama": 0.5}))
    text = explain_nn(_info([("movie's average rating", 0.76), ("release era", -0.3)], {}, prediction=4.36))
    assert text == (
        "Predicted 4.4 stars\n"
        "• this movie's average rating (+0.76)\n"
        "• its release era (-0.30)"
    )

def test_real_model_never_misstates_taste(toy_data):
    matrix, features, names = toy_data
    model = NeuralRecommender(epochs=5, batch_size=64).fit(matrix, features, names)
    genre = "|".join(re.escape(g) for g in GENRES)
    claim = re.compile(rf"(?:you rate ({genre}) (above|below)|({genre}) \w+ the estimate here although you rate it (above|below))")
    checked = 0
    for user in range(1, 41):
        for movie, _ in model.recommend(user, 5):
            text = explain_nn(model.explain(user, movie))
            X, _ = model.builder.build(np.array([user - 1]), np.array([movie - 1]))
            for g1, s1, g2, s2 in claim.findall(text):
                taste = X[0, model.builder.taste_column(GENRES.index(g1 or g2))]
                assert ((s1 or s2) == "above") == (taste > 0), text
                checked += 1
    assert checked > 0

def test_explain_nb_wording():
    info = {"probability": 0.84, "prior_like": 0.39, "genre_contributions": [("Fantasy", 1.2), ("Drama", -0.4)]}
    assert explain_nb(info, None) == (
        "Chance of a like: 84%  (base rate 39%)\n"
        "• Fantasy raises the odds\n"
        "• Drama lowers the odds"
    )
    info["genre_contributions"] = [("Drama", 0.0)]
    assert "genres give no clear signal" in explain_nb(info, None)

def test_format_recommendations_shows_titles_and_reasons():
    movies = pd.DataFrame({"item_id": [1, 2], "title": ["Toy Story (1995)", "GoldenEye (1995)"]})
    text = format_recommendations([(2, 4.25), (1, 3.5)], movies, {2: "Because.\n• detail"})
    assert text == (
        " 1. GoldenEye (1995)\n"
        "    score 4.25\n"
        "    Because.\n"
        "    • detail\n"
        "\n"
        " 2. Toy Story (1995)\n"
        "    score 3.50"
    )
