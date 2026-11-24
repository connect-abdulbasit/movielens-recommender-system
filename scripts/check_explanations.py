from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import GENRES, N_USERS, build_matrix, item_feature_matrix, load_movies, load_ratings
from src.explain import explain_nn
from src.neural_recommender import NeuralRecommender

GENRE = "|".join(re.escape(g) for g in GENRES)
TASTE_CLAIM = re.compile(
    rf"(?:you rate (?P<g1>{GENRE}) (?P<s1>above|below) your average"
    rf"|(?P<g2>{GENRE}) (?:raises|lowers) the estimate here although you rate it (?P<s2>above|below) your average)"
)
EFFECT_CLAIM = re.compile(rf"(?P<g>{GENRE}) (?P<effect>raises|lowers) the estimate here[^(]*\((?P<delta>[+-][0-9.]+)\)")

def count_contradictions(model: NeuralRecommender, users, n: int = 10) -> dict[str, int]:
    builder = model.builder
    counts = {"explanations": 0, "taste_claims": 0, "taste_contradictions": 0,
              "effect_claims": 0, "effect_contradictions": 0}
    for user in users:
        for movie, _ in model.recommend(user, n):
            sentence = explain_nn(model.explain(user, movie))
            counts["explanations"] += 1
            X, _ = builder.build(np.array([user - 1]), np.array([movie - 1]))
            for match in TASTE_CLAIM.finditer(sentence):
                genre = match.group("g1") or match.group("g2")
                side = match.group("s1") or match.group("s2")
                taste = X[0, builder.taste_column(GENRES.index(genre))]
                counts["taste_claims"] += 1
                counts["taste_contradictions"] += (side == "above") != (taste > 0)
            for match in EFFECT_CLAIM.finditer(sentence):
                counts["effect_claims"] += 1
                counts["effect_contradictions"] += (match.group("effect") == "raises") != (float(match.group("delta")) > 0)
    return counts

def main() -> int:
    train, _ = load_ratings(1)
    features, names = item_feature_matrix(load_movies())
    model = NeuralRecommender().fit(build_matrix(train), features, names)
    counts = count_contradictions(model, range(1, N_USERS + 1))
    print(f"Fold 1, all {N_USERS} users, Top-10: {counts['explanations']} explanations")
    print(f"  stated genre tastes:                {counts['taste_claims']}")
    print(f"  contradicting the user's real taste: {counts['taste_contradictions']}")
    print(f"  'raises/lowers the estimate' claims: {counts['effect_claims']}")
    print(f"  contradicting the effect's sign:     {counts['effect_contradictions']}")
    return int(counts["taste_contradictions"] + counts["effect_contradictions"] > 0)

if __name__ == "__main__":
    sys.exit(main())
