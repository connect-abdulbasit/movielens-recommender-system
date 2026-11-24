"""Blind human-evaluation kit: do the Top-10 lists look plausible for a real user?

Raters (classmates) do not judge by their own taste. For each selected
MovieLens user they see a short profile (the user's highest- and lowest-rated
training movies) and three unlabeled Top-10 lists, one each from the neural
network, Naive Bayes and Popularity, shuffled as List A/B/C in a different
order per user. Part 2 shows one NN and one NB explanation per user, labeled
X/Y in shuffled order. Model names appear only in ``answer_key.csv``.

All models are trained on fold 1 (``u1.base``) with their default settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import PopularityRecommender
from src.data_loader import GENRES, build_matrix, item_feature_matrix, load_movies, load_ratings
from src.explain import explain_nb, explain_nn
from src.naive_bayes import NaiveBayesRecommender
from src.neural_recommender import NeuralRecommender

OUT_DIR = Path(__file__).resolve().parent.parent / "human_eval"
MODELS = ["Neural Network", "Naive Bayes", "Popularity"]
EXPLAINED = ["Neural Network", "Naive Bayes"]
LIST_LABELS, REASON_LABELS = ["A", "B", "C"], ["X", "Y"]
MIN_TRAIN_RATINGS = 40
TOP_N, PROFILE_BEST, PROFILE_WORST = 10, 5, 3
TEMPLATE_COLUMNS = ["rater", "user_id", "list_label", "fit_score", "best_list", "clarity_X", "clarity_Y"]


@dataclass
class Kit:
    """Everything needed to write the rater sheet, the template and the key."""

    users: list[int]
    titles: pd.Series  # movie id -> title
    profiles: dict[int, tuple[list, list]] = field(default_factory=dict)  # user -> (best, worst) [(movie, stars)]
    lists: dict[int, dict[str, list[int]]] = field(default_factory=dict)  # user -> model -> movie ids
    list_labels: dict[int, dict[str, str]] = field(default_factory=dict)  # user -> label -> model
    reasons: dict[int, dict[str, tuple[str, int, str]]] = field(default_factory=dict)  # user -> label -> (model, movie, text)


def pick_users(train_matrix: np.ndarray, n: int, seed: int, min_ratings: int = MIN_TRAIN_RATINGS) -> list[int]:
    """``n`` random 1-based user ids (sorted) with at least ``min_ratings`` training ratings."""
    eligible = np.flatnonzero((train_matrix > 0).sum(axis=1) >= min_ratings) + 1
    if n > len(eligible):
        raise ValueError(f"only {len(eligible)} users have >= {min_ratings} ratings")
    return sorted(int(u) for u in np.random.default_rng(seed).choice(eligible, n, replace=False))


def profile(train_matrix: np.ndarray, user: int, best: int = PROFILE_BEST, worst: int = PROFILE_WORST):
    """The user's ``best`` highest- and ``worst`` lowest-rated training movies as ``[(movie_id, stars)]``.

    Ties are broken by movie id, so the profile is deterministic.
    """
    row = train_matrix[user - 1]
    rated = np.flatnonzero(row)
    high = rated[np.lexsort((rated, -row[rated]))][:best]
    low = rated[np.lexsort((rated, row[rated]))][:worst]
    return [(int(i) + 1, int(row[i])) for i in high], [(int(i) + 1, int(row[i])) for i in low]


def build_kit(fold: int = 1, n_users: int = 10, seed: int = 0, nn_params: dict | None = None) -> Kit:
    """Train the three models on ``fold`` and assemble the blind lists and explanations."""
    train, _ = load_ratings(fold)
    matrix, movies = build_matrix(train), load_movies()
    features, names = item_feature_matrix(movies)
    models = {
        "Neural Network": NeuralRecommender(**(nn_params or {})).fit(matrix, features, names),
        "Naive Bayes": NaiveBayesRecommender().fit(matrix, features[:, : len(GENRES)]),
        "Popularity": PopularityRecommender().fit(matrix),
    }
    kit = Kit(users=pick_users(matrix, n_users, seed), titles=movies.set_index("item_id")["title"])
    for user in kit.users:
        rng = np.random.default_rng([seed, user])  # own shuffle per user
        kit.profiles[user] = profile(matrix, user)
        kit.lists[user] = {name: [movie for movie, _ in model.recommend(user, TOP_N)] for name, model in models.items()}
        kit.list_labels[user] = dict(zip(LIST_LABELS, (MODELS[i] for i in rng.permutation(len(MODELS)))))
        order = [EXPLAINED[i] for i in rng.permutation(len(EXPLAINED))]
        kit.reasons[user] = {label: _reason(models, name, kit.lists[user][name][0], user, movies)
                             for label, name in zip(REASON_LABELS, order)}
    return kit


def _reason(models: dict, name: str, movie: int, user: int, movies: pd.DataFrame) -> tuple[str, int, str]:
    """``(model, movie, sentence)``: the model's explanation of its own top recommendation."""
    if name == "Neural Network":
        return name, movie, explain_nn(models[name].explain(user, movie))
    return name, movie, explain_nb(models[name].explain(user, movie), movies)


def render_sheet(kit: Kit) -> str:
    """Printable Markdown rater sheet. Contains no model names and no scores."""
    lines = [
        "# Movie recommendation study: rater sheet",
        "",
        "Thank you for helping. You will see real MovieLens users from the 1990s. For each user you get a",
        "short profile (movies they rated highest and lowest) and three Top-10 lists made by different",
        "methods. **Judge how well each list fits THIS user, not your own taste.** There are no right answers.",
        "",
        "Scale for every score: 1 = very poor fit / not clear at all, 3 = so-so, 5 = excellent fit / very clear.",
        "",
        "Rater name or code: ____________________",
        "",
        "## Part 1: Which list fits the user?",
    ]
    for number, user in enumerate(kit.users, start=1):
        best, worst = kit.profiles[user]
        lines += ["", f"### User {number} of {len(kit.users)} (MovieLens user {user})", "",
                  "Rated highest: " + "; ".join(f"{kit.titles[m]} ({s}★)" for m, s in best),
                  "",
                  "Rated lowest: " + "; ".join(f"{kit.titles[m]} ({s}★)" for m, s in worst),
                  "", "| Rank | " + " | ".join(f"List {label}" for label in LIST_LABELS) + " |",
                  "|---:|" + "---|" * len(LIST_LABELS)]
        columns = [kit.lists[user][kit.list_labels[user][label]] for label in LIST_LABELS]
        for rank, row in enumerate(zip(*columns), start=1):
            lines.append(f"| {rank} | " + " | ".join(kit.titles[m] for m in row) + " |")
        lines += ["", "How well does this list fit this user? (1-5)   "
                  + "   ".join(f"List {label}: ____" for label in LIST_LABELS),
                  "", "Which list fits best? (A / B / C): ____"]
    lines += ["", "## Part 2: How clear is the reason?", "",
              "Each reason below was produced by a method for a movie it recommended to that user.",
              "Rate how clear and convincing the reason is (1-5)."]
    for number, user in enumerate(kit.users, start=1):
        lines += ["", f"### User {number} of {len(kit.users)} (MovieLens user {user})"]
        for label in REASON_LABELS:
            _, movie, text = kit.reasons[user][label]
            lines += ["", f"**Reason {label}**, for *{kit.titles[movie]}*:", "", text, "",
                      f"How clear and convincing is this reason? (1-5): ____"]
    return "\n".join(lines) + "\n"


def template_frame(kit: Kit) -> pd.DataFrame:
    """Empty ratings sheet: one row per (user, list label); every answer cell is empty."""
    rows = [{"rater": "", "user_id": user, "list_label": label, "fit_score": "", "best_list": "",
             "clarity_X": "", "clarity_Y": ""} for user in kit.users for label in LIST_LABELS]
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)


def key_frame(kit: Kit) -> pd.DataFrame:
    """Answer key: which model is behind every list label and every reason label."""
    rows = []
    for user in kit.users:
        rows += [{"user_id": user, "part": "list", "label": label, "model": model, "movie_id": ""}
                 for label, model in kit.list_labels[user].items()]
        rows += [{"user_id": user, "part": "explanation", "label": label, "model": model, "movie_id": movie}
                 for label, (model, movie, _) in kit.reasons[user].items()]
    return pd.DataFrame(rows)


def write_kit(out_dir: Path = OUT_DIR, n_users: int = 10, seed: int = 0, nn_params: dict | None = None) -> list[Path]:
    """Build the kit and write rater_sheet.md, ratings_template.csv and answer_key.csv."""
    kit = build_kit(n_users=n_users, seed=seed, nn_params=nn_params)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "rater_sheet.md", out_dir / "ratings_template.csv", out_dir / "answer_key.csv"]
    paths[0].write_text(render_sheet(kit), encoding="utf-8")
    template_frame(kit).to_csv(paths[1], index=False)
    key_frame(kit).to_csv(paths[2], index=False)
    print(f"Users (fold 1, >= {MIN_TRAIN_RATINGS} training ratings, seed {seed}): {kit.users}")
    for path in paths:
        print(f"wrote {path}")
    print("Keep answer_key.csv away from raters. See human_eval/README.md for how to run the study.")
    return paths
