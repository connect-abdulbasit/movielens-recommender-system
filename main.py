"""Command-line interface for the movie recommender project.

Usage:
    python main.py recommend --user 196 --method nn
    python main.py evaluate
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with ``recommend`` and ``evaluate`` commands."""
    parser = argparse.ArgumentParser(description="MovieLens recommender: Neural Network vs Naive Bayes")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("recommend", help="Top-N recommendations with explanations")
    rec.add_argument("--user", type=int, required=True, help="1-based MovieLens user id")
    rec.add_argument("--method", choices=["nn", "nb"], required=True)
    rec.add_argument("-n", type=int, default=10, help="number of recommendations")
    rec.add_argument("--fold", type=int, default=1, choices=range(1, 6), help="training split")

    sub.add_parser("evaluate", help="Evaluate all models on u1..u5, write results/results.csv")
    return parser


def cmd_recommend(user: int, method: str, n: int, fold: int) -> None:
    """Fit the chosen model on the fold's training data and print explained recommendations."""
    from src.data_loader import GENRES, N_USERS, build_matrix, item_feature_matrix, load_movies, load_ratings
    from src.explain import explain_nb, explain_nn, format_recommendations

    if not 1 <= user <= N_USERS:
        raise SystemExit(f"--user must be between 1 and {N_USERS}")
    if n < 1:
        raise SystemExit("-n must be positive")
    train, _ = load_ratings(fold)
    matrix, movies = build_matrix(train), load_movies()
    features, names = item_feature_matrix(movies)

    if method == "nn":
        from src.neural_recommender import NeuralRecommender

        print(f"Training the neural network on fold {fold} ...")
        model = NeuralRecommender().fit(matrix, features, names)
        recs = model.recommend(user, n)
        reasons = {movie: explain_nn(model.explain(user, movie), movies) for movie, _ in recs}
    else:
        from src.naive_bayes import NaiveBayesRecommender

        model = NaiveBayesRecommender().fit(matrix, features[:, : len(GENRES)])
        recs = model.recommend(user, n)
        reasons = {movie: explain_nb(model.explain(user, movie), movies) for movie, _ in recs}
    print(f"Top {n} recommendations for user {user}:")
    print(format_recommendations(recs, movies, reasons))


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the chosen command."""
    args = build_parser().parse_args(argv)
    if args.command == "recommend":
        cmd_recommend(args.user, args.method, args.n, args.fold)
    elif args.command == "evaluate":
        from src.evaluate import run_all

        run_all()


if __name__ == "__main__":
    main()
