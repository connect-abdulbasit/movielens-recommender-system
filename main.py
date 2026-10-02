"""Command-line interface for the movie recommender project.

Usage:
    python main.py recommend --user 196 --method cf
    python main.py evaluate
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with ``recommend`` and ``evaluate`` commands."""
    parser = argparse.ArgumentParser(description="MovieLens recommender: CF vs Naive Bayes")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("recommend", help="Top-N recommendations with explanations")
    rec.add_argument("--user", type=int, required=True, help="1-based MovieLens user id")
    rec.add_argument("--method", choices=["cf", "nb"], required=True)
    rec.add_argument("-n", type=int, default=10, help="number of recommendations")
    rec.add_argument("--fold", type=int, default=1, choices=range(1, 6), help="training split")

    sub.add_parser("evaluate", help="Evaluate all models on u1..u5, write results/results.csv")
    return parser


def cmd_recommend(user: int, method: str, n: int, fold: int) -> None:
    """Fit the chosen model on the fold's training data and print explained recommendations."""
    raise NotImplementedError("TODO")


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
