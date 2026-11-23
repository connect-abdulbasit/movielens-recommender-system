"""Tabulate a filled human-evaluation ratings file through the answer key.

Usage (from the project root):
    python human_eval/tabulate.py                       # human_eval/ratings.csv + human_eval/answer_key.csv
    python human_eval/tabulate.py my_ratings.csv answer_key.csv

The ratings file has the columns of ratings_template.csv, one block of rows per
rater. ``best_list``, ``clarity_X`` and ``clarity_Y`` are per-user answers: fill
them on the List A row of each user (blank or identical on the B and C rows).
Refuses to run on the empty template.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COLUMNS = ["rater", "user_id", "list_label", "fit_score", "best_list", "clarity_X", "clarity_Y"]
ANSWERS = ["fit_score", "best_list", "clarity_X", "clarity_Y"]


def load(path: Path) -> pd.DataFrame:
    """Read a CSV as stripped strings; empty cells become ''."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.apply(lambda column: column.str.strip())


def validate(ratings: pd.DataFrame) -> None:
    """Raise ValueError if the file is the empty template or contains invalid answers."""
    missing = [c for c in COLUMNS if c not in ratings.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if (ratings[ANSWERS] == "").all().all():
        raise ValueError("no answers are filled in: this is the empty ratings_template.csv. "
                         "Copy it to ratings.csv per rater, fill in the scores, then run again.")
    answered = (ratings[ANSWERS] != "").any(axis=1)
    if (ratings.loc[answered, "rater"] == "").any():
        raise ValueError("every row with answers needs a rater name or code")
    for column in ("fit_score", "clarity_X", "clarity_Y"):
        bad = ~ratings[column].isin(["", "1", "2", "3", "4", "5"])
        if bad.any():
            raise ValueError(f"{column} must be 1-5 or empty; got {sorted(ratings.loc[bad, column].unique())}")
    bad = ~ratings["best_list"].str.upper().isin(["", "A", "B", "C"])
    if bad.any():
        raise ValueError(f"best_list must be A, B or C; got {sorted(ratings.loc[bad, 'best_list'].unique())}")


def per_user_answer(ratings: pd.DataFrame, column: str) -> pd.Series:
    """One value per (rater, user_id) for a per-user question; error if a rater gave two."""
    values = ratings[ratings[column] != ""].groupby(["rater", "user_id"])[column].unique()
    conflicts = values[values.map(len) > 1]
    if len(conflicts):
        raise ValueError(f"conflicting {column} answers for (rater, user): {list(conflicts.index)}")
    return values.map(lambda v: v[0])


def tabulate(ratings: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Map labels to models and summarise. Returns DataFrames/numbers (see :func:`report`)."""
    validate(ratings)
    key = key.astype({"user_id": str})
    lists = key[key["part"] == "list"][["user_id", "label", "model"]]
    reasons = key[key["part"] == "explanation"][["user_id", "label", "model"]]

    fits = ratings[ratings["fit_score"] != ""].merge(
        lists, left_on=["user_id", "list_label"], right_on=["user_id", "label"], how="left", validate="many_to_one")
    if fits["model"].isna().any():
        raise ValueError("some (user_id, list_label) pairs are not in the answer key")
    fit = fits.assign(score=fits["fit_score"].astype(int)).groupby("model")["score"].agg(["mean", "count"])

    best = per_user_answer(ratings, "best_list").str.upper().rename("label").reset_index()
    best = best.merge(lists, on=["user_id", "label"], how="left")
    wins = best["model"].value_counts()

    clarity_parts = []
    for label in ("X", "Y"):
        answers = per_user_answer(ratings, f"clarity_{label}").rename("score").reset_index().assign(label=label)
        clarity_parts.append(answers)
    clarity = pd.concat(clarity_parts).merge(reasons, on=["user_id", "label"], how="left")
    clarity = clarity.assign(score=clarity["score"].astype(int)).groupby("model")["score"].agg(["mean", "count"])

    raters = ratings.loc[(ratings[ANSWERS] != "").any(axis=1), "rater"].nunique()
    return {"fit": fit, "wins": wins, "clarity": clarity, "raters": raters}


def report(result: dict) -> str:
    """Human-readable summary of :func:`tabulate`."""
    lines = [f"Raters: {result['raters']}", "", "Mean fit score per model (1-5):"]
    lines += [f"  {model:<16} {row['mean']:.2f}  (n={int(row['count'])})" for model, row in result["fit"].iterrows()]
    lines += ["", "'Best list' wins per model:"]
    lines += [f"  {model:<16} {int(count)}" for model, count in result["wins"].items()]
    lines += ["", "Mean explanation clarity per model (1-5):"]
    lines += [f"  {model:<16} {row['mean']:.2f}  (n={int(row['count'])})" for model, row in result["clarity"].iterrows()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit status."""
    argv = sys.argv[1:] if argv is None else argv
    ratings_path = Path(argv[0]) if argv else HERE / "ratings.csv"
    key_path = Path(argv[1]) if len(argv) > 1 else HERE / "answer_key.csv"
    for path in (ratings_path, key_path):
        if not path.exists():
            print(f"Cannot find {path}. Fill in a copy of ratings_template.csv first (see README.md).")
            return 1
    try:
        result = tabulate(load(ratings_path), load(key_path))
    except ValueError as error:
        print(f"Refusing to tabulate {ratings_path}: {error}")
        return 1
    print(report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
