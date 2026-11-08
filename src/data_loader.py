"""Data loading for MovieLens 100K.

All user and movie ids in DataFrames are the original 1-based MovieLens ids.
:func:`build_matrix` converts them to 0-based row/column indices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-100k"
N_USERS = 943
N_ITEMS = 1682
N_FOLDS = 5

# Release-era flags used as extra movie features (see item_feature_matrix).
ERA_NAMES: list[str] = [
    "era <=1959", "era 1960s", "era 1970s", "era 1980s",
    "era 1990-94", "era 1995", "era 1996", "era 1997", "era 1998",
]

GENRES: list[str] = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official train/test split ``u{fold}.base`` / ``u{fold}.test``.

    Args:
        fold: Fold number in 1..5.

    Returns:
        ``(train, test)`` DataFrames with columns
        ``user_id, item_id, rating, timestamp`` (ids are 1-based).
    """
    if fold not in range(1, N_FOLDS + 1):
        raise ValueError(f"fold must be in 1..{N_FOLDS}, got {fold}")
    columns = ["user_id", "item_id", "rating", "timestamp"]
    train = pd.read_csv(DATA_DIR / f"u{fold}.base", sep="\t", names=columns)
    test = pd.read_csv(DATA_DIR / f"u{fold}.test", sep="\t", names=columns)
    return train, test


def load_movies() -> pd.DataFrame:
    """Load movie metadata from ``u.item`` (encoding latin-1, '|' separated).

    Returns:
        DataFrame with columns ``item_id, title, release_date,
        video_release_date, imdb_url`` followed by the 19 binary genre
        columns in :data:`GENRES` order.
    """
    columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    # u.item is not UTF-8: titles contain accented characters stored as latin-1.
    return pd.read_csv(DATA_DIR / "u.item", sep="|", names=columns, encoding="latin-1")


def build_matrix(
    df: pd.DataFrame, n_users: int = N_USERS, n_items: int = N_ITEMS
) -> np.ndarray:
    """Build a dense user-item rating matrix from a ratings DataFrame.

    Args:
        df: Ratings with 1-based ``user_id`` and ``item_id`` columns.
        n_users: Number of rows.
        n_items: Number of columns.

    Returns:
        Float array of shape ``(n_users, n_items)``; entry ``[u-1, i-1]`` is
        the rating, and 0 means "not rated".
    """
    matrix = np.zeros((n_users, n_items))
    # Ids are 1-based in the files but arrays are 0-based, hence the "- 1".
    matrix[df["user_id"].to_numpy() - 1, df["item_id"].to_numpy() - 1] = df["rating"].to_numpy()
    return matrix


def era_index(year: float) -> int | None:
    """Position (0..8) of a release year in :data:`ERA_NAMES`, or None if unknown/out of range."""
    if np.isnan(year):
        return None
    year = int(year)
    if year <= 1959:
        return 0
    if year < 1990:
        return 1 + (year - 1960) // 10  # 1960s -> 1, 1970s -> 2, 1980s -> 3
    if year <= 1994:
        return 4
    if year <= 1998:
        return 5 + (year - 1995)  # 1995 -> 5 ... 1998 -> 8
    return None


def item_feature_matrix(movies: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Describe every movie with 19 genre flags followed by 9 release-era flags.

    Args:
        movies: Output of :func:`load_movies`.

    Returns:
        ``(features, names)``: a float array of shape ``(n_items, 28)`` whose row
        ``i - 1`` describes movie ``i``, and the 28 column names. A movie with an
        unknown release date gets all era flags equal to 0.
    """
    movies = movies.sort_values("item_id")  # row i-1 must be movie i
    genre_flags = movies[GENRES].to_numpy(dtype=float)
    # Dates look like "01-Jan-1995"; unparseable/missing dates become NaT -> NaN year.
    years = pd.to_datetime(movies["release_date"], format="%d-%b-%Y", errors="coerce").dt.year
    era_flags = np.zeros((len(movies), len(ERA_NAMES)))
    for row, year in enumerate(years.to_numpy(dtype=float)):
        era = era_index(year)
        if era is not None:
            era_flags[row, era] = 1.0
    return np.hstack([genre_flags, era_flags]), GENRES + ERA_NAMES
