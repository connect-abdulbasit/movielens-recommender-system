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
    raise NotImplementedError("TODO: read tab-separated split files")


def load_movies() -> pd.DataFrame:
    """Load movie metadata from ``u.item`` (encoding latin-1, '|' separated).

    Returns:
        DataFrame with columns ``item_id, title, release_date,
        video_release_date, imdb_url`` followed by the 19 binary genre
        columns in :data:`GENRES` order.
    """
    raise NotImplementedError("TODO: read u.item with encoding='latin-1'")


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
    raise NotImplementedError("TODO: fill matrix, converting ids to 0-based")
