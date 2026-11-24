from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-100k"
N_USERS = 943
N_ITEMS = 1682
N_FOLDS = 5

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
    if fold not in range(1, N_FOLDS + 1):
        raise ValueError(f"fold must be in 1..{N_FOLDS}, got {fold}")
    columns = ["user_id", "item_id", "rating", "timestamp"]
    train = pd.read_csv(DATA_DIR / f"u{fold}.base", sep="\t", names=columns)
    test = pd.read_csv(DATA_DIR / f"u{fold}.test", sep="\t", names=columns)
    return train, test

def load_movies() -> pd.DataFrame:
    columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    return pd.read_csv(DATA_DIR / "u.item", sep="|", names=columns, encoding="latin-1")

def build_matrix(
    df: pd.DataFrame, n_users: int = N_USERS, n_items: int = N_ITEMS
) -> np.ndarray:
    matrix = np.zeros((n_users, n_items))
    matrix[df["user_id"].to_numpy() - 1, df["item_id"].to_numpy() - 1] = df["rating"].to_numpy()
    return matrix

def era_index(year: float) -> int | None:
    if np.isnan(year):
        return None
    year = int(year)
    if year <= 1959:
        return 0
    if year < 1990:
        return 1 + (year - 1960) // 10
    if year <= 1994:
        return 4
    if year <= 1998:
        return 5 + (year - 1995)
    return None

def item_feature_matrix(movies: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    movies = movies.sort_values("item_id")
    genre_flags = movies[GENRES].to_numpy(dtype=float)
    years = pd.to_datetime(movies["release_date"], format="%d-%b-%Y", errors="coerce").dt.year
    era_flags = np.zeros((len(movies), len(ERA_NAMES)))
    for row, year in enumerate(years.to_numpy(dtype=float)):
        era = era_index(year)
        if era is not None:
            era_flags[row, era] = 1.0
    return np.hstack([genre_flags, era_flags]), GENRES + ERA_NAMES
