from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import N_ITEMS, N_USERS, build_matrix, item_feature_matrix, load_movies, load_ratings
from src.evaluate import build_models

EXPORT_DIR = Path(__file__).resolve().parent.parent / "results" / "recommendations"
COLUMNS = ["user_id", "rank", "movie_id", "title", "score", "test_rating"]

def ranking_table(model, users, n: int, titles: pd.Series, test: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for user in users:
        recs = model.recommend(int(user), n)
        if not recs:
            continue
        movies, scores = zip(*recs)
        parts.append(pd.DataFrame({"user_id": int(user), "rank": np.arange(1, len(recs) + 1),
                                   "movie_id": movies, "score": scores}))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    table = pd.concat(parts, ignore_index=True)
    table["title"] = table["movie_id"].map(titles)
    held_out = test.rename(columns={"item_id": "movie_id", "rating": "test_rating"})
    table = table.merge(held_out[["user_id", "movie_id", "test_rating"]], on=["user_id", "movie_id"], how="left")
    table["test_rating"] = table["test_rating"].astype("Int64")
    return table[COLUMNS]

def export_rankings(fold: int = 1, top: int | None = None, out_dir: Path = EXPORT_DIR) -> list[Path]:
    train, test = load_ratings(fold)
    matrix, movies = build_matrix(train), load_movies()
    features, names = item_feature_matrix(movies)
    titles = movies.set_index("item_id")["title"]
    n = N_ITEMS if top is None else top
    folder = out_dir / f"fold{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (model, fit) in build_models(features, names).items():
        print(f"{name}: training and ranking for {N_USERS} users ...")
        fit(matrix)
        table = ranking_table(model, range(1, N_USERS + 1), n, titles, test)
        path = folder / f"{name.lower().replace(' ', '_')}.csv"
        table.to_csv(path, index=False, float_format="%.4f")
        print(f"  wrote {len(table):,} rows to {path}")
        written.append(path)
    return written
