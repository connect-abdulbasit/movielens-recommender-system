"""Turn model internals into plain-English reasons."""
from __future__ import annotations

from typing import Any

import pandas as pd


def explain_cf(info: dict[str, Any], movies: pd.DataFrame, max_neighbours: int = 3) -> str:
    """Describe a user-CF prediction.

    Args:
        info: Output of ``UserBasedCF.explain``.
        movies: Movie metadata from ``load_movies`` (for titles).
        max_neighbours: How many top neighbours to mention.

    Returns:
        e.g. "Recommended because users similar to you (#12, #87) rated it 5, 4 ...".
    """
    raise NotImplementedError("TODO")


def explain_nb(info: dict[str, Any], movies: pd.DataFrame, max_genres: int = 3) -> str:
    """Describe a Naive Bayes probability.

    Args:
        info: Output of ``NaiveBayesRecommender.explain``.
        movies: Movie metadata from ``load_movies`` (for titles).
        max_genres: How many strongest genres to mention.

    Returns:
        e.g. "You usually like Drama and War, which this movie is ...".
    """
    raise NotImplementedError("TODO")


def format_recommendations(
    recs: list[tuple[int, float]], movies: pd.DataFrame, reasons: dict[int, str] | None = None
) -> str:
    """Format ``(movie_id, score)`` pairs as a readable numbered list with titles."""
    raise NotImplementedError("TODO")
