"""Turn model internals into plain-English reasons."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_loader import GENRES
from src.features import LABEL_ERA, LABEL_MOVIE_AVG, LABEL_POPULARITY, LABEL_USER_LEVEL

MIN_REASON = 0.05  # smallest rating change (in stars) worth mentioning
PHRASES = {
    LABEL_MOVIE_AVG: "this movie's average rating",
    LABEL_POPULARITY: "how many people rated this movie",
    LABEL_USER_LEVEL: "your usual rating level",
    LABEL_ERA: "its release era",
}


def explain_nn(info: dict[str, Any], movies: pd.DataFrame, max_reasons: int = 3) -> str:
    """Describe a neural-network prediction in one sentence.

    Args:
        info: Output of ``NeuralRecommender.explain``.
        movies: Movie metadata from ``load_movies``; unused here, kept so every
            ``explain_*`` function has the same signature.
        max_reasons: How many of the strongest reasons to mention.

    Returns:
        e.g. "Predicted 4.4 stars. Mainly because: this movie's average rating
        (+0.76), you rate Sci-Fi above your average (+0.32)."
    """
    start = f"Predicted {info['prediction']:.1f} stars."
    # Tiny effects are noise; do not present them as reasons.
    strong = [(label, delta) for label, delta in info["contributions"] if abs(delta) >= MIN_REASON]
    strong.sort(key=lambda pair: abs(pair[1]), reverse=True)
    if not strong:
        return f"{start} No single factor stands out."
    reasons = [f"{_reason_text(label, delta)} ({delta:+.2f})" for label, delta in strong[:max_reasons]]
    return f"{start} Mainly because: {', '.join(reasons)}."


def _reason_text(label: str, delta: float) -> str:
    """Plain-English phrase for one explanation group."""
    if label in GENRES:
        direction = "above" if delta > 0 else "below"
        return f"you rate {label} {direction} your average"
    return PHRASES.get(label, label)


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
    titles = movies.set_index("item_id")["title"]
    lines = []
    for rank, (movie_id, score) in enumerate(recs, start=1):
        lines.append(f"{rank:>2}. {titles.loc[movie_id]}  (score {score:.2f})")
        if reasons and movie_id in reasons:
            lines.append(f"      {reasons[movie_id]}")
    return "\n".join(lines)
