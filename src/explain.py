"""Turn model internals into plain-English reasons."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_loader import GENRES
from src.features import LABEL_ERA, LABEL_MOVIE_AVG, LABEL_POPULARITY, LABEL_USER_LEVEL

MIN_REASON = 0.05  # smallest rating change (in stars) worth mentioning
TASTE_EPS = 1e-9  # a taste this close to 0 has no direction (e.g. the user never rated the genre)
PHRASES = {
    LABEL_MOVIE_AVG: "this movie's average rating",
    LABEL_POPULARITY: "how many people rated this movie",
    LABEL_USER_LEVEL: "your usual rating level",
    LABEL_ERA: "its release era",
}


def explain_nn(info: dict[str, Any], max_reasons: int = 3) -> str:
    """Describe a neural-network prediction in one sentence.

    Every statement is true by construction: "you rate X above/below your
    average" follows the sign of the user's real taste for X, and the number in
    brackets is X's effect on this prediction (occlusion delta). When the two
    disagree, the sentence says so instead of implying a taste the user does
    not have.

    Args:
        info: Output of ``NeuralRecommender.explain`` (uses ``genre_taste`` if present).
        max_reasons: How many of the strongest reasons to mention.

    Returns:
        Real output of ``python main.py recommend --user 196 --method nn``
        (fold 1, E.T. the Extra-Terrestrial): "Predicted 4.2 stars. Mainly
        because: you rate Fantasy above your average (+0.47), Children's raises
        the estimate here (+0.45), Sci-Fi raises the estimate here although you
        rate it below your average (+0.37)."
    """
    start = f"Predicted {info['prediction']:.1f} stars."
    # Tiny effects are noise; do not present them as reasons.
    strong = [(label, delta) for label, delta in info["contributions"] if abs(delta) >= MIN_REASON]
    strong.sort(key=lambda pair: abs(pair[1]), reverse=True)
    if not strong:
        return f"{start} No single factor stands out."
    tastes = info.get("genre_taste", {})
    reasons = [f"{_reason_text(label, delta, tastes.get(label))} ({delta:+.2f})" for label, delta in strong[:max_reasons]]
    return f"{start} Mainly because: {', '.join(reasons)}."


def _reason_text(label: str, delta: float, taste: float | None = None) -> str:
    """Plain-English phrase for one explanation group."""
    if label in GENRES:
        return _genre_reason(label, delta, taste)
    return PHRASES.get(label, label)


def _genre_reason(genre: str, delta: float, taste: float | None) -> str:
    """Genre phrase: the taste direction comes from ``taste``, the effect from ``delta``."""
    effect = "raises" if delta > 0 else "lowers"
    if taste is None or abs(taste) < TASTE_EPS:
        return f"{genre} {effect} the estimate here"
    side = "above" if taste > 0 else "below"
    if (taste > 0) == (delta > 0):
        return f"you rate {genre} {side} your average"
    # The genre flag itself can push against the user's taste; say that plainly.
    return f"{genre} {effect} the estimate here although you rate it {side} your average"


def explain_nb(info: dict[str, Any], movies: pd.DataFrame, max_genres: int = 3) -> str:
    """Describe a Naive Bayes probability.

    Args:
        info: Output of ``NaiveBayesRecommender.explain``.
        movies: Movie metadata from ``load_movies`` (currently unused).
        max_genres: How many strongest genres to mention.

    Returns:
        e.g. "Estimated chance of a like: 84% (prior 39%). Among its genres,
        Fantasy raises the odds, ...".
    """
    summary = f"Estimated chance of a like: {info['probability']:.0%} (prior {info['prior_like']:.0%})."
    reasons = []
    for genre, effect in info["genre_contributions"][:max_genres]:
        if abs(effect) > 1e-9:
            direction = "raises" if effect > 0 else "lowers"
            reasons.append(f"{genre} {direction} the odds")
    if reasons:
        return summary + " Among its genres, " + ", ".join(reasons) + "."
    return summary + " Its listed genres provide no directional evidence."


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
