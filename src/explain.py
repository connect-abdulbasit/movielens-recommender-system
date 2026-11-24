from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_loader import GENRES
from src.features import LABEL_ERA, LABEL_MOVIE_AVG, LABEL_POPULARITY, LABEL_USER_LEVEL

MIN_REASON = 0.05
TASTE_EPS = 1e-9
PHRASES = {
    LABEL_MOVIE_AVG: "this movie's average rating",
    LABEL_POPULARITY: "how many people rated this movie",
    LABEL_USER_LEVEL: "your usual rating level",
    LABEL_ERA: "its release era",
}

def explain_nn(info: dict[str, Any], max_reasons: int = 3) -> str:
    start = f"Predicted {info['prediction']:.1f} stars"
    strong = [(label, delta) for label, delta in info["contributions"] if abs(delta) >= MIN_REASON]
    strong.sort(key=lambda pair: abs(pair[1]), reverse=True)
    if not strong:
        return f"{start}\n(No single factor stands out.)"
    tastes = info.get("genre_taste", {})
    reasons = [f"{_reason_text(label, delta, tastes.get(label))} ({delta:+.2f})" for label, delta in strong[:max_reasons]]
    return start + "\n" + "\n".join(f"• {r}" for r in reasons)

def _reason_text(label: str, delta: float, taste: float | None = None) -> str:
    if label in GENRES:
        return _genre_reason(label, delta, taste)
    return PHRASES.get(label, label)

def _genre_reason(genre: str, delta: float, taste: float | None) -> str:
    effect = "raises" if delta > 0 else "lowers"
    if taste is None or abs(taste) < TASTE_EPS:
        return f"{genre} {effect} the estimate here"
    side = "above" if taste > 0 else "below"
    if (taste > 0) == (delta > 0):
        return f"you rate {genre} {side} your average"
    return f"{genre} {effect} the estimate here although you rate it {side} your average"

def explain_nb(info: dict[str, Any], movies: pd.DataFrame, max_genres: int = 3) -> str:
    summary = f"Chance of a like: {info['probability']:.0%}  (base rate {info['prior_like']:.0%})"
    reasons = []
    for genre, effect in info["genre_contributions"][:max_genres]:
        if abs(effect) > 1e-9:
            direction = "raises" if effect > 0 else "lowers"
            reasons.append(f"{genre} {direction} the odds")
    if reasons:
        return summary + "\n" + "\n".join(f"• {r}" for r in reasons)
    return summary + "\n(genres give no clear signal)"

def format_recommendations(
    recs: list[tuple[int, float]], movies: pd.DataFrame, reasons: dict[int, str] | None = None
) -> str:
    titles = movies.set_index("item_id")["title"]
    blocks = []
    for rank, (movie_id, score) in enumerate(recs, start=1):
        lines = [f"{rank:>2}. {titles.loc[movie_id]}", f"    score {score:.2f}"]
        if reasons and movie_id in reasons:
            lines.extend(f"    {line}" for line in reasons[movie_id].splitlines())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
