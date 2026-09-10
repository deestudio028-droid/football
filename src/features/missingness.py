"""Central place for (a) the per-match field extractors every rolling
feature is built from, so there's exactly one definition of "what does
'shots for' mean" instead of duplicated lambdas scattered across
modules, and (b) post-hoc coverage/missingness summarization used by
the Phase 2 feature quality audit. No leakage-relevant logic lives
here -- extractors only ever look at a single already-past match's own
fields, never at the target.
"""
from __future__ import annotations

from typing import Any, Callable

Extractor = Callable[[dict[str, Any]], "float | None"]

EXTRACTORS: dict[str, Extractor] = {
    "goals_for": lambda m: m["goals_for"],
    "goals_against": lambda m: m["goals_against"],
    "xg_for": lambda m: m["xg_for"],
    "xg_against": lambda m: m["xg_against"],
    "shots_for": lambda m: m["shots_for"],
    "shots_against": lambda m: m["shots_against"],
    "shots_on_for": lambda m: m["shots_on_for"],
    "shots_on_against": lambda m: m["shots_on_against"],
    "attacks_for": lambda m: m["attacks_for"],
    "dang_attacks_for": lambda m: m["dang_attacks_for"],
    "pressure_for": lambda m: m["pressure_for"],
    "fouls_for": lambda m: m["fouls_for"],
    "yellow_cards_for": lambda m: m["yellow_cards_for"],
    # --- v1.1 successor extractors (D-18) -----------------------------
    # Additive only. EXTRACTORS is consumed by key lookup and is never
    # iterated to decide what gets emitted (feature_builder.py uses
    # EXTRACTORS[key] at four sites), so adding entries here cannot
    # change any V1 feature. Which of these are actually emitted is
    # decided solely by the registry passed to the builder.
    "possession_for": lambda m: m["possession_for"],
    "corners_for": lambda m: m["corners_for"],
    "red_cards_for": lambda m: m["red_cards_for"],
}


def describe_missingness(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    """Null-rate summary for one column of the generated feature dataset
    (not the raw match data) -- used by the Phase 2 feature engineering
    report, not by the feature-generation pipeline itself.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0, "null": 0, "pct_null": None}
    null = sum(1 for r in rows if r.get(column) is None)
    return {"n": n, "null": null, "pct_null": round(100 * null / n, 2)}
