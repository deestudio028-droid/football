"""Pure functions that turn a team's pre-target match history into
windowed feature values. Nothing here touches the database or the
target fixture directly -- every function takes an already-filtered
"history strictly before T" list (see history.FeatureContext) as input,
so leakage-safety is a property of what's passed in, not of this module.
"""
from __future__ import annotations

from typing import Any, Callable

from .config import VENUE_MIN_COVERAGE, WINDOW_MIN_COVERAGE


def take_window(history: list[dict[str, Any]], window: str, target_season_id: int) -> list[dict[str, Any]]:
    """`history` must already be sorted oldest-first and contain only
    matches strictly before the target fixture.
    """
    if window == "season":
        return [m for m in history if m["season_id"] == target_season_id]
    if window == "last5":
        return history[-5:]
    if window == "last10":
        return history[-10:]
    raise ValueError(f"Unknown window: {window}")


def windowed_mean(
    matches: list[dict[str, Any]], extractor: Callable[[dict[str, Any]], float | None]
) -> tuple[float | None, int, int]:
    """Returns (mean, n, coverage_n). `n` is how many matches were in the
    window; `coverage_n` is how many of those had a non-None value for
    this specific field. The mean is None whenever coverage_n is 0 --
    callers apply the window's minimum-coverage threshold on top of this
    to decide whether to keep or null out the value (see feature_builder).
    """
    n = len(matches)
    values = [extractor(m) for m in matches]
    non_null = [v for v in values if v is not None]
    coverage_n = len(non_null)
    mean = (sum(non_null) / coverage_n) if coverage_n > 0 else None
    return mean, n, coverage_n


def gated_mean(
    history: list[dict[str, Any]],
    window: str,
    target_season_id: int,
    extractor: Callable[[dict[str, Any]], float | None],
) -> tuple[float | None, int, int]:
    """windowed_mean, with the window's minimum-coverage threshold applied
    (spec §1/§0). Returns (value_or_None, n, coverage_n) -- n and
    coverage_n are always returned even when the value is nulled out, so
    callers can store the "why" alongside the NULL.
    """
    matches = take_window(history, window, target_season_id)
    mean, n, coverage_n = windowed_mean(matches, extractor)
    min_required = WINDOW_MIN_COVERAGE[window]
    value = mean if coverage_n >= min_required else None
    return value, n, coverage_n


def venue_restricted_mean(
    history: list[dict[str, Any]],
    target_season_id: int,
    extractor: Callable[[dict[str, Any]], float | None],
    home_only: bool,
) -> tuple[float | None, int, int]:
    """Season-window mean restricted to the team's own home (or away)
    matches only (spec §A/§B venue-restricted rows). 2-match minimum.
    """
    season_matches = [m for m in history if m["season_id"] == target_season_id]
    venue_matches = [m for m in season_matches if m["is_home"] == home_only]
    mean, n, coverage_n = windowed_mean(venue_matches, extractor)
    value = mean if coverage_n >= VENUE_MIN_COVERAGE else None
    return value, n, coverage_n
