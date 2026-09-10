"""Form features (spec §E): points, win/draw/loss rate, goal difference,
and coverage-gated xG difference, over the last5/last10 windows only.
"""
from __future__ import annotations

from typing import Any

from .config import FORM_WINDOWS
from .rolling import gated_mean


def _points(m: dict[str, Any]) -> float | None:
    if m["result"] == "W":
        return 3.0
    if m["result"] == "D":
        return 1.0
    if m["result"] == "L":
        return 0.0
    return None


def _is_win(m):
    return None if m["result"] is None else (1.0 if m["result"] == "W" else 0.0)


def _is_draw(m):
    return None if m["result"] is None else (1.0 if m["result"] == "D" else 0.0)


def _is_loss(m):
    return None if m["result"] is None else (1.0 if m["result"] == "L" else 0.0)


def _goal_diff(m: dict[str, Any]) -> float | None:
    if m["goals_for"] is None or m["goals_against"] is None:
        return None
    return m["goals_for"] - m["goals_against"]


def _xg_diff(m: dict[str, Any]) -> float | None:
    if m["xg_for"] is None or m["xg_against"] is None:
        return None
    return m["xg_for"] - m["xg_against"]


def form_features(history: list[dict[str, Any]], target_season_id: int, prefix: str) -> dict[str, Any]:
    """`prefix` is "home" or "away" -- whose form this is, relative to
    the target fixture (the team may itself have played home or away in
    any of its *past* matches; that's tracked per-match via `is_home`
    and doesn't affect which of these columns the result lands in).
    """
    out: dict[str, Any] = {}
    for window in FORM_WINDOWS:
        points, n, cov = gated_mean(history, window, target_season_id, _points)
        out[f"{prefix}_points_{window}"] = points
        out[f"{prefix}_points_{window}_n"] = n
        out[f"{prefix}_points_{window}_coverage_n"] = cov

        win_rate, _, _ = gated_mean(history, window, target_season_id, _is_win)
        draw_rate, _, _ = gated_mean(history, window, target_season_id, _is_draw)
        loss_rate, _, _ = gated_mean(history, window, target_season_id, _is_loss)
        out[f"{prefix}_win_rate_{window}"] = win_rate
        out[f"{prefix}_draw_rate_{window}"] = draw_rate
        out[f"{prefix}_loss_rate_{window}"] = loss_rate

        goal_diff, _, _ = gated_mean(history, window, target_season_id, _goal_diff)
        out[f"{prefix}_goal_diff_{window}"] = goal_diff

        xg_diff, xg_n, xg_cov = gated_mean(history, window, target_season_id, _xg_diff)
        out[f"{prefix}_xg_diff_form_{window}"] = xg_diff
        out[f"{prefix}_xg_diff_form_{window}_n"] = xg_n
        out[f"{prefix}_xg_diff_form_{window}_coverage_n"] = xg_cov
    return out
