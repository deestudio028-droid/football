"""Match-context features that aren't rolling stats: sample-size /
"insufficient history" signals (spec §5, the promotion/new-team
proxy), and league-level home advantage (spec §F).
"""
from __future__ import annotations

from typing import Any

from .config import MIN_HISTORY_THRESHOLD
from .history import LeagueSeasonAccumulator
from .rolling import venue_restricted_mean


def sample_size_features(home_history: list[dict[str, Any]], away_history: list[dict[str, Any]], prefix_pair=("home", "away")) -> dict[str, Any]:
    home_n = len(home_history)
    away_n = len(away_history)
    return {
        "home_matches_played_before_target": home_n,
        "away_matches_played_before_target": away_n,
        "home_insufficient_history": home_n < MIN_HISTORY_THRESHOLD,
        "away_insufficient_history": away_n < MIN_HISTORY_THRESHOLD,
    }


def home_advantage_features(
    home_history: list[dict[str, Any]],
    away_history: list[dict[str, Any]],
    target_season_id: int,
    league_acc: LeagueSeasonAccumulator,
) -> dict[str, Any]:
    home_goal_diff, home_n, home_cov = venue_restricted_mean(
        home_history, target_season_id, lambda m: (m["goals_for"] - m["goals_against"]) if None not in (m["goals_for"], m["goals_against"]) else None, home_only=True,
    )
    away_goal_diff, away_n, away_cov = venue_restricted_mean(
        away_history, target_season_id, lambda m: (m["goals_for"] - m["goals_against"]) if None not in (m["goals_for"], m["goals_against"]) else None, home_only=False,
    )
    return {
        "home_team_home_goal_diff_season": home_goal_diff,
        "home_team_home_goal_diff_season_n": home_n,
        "away_team_away_goal_diff_season": away_goal_diff,
        "away_team_away_goal_diff_season_n": away_n,
        "league_home_advantage_season": league_acc.home_advantage(),
        "league_home_advantage_season_n": league_acc.matches,
    }
