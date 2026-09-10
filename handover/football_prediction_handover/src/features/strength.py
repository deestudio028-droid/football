"""Attack/defence strength via empirical-Bayes shrinkage toward the
league's own season-to-date mean (spec §4). No iterative opponent-
adjusted rating -- deliberately deferred, see spec §4 for why.
"""
from __future__ import annotations

from typing import Any

from .config import SHRINKAGE_K
from .history import LeagueSeasonAccumulator


def _season_matches(history: list[dict[str, Any]], target_season_id: int) -> list[dict[str, Any]]:
    return [m for m in history if m["season_id"] == target_season_id]


def shrunk_rate(team_sum: float, team_matches: int, league_mean: float | None, k: float = SHRINKAGE_K) -> float | None:
    if league_mean is None:
        # No league history yet at all (first-ever match of a league+season)
        # -- nothing to shrink toward and no team data either. NULL, not 0.
        if team_matches == 0:
            return None
        league_mean = 0.0  # degrade gracefully if team has matches but league accumulator is somehow empty
    return (team_sum + k * league_mean) / (team_matches + k)


def strength_features(
    home_history: list[dict[str, Any]],
    away_history: list[dict[str, Any]],
    target_season_id: int,
    league_acc: LeagueSeasonAccumulator,
) -> dict[str, Any]:
    league_mean = league_acc.league_mean_goals_per_team_match()

    home_season = _season_matches(home_history, target_season_id)
    away_season = _season_matches(away_history, target_season_id)

    home_gf_sum = sum(m["goals_for"] for m in home_season if m["goals_for"] is not None)
    home_ga_sum = sum(m["goals_against"] for m in home_season if m["goals_against"] is not None)
    away_gf_sum = sum(m["goals_for"] for m in away_season if m["goals_for"] is not None)
    away_ga_sum = sum(m["goals_against"] for m in away_season if m["goals_against"] is not None)

    home_n = len(home_season)
    away_n = len(away_season)

    home_attack = shrunk_rate(home_gf_sum, home_n, league_mean)
    home_defence = shrunk_rate(home_ga_sum, home_n, league_mean)
    away_attack = shrunk_rate(away_gf_sum, away_n, league_mean)
    away_defence = shrunk_rate(away_ga_sum, away_n, league_mean)

    strength_diff = None
    if None not in (home_attack, home_defence, away_attack, away_defence):
        strength_diff = (home_attack - home_defence) - (away_attack - away_defence)

    return {
        "home_attack_strength_score": home_attack,
        "home_defence_strength_score": home_defence,
        "away_attack_strength_score": away_attack,
        "away_defence_strength_score": away_defence,
        "strength_diff": strength_diff,
        "league_mean_goals_per_team_match_season": league_mean,
        "shrinkage_k": SHRINKAGE_K,
    }
