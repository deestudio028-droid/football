"""Loads fixtures from matches.db and builds the per-team, per-league
chronological accumulators that every other module in this package
reads from. This is where the leakage guarantee actually lives:
`FeatureContext.process_next` only ever exposes state accumulated from
fixtures processed *before* the current one, and only appends the
current fixture's own result to that state *after* its features have
been computed and returned. A fixture can structurally never see
itself or anything chronologically after it.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import PLAYED_STATUSES


def load_fixtures_chronological(db_path: Path) -> list[dict[str, Any]]:
    """Load every fixture row from matches.db, sorted by (unix, fixture_id)
    for a deterministic processing order. Includes all statuses (a
    target row is generated for every fixture, per spec §3) -- the
    PLAYED_STATUSES filter is applied later, only when deciding whether
    a fixture is eligible to be appended to a team's *history*.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fixtures ORDER BY unix ASC, fixture_id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def team_match_record(fixture: dict[str, Any], side: str) -> dict[str, Any]:
    """Build one team-perspective history entry from a fixture row.
    `side` is "home" or "away" -- whichever side the team played on in
    this fixture.
    """
    opp_side = "away" if side == "home" else "home"

    goals_for = fixture.get(f"{side}_goals")
    goals_against = fixture.get(f"{opp_side}_goals")
    result = None
    if goals_for is not None and goals_against is not None:
        if goals_for > goals_against:
            result = "W"
        elif goals_for < goals_against:
            result = "L"
        else:
            result = "D"

    return {
        "fixture_id": fixture["fixture_id"],
        "unix": fixture["unix"],
        "competition_id": fixture["competition_id"],
        "season_id": fixture["season_id"],
        "is_home": side == "home",
        "goals_for": goals_for,
        "goals_against": goals_against,
        "result": result,
        "xg_for": fixture.get(f"stat_{side}_xg"),
        "xg_against": fixture.get(f"stat_{opp_side}_xg"),
        "shots_for": fixture.get(f"stat_{side}_shots"),
        "shots_against": fixture.get(f"stat_{opp_side}_shots"),
        "shots_on_for": fixture.get(f"stat_{side}_shots_on"),
        "shots_on_against": fixture.get(f"stat_{opp_side}_shots_on"),
        "attacks_for": fixture.get(f"stat_{side}_attacks"),
        "dang_attacks_for": fixture.get(f"stat_{side}_dang_attacks"),
        "pressure_for": fixture.get(f"stat_{side}_pressure"),
        "fouls_for": fixture.get(f"stat_{side}_fouls"),
        "yellow_cards_for": fixture.get(f"stat_{side}_yellow_cards"),
        # --- v1.1 successor keys (D-16/D-18) ------------------------------
        # Own-side only, same shape and semantics as fouls_for /
        # yellow_cards_for above. Additive: no existing key's meaning
        # changes, and nothing enumerates this dict to emit features --
        # every feature reads a specific key via missingness.EXTRACTORS --
        # so these are inert for V1 output. Read by the v1.1 registry only.
        "possession_for": fixture.get(f"stat_{side}_possession"),
        "corners_for": fixture.get(f"stat_{side}_corners"),
        "red_cards_for": fixture.get(f"stat_{side}_red_cards"),
    }


@dataclass
class LeagueSeasonAccumulator:
    """Running totals for a single (competition_id, season_id), updated
    incrementally in chronological order. `league_mean_gf` etc. reflect
    only matches processed so far -- i.e. strictly before whatever
    fixture is currently being featurized.
    """
    matches: int = 0
    goals_for_sum: float = 0.0
    goals_against_sum: float = 0.0  # kept for symmetry; equals goals_for_sum league-wide
    home_goal_diff_sum: float = 0.0  # sum of (home_goals - away_goals) across all matches so far

    def league_mean_goals_per_team_match(self) -> float | None:
        # Each match contributes two "team match" observations (home + away).
        if self.matches == 0:
            return None
        return self.goals_for_sum / (self.matches * 2)

    def home_advantage(self) -> float | None:
        if self.matches == 0:
            return None
        return self.home_goal_diff_sum / self.matches

    def record(self, home_goals: int | None, away_goals: int | None) -> None:
        if home_goals is None or away_goals is None:
            return
        self.matches += 1
        self.goals_for_sum += home_goals + away_goals
        self.home_goal_diff_sum += (home_goals - away_goals)


class FeatureContext:
    """Owns all mutable, incrementally-updated state: per-team match
    history lists and per-(competition,season) league accumulators.
    Fixtures must be processed in chronological order via
    `history_before(team_id)` / `league_accumulator(comp_id, season_id)`
    (read) followed by `record(fixture)` (write, always last).
    """

    def __init__(self) -> None:
        self._team_history: dict[int, list[dict[str, Any]]] = {}
        self._league_season: dict[tuple[int, int], LeagueSeasonAccumulator] = {}

    def history_before(self, team_id: int) -> list[dict[str, Any]]:
        """All of this team's past played matches, oldest first. Since
        fixtures are processed strictly in chronological order and a
        fixture's own result is only appended *after* its features are
        computed (see `record`), this list can never contain the
        current fixture or anything after it.
        """
        return self._team_history.get(team_id, [])

    def league_accumulator(self, competition_id: int, season_id: int) -> LeagueSeasonAccumulator:
        key = (competition_id, season_id)
        if key not in self._league_season:
            self._league_season[key] = LeagueSeasonAccumulator()
        return self._league_season[key]

    def record(self, fixture: dict[str, Any]) -> None:
        """Append this fixture's result to team histories and the
        league accumulator. Must be called AFTER computing features for
        this fixture, never before -- this ordering is the entire
        leakage guarantee.
        """
        if fixture.get("status") not in PLAYED_STATUSES:
            return  # ABANDONED etc. never becomes part of anyone's history

        home_id, away_id = fixture.get("home_id"), fixture.get("away_id")
        if home_id is not None:
            self._team_history.setdefault(home_id, []).append(team_match_record(fixture, "home"))
        if away_id is not None:
            self._team_history.setdefault(away_id, []).append(team_match_record(fixture, "away"))

        acc = self.league_accumulator(fixture["competition_id"], fixture["season_id"])
        acc.record(fixture.get("home_goals"), fixture.get("away_goals"))
