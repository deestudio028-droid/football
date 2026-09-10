"""Raw OddAlerts fixture dict -> normalized flat record.

This module does no feature engineering. It only:
  1. Flattens the nested `stats` block into `stat_*` columns.
  2. Makes every expected field explicit -- if the API omitted a field
     or returned null, the normalized record has an explicit `None`
     rather than a silently-absent key, and the field name is recorded
     in `missing_fields` so downstream code can tell "confirmed zero/absent"
     apart from "field was requested but the API didn't have it here."
  3. Leaves every value's type/meaning untouched (no per-90 math, no
     ratios, no derived quantities of any kind).

Fields tracked here mirror what was confirmed present on live OddAlerts
fixture objects during the audit (see
data/audit/fixtures_between_epl_sample_note.json), not a documentation
guess.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Top-level fixture fields we expect and persist explicitly.
FIXTURE_FIELDS = [
    "id", "home_name", "away_name", "competition_id", "competition_country",
    "competition_name", "competition_type", "competition_predictability",
    "season_id", "season", "season_progress", "home_id", "away_id",
    "winning_team", "status", "home_goals", "away_goals", "ht_score",
    "elapsed", "elapsed_seconds", "time_added", "home_position",
    "away_position", "unix", "home_played", "away_played",
    "home_formation", "away_formation", "venue", "has_odds",
    "referee_id", "date", "ko_human", "is_friendly", "is_cup",
]

# stats.* fields we expect and persist explicitly (flattened to stat_*).
STAT_FIELDS = [
    "home_possession", "away_possession", "home_pressure", "home_pressure_avg",
    "away_pressure", "away_pressure_avg", "cards", "home_yellow_cards",
    "away_yellow_cards", "home_red_cards", "away_red_cards", "corners",
    "home_corners", "away_corners", "home_fouls", "away_fouls", "shots",
    "home_shots", "away_shots", "shots_on", "home_shots_on", "away_shots_on",
    "attacks", "home_attacks", "away_attacks", "dang_attacks",
    "home_dang_attacks", "away_dang_attacks", "offsides", "home_offsides",
    "away_offsides", "tackles", "home_tackles", "away_tackles", "goal_kicks",
    "home_goal_kicks", "away_goal_kicks", "throw_ins", "home_throw_ins",
    "away_throw_ins", "home_xg", "away_xg", "home_xgot", "away_xgot",
]


class NormalizationError(ValueError):
    pass


def normalize_fixture(raw_fixture: dict[str, Any]) -> dict[str, Any]:
    if "id" not in raw_fixture or raw_fixture["id"] is None:
        raise NormalizationError("Fixture is missing a required 'id' field; cannot store.")

    record: dict[str, Any] = {}
    missing_fields: list[str] = []

    for field in FIXTURE_FIELDS:
        if field in raw_fixture and raw_fixture[field] is not None:
            record[field] = raw_fixture[field]
        else:
            record[field] = None
            missing_fields.append(field)

    stats = raw_fixture.get("stats")
    if stats is None:
        stats = {}
        # The whole stats block being absent (as opposed to present-but-null
        # fields) is itself worth recording explicitly.
        missing_fields.append("stats")

    for field in STAT_FIELDS:
        key = f"stat_{field}"
        if field in stats and stats[field] is not None:
            record[key] = stats[field]
        else:
            record[key] = None
            missing_fields.append(key)

    record["fixture_id"] = record.pop("id")
    record["missing_fields"] = sorted(missing_fields)
    record["missing_field_count"] = len(missing_fields)
    record["is_complete"] = len(missing_fields) == 0
    record["ingested_at"] = datetime.now(timezone.utc).isoformat()

    return record
