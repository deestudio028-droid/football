"""SQLite-backed normalized fixture storage.

One row per fixture, keyed by OddAlerts' own fixture `id`
(`fixture_id` here) via a PRIMARY KEY constraint -- this is the
deterministic dedup mechanism requirement #11 asks for: re-ingesting
the same fixture (e.g. because a date window overlaps a previous run,
or because a resume re-fetches an already-saved page) is a no-op
UPSERT, never a duplicate row.

This module only stores what `normalize.py` produces. No derived/
feature columns exist here by design -- that is explicitly out of
scope for this phase.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .normalize import FIXTURE_FIELDS, STAT_FIELDS

_NON_ID_FIXTURE_FIELDS = [f for f in FIXTURE_FIELDS if f != "id"]
_STAT_COLUMNS = [f"stat_{f}" for f in STAT_FIELDS]

# Explicit column type mapping, keyed to the exact field lists in
# normalize.py (FIXTURE_FIELDS / STAT_FIELDS). This is deliberately NOT
# a blanket "cast everything that looks numeric" pass -- every entry
# below is either a field this project has directly observed as numeric
# on a real OddAlerts fixture (see data/audit/fixtures_between_epl_sample_note.json
# and the Phase 1.1 smoke-test evidence) or is a boolean-shaped field
# stored as SQLite's conventional 0/1 INTEGER. Fields not listed here
# fall through to TEXT via `_column_def`'s default -- that is correct
# for the genuinely textual/categorical fields (names, statuses, dates,
# formations like "4-3-3", the "1-0"-style ht_score string, etc.), not
# an oversight.
#
# Rationale for the type chosen per field:
#   INTEGER : whole-number counts (goals, shots, cards, corners, fouls,
#             attacks, tackles, offsides, goal kicks, throw-ins,
#             possession-as-percentage, unix timestamps, IDs) and
#             booleans (SQLite has no native BOOLEAN type; 0/1 INTEGER
#             is the standard convention, matches has_odds/is_friendly/
#             is_cup/is_complete already using this pattern).
#   REAL    : fields observed with fractional/decimal values (xG, xGOT,
#             pressure/pressure_avg, season_progress).
_COLUMN_TYPES = {
    "fixture_id": "INTEGER PRIMARY KEY",

    # -- top-level fixture fields --
    "competition_id": "INTEGER",
    "season_id": "INTEGER",
    "season_progress": "REAL",
    "home_id": "INTEGER",
    "away_id": "INTEGER",
    "home_goals": "INTEGER",
    "away_goals": "INTEGER",
    "elapsed": "INTEGER",
    "elapsed_seconds": "INTEGER",
    "time_added": "INTEGER",
    "home_position": "INTEGER",
    "away_position": "INTEGER",
    "unix": "INTEGER",
    "home_played": "INTEGER",
    "away_played": "INTEGER",
    "has_odds": "INTEGER",       # boolean, stored as 0/1
    "referee_id": "INTEGER",
    "is_friendly": "INTEGER",    # boolean, stored as 0/1
    "is_cup": "INTEGER",         # boolean, stored as 0/1

    # -- stats.* fields (flattened to stat_*) --
    "stat_home_possession": "INTEGER",   # percentage, observed as whole numbers (e.g. 46, 54)
    "stat_away_possession": "INTEGER",
    "stat_home_pressure": "REAL",
    "stat_home_pressure_avg": "REAL",
    "stat_away_pressure": "REAL",
    "stat_away_pressure_avg": "REAL",
    "stat_cards": "INTEGER",
    "stat_home_yellow_cards": "INTEGER",
    "stat_away_yellow_cards": "INTEGER",
    "stat_home_red_cards": "INTEGER",
    "stat_away_red_cards": "INTEGER",
    "stat_corners": "INTEGER",
    "stat_home_corners": "INTEGER",
    "stat_away_corners": "INTEGER",
    "stat_home_fouls": "INTEGER",
    "stat_away_fouls": "INTEGER",
    "stat_shots": "INTEGER",
    "stat_home_shots": "INTEGER",
    "stat_away_shots": "INTEGER",
    "stat_shots_on": "INTEGER",
    "stat_home_shots_on": "INTEGER",
    "stat_away_shots_on": "INTEGER",
    "stat_attacks": "INTEGER",
    "stat_home_attacks": "INTEGER",
    "stat_away_attacks": "INTEGER",
    "stat_dang_attacks": "INTEGER",
    "stat_home_dang_attacks": "INTEGER",
    "stat_away_dang_attacks": "INTEGER",
    "stat_offsides": "INTEGER",
    "stat_home_offsides": "INTEGER",
    "stat_away_offsides": "INTEGER",
    "stat_tackles": "INTEGER",
    "stat_home_tackles": "INTEGER",
    "stat_away_tackles": "INTEGER",
    "stat_goal_kicks": "INTEGER",
    "stat_home_goal_kicks": "INTEGER",
    "stat_away_goal_kicks": "INTEGER",
    "stat_throw_ins": "INTEGER",
    "stat_home_throw_ins": "INTEGER",
    "stat_away_throw_ins": "INTEGER",
    "stat_home_xg": "REAL",
    "stat_away_xg": "REAL",
    "stat_home_xgot": "REAL",
    "stat_away_xgot": "REAL",

    # -- pipeline bookkeeping columns (not from the API) --
    "missing_field_count": "INTEGER",
    "is_complete": "INTEGER",    # boolean, stored as 0/1
}

ALL_COLUMNS = ["fixture_id"] + _NON_ID_FIXTURE_FIELDS + _STAT_COLUMNS + [
    "missing_fields", "missing_field_count", "is_complete", "ingested_at", "updated_at",
]

# Every column not explicitly typed above is intentionally left as TEXT
# (the SQLite default via `_column_def`): home_name, away_name,
# competition_country/name/type/predictability, season, winning_team,
# status, ht_score, home_formation, away_formation, venue, date,
# ko_human, missing_fields (JSON-encoded string), ingested_at, updated_at.
_UNTYPED_COLUMNS_ARE_INTENTIONALLY_TEXT = [
    c for c in ALL_COLUMNS if c not in _COLUMN_TYPES
]


def _column_def(name: str) -> str:
    return f'"{name}" {_COLUMN_TYPES.get(name, "TEXT")}'


CREATE_FIXTURES_TABLE = f"""
CREATE TABLE IF NOT EXISTS fixtures (
    {", ".join(_column_def(c) for c in ALL_COLUMNS)}
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_fixtures_competition_season ON fixtures(competition_id, season_id);",
    "CREATE INDEX IF NOT EXISTS idx_fixtures_unix ON fixtures(unix);",
    "CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures(status);",
]

UPSERT_SQL = f"""
INSERT INTO fixtures ({", ".join(f'"{c}"' for c in ALL_COLUMNS)})
VALUES ({", ".join("?" for _ in ALL_COLUMNS)})
ON CONFLICT(fixture_id) DO UPDATE SET
    {", ".join(f'"{c}" = excluded."{c}"' for c in ALL_COLUMNS if c != "fixture_id" and c != "ingested_at")}
;
"""


class FixtureDB:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        # WAL mode is deliberately NOT used here: it relies on shared-memory
        # (.shm) files and byte-range locking that some network/mounted
        # filesystems (including this project's sandboxed mount) do not
        # support, which surfaces as "disk I/O error" on the very first
        # write. The default rollback-journal mode is slightly slower
        # under concurrent access but works everywhere, and this pipeline
        # is single-writer by design anyway.
        self._conn.execute("PRAGMA journal_mode=DELETE;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(CREATE_FIXTURES_TABLE)
        for stmt in CREATE_INDEXES:
            self._conn.execute(stmt)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FixtureDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def upsert_fixture(self, record: dict[str, Any]) -> None:
        self.upsert_fixtures([record])

    def upsert_fixtures(self, records: Iterable[dict[str, Any]]) -> int:
        rows = []
        for record in records:
            row = []
            for col in ALL_COLUMNS:
                if col == "missing_fields":
                    row.append(json.dumps(record.get("missing_fields", [])))
                elif col == "is_complete":
                    row.append(1 if record.get("is_complete") else 0)
                elif col == "updated_at":
                    row.append(record.get("ingested_at"))
                else:
                    row.append(record.get(col))
            rows.append(tuple(row))
        cur = self._conn.executemany(UPSERT_SQL, rows)
        self._conn.commit()
        return cur.rowcount if cur.rowcount is not None else len(rows)

    def count_fixtures(self, competition_id: int | None = None, season_id: int | None = None) -> int:
        query = "SELECT COUNT(*) FROM fixtures WHERE 1=1"
        params: list[Any] = []
        if competition_id is not None:
            query += " AND competition_id = ?"
            params.append(competition_id)
        if season_id is not None:
            query += " AND season_id = ?"
            params.append(season_id)
        return self._conn.execute(query, params).fetchone()[0]

    def get_fixture(self, fixture_id: int) -> dict[str, Any] | None:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute("SELECT * FROM fixtures WHERE fixture_id = ?", (fixture_id,))
        row = cur.fetchone()
        self._conn.row_factory = None
        return dict(row) if row else None

    def fixtures_with_missing_stats(self, min_missing: int = 1) -> list[dict[str, Any]]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT fixture_id, competition_id, season_id, missing_field_count, missing_fields "
            "FROM fixtures WHERE missing_field_count >= ? ORDER BY missing_field_count DESC",
            (min_missing,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        self._conn.row_factory = None
        return rows
