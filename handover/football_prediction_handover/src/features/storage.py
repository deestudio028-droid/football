"""Feature dataset storage: a separate SQLite database
(`data/processed/features.db`), never `matches.db` -- the raw fixture
table is the immutable source of truth (see raw_store.py's docstring
for the same principle applied one layer down) and feature generation
must always be reproducible FROM it, not entangled with it.

Column typing: every feature column is declared with NO explicit SQL
type (SQLite's "BLOB affinity" when a column's declared type is
omitted). This deliberately avoids the exact bug Phase 1.1 fixed in
db.py (numeric values silently coerced to TEXT by default affinity) --
BLOB affinity stores whatever Python type was inserted (int stays
INTEGER, float stays REAL, None stays NULL, str stays TEXT) with zero
coercion, which is exactly right for a table whose ~150 columns are
overwhelmingly floats/ints/None and not worth hand-typing one by one.
`fixture_id` is the only column that needs an explicit type, because it
is the PRIMARY KEY.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

# Columns that are labels (the actual outcome), not legal model inputs.
# Kept here, in one place, so anything consuming this table can filter
# them out programmatically instead of relying on convention alone.
LABEL_COLUMNS = ["label_home_goals", "label_away_goals", "label_result"]

# Columns that are identifiers/bookkeeping, not features either.
METADATA_COLUMNS = [
    "fixture_id", "competition_id", "season_id", "unix", "home_id", "away_id",
    "feature_version", "generated_at",
]


class FeatureDB:
    def __init__(self, db_path: Path, columns: list[str]) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=DELETE;")  # see db.py -- mounted-fs WAL limitation
        self._columns = list(columns)
        if "fixture_id" not in self._columns:
            raise ValueError("columns must include fixture_id")
        self._init_schema()

    def _init_schema(self) -> None:
        col_defs = []
        for c in self._columns:
            if c == "fixture_id":
                col_defs.append('"fixture_id" INTEGER PRIMARY KEY')
            else:
                col_defs.append(f'"{c}"')  # no type token -> BLOB affinity, no coercion
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS feature_rows ({', '.join(col_defs)});")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feature_rows_comp_season ON feature_rows(competition_id, season_id);"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FeatureDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def upsert_rows(self, rows: Iterable[dict[str, Any]]) -> int:
        placeholders = ", ".join("?" for _ in self._columns)
        col_list = ", ".join(f'"{c}"' for c in self._columns)
        update_list = ", ".join(f'"{c}" = excluded."{c}"' for c in self._columns if c != "fixture_id")
        sql = (
            f"INSERT INTO feature_rows ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT(fixture_id) DO UPDATE SET {update_list};"
        )
        data = [tuple(row.get(c) for c in self._columns) for row in rows]
        cur = self._conn.executemany(sql, data)
        self._conn.commit()
        return len(data)

    def count_rows(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0]

    def get_row(self, fixture_id: int) -> dict[str, Any] | None:
        self._conn.row_factory = sqlite3.Row
        row = self._conn.execute("SELECT * FROM feature_rows WHERE fixture_id = ?", (fixture_id,)).fetchone()
        self._conn.row_factory = None
        return dict(row) if row else None

    def all_rows(self) -> list[dict[str, Any]]:
        self._conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in self._conn.execute("SELECT * FROM feature_rows").fetchall()]
        self._conn.row_factory = None
        return rows
