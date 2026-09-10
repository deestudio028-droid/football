"""Shared helper for feature tests: builds a real matches.db-shaped
SQLite file from a list of minimal synthetic fixture dicts, reusing the
already-tested ingestion normalize/db code so feature tests exercise
the exact same schema the real pipeline reads from.
"""
import _pathfix  # noqa: F401
from pathlib import Path

from ingestion.db import FixtureDB
from ingestion.normalize import normalize_fixture


def make_raw_fixture(
    fixture_id, unix, home_id, away_id, home_goals, away_goals,
    competition_id=1, season_id=100, status="FT",
    home_xg=None, away_xg=None, home_shots=None, away_shots=None,
    home_shots_on=None, away_shots_on=None,
):
    return {
        "id": fixture_id, "unix": unix, "home_id": home_id, "away_id": away_id,
        "home_name": f"Team{home_id}", "away_name": f"Team{away_id}",
        "home_goals": home_goals, "away_goals": away_goals,
        "competition_id": competition_id, "season_id": season_id, "status": status,
        "stats": {
            "home_xg": home_xg, "away_xg": away_xg,
            "home_shots": home_shots, "away_shots": away_shots,
            "home_shots_on": home_shots_on, "away_shots_on": away_shots_on,
        },
    }


def build_matches_db(tmp_path: Path, raw_fixtures: list[dict]) -> Path:
    db_path = tmp_path / "matches.db"
    db = FixtureDB(db_path)
    records = [normalize_fixture(f) for f in raw_fixtures]
    db.upsert_fixtures(records)
    db.close()
    return db_path
