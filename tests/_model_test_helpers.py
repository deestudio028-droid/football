"""Shared helper for model-layer tests: builds a small synthetic
features.db-shaped SQLite file using the real FeatureDB writer from
src/features/storage.py, so these tests exercise the exact schema/
column-typing behavior the real pipeline produces.
"""
import _pathfix  # noqa: F401
from pathlib import Path

from features.storage import FeatureDB

# A deliberately small, hand-picked column set: enough to exercise every
# rule in models/data.py (labels, metadata, a nullable feature column,
# the recommended competition_id feature) without needing all 274 real columns.
TEST_COLUMNS = [
    "fixture_id", "competition_id", "season_id", "unix", "home_id", "away_id",
    "feature_version", "generated_at",
    "label_home_goals", "label_away_goals", "label_result",
    "home_goals_for_per_match_season", "home_goals_for_per_match_season_n",
    "home_xg_for_per_match_season",  # deliberately nullable, like real xG
]


def make_row(
    fixture_id, competition_id, season_id, unix,
    label_result="H", label_home_goals=1, label_away_goals=0,
    feature_version="v1.0", home_goals=1.5, xg=None,
):
    return {
        "fixture_id": fixture_id, "competition_id": competition_id, "season_id": season_id,
        "unix": unix, "home_id": 1, "away_id": 2, "feature_version": feature_version,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "label_home_goals": label_home_goals, "label_away_goals": label_away_goals,
        "label_result": label_result,
        "home_goals_for_per_match_season": home_goals,
        "home_goals_for_per_match_season_n": 5,
        "home_xg_for_per_match_season": xg,
    }


def build_features_db(tmp_path: Path, rows: list[dict], columns: list[str] = TEST_COLUMNS) -> Path:
    db_path = tmp_path / "features.db"
    db = FeatureDB(db_path, columns)
    db.upsert_rows(rows)
    db.close()
    return db_path
