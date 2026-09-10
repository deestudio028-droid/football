"""General correctness/shape tests for the feature pipeline -- not
leakage-specific (see test_feature_leakage.py for that), just "does it
produce the right shape of output, deterministically."
"""
import _pathfix  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from _feature_test_helpers import build_matches_db, make_raw_fixture
from features.config import FEATURE_VERSION
from features.feature_builder import build_feature_dataset
from features.storage import LABEL_COLUMNS, METADATA_COLUMNS


class TestFeatureBuilder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_row_per_requested_fixture(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=3, away_id=4, home_goals=2, away_goals=2),
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        rows = build_feature_dataset(db_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["fixture_id"] for r in rows}, {1, 2})

    def test_metadata_and_label_columns_present(self):
        fixtures = [make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=2, away_goals=1)]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path)[0]
        for col in METADATA_COLUMNS + LABEL_COLUMNS:
            self.assertIn(col, row, f"missing column: {col}")

    def test_feature_version_recorded(self):
        fixtures = [make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=0, away_goals=0)]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path)[0]
        self.assertEqual(row["feature_version"], FEATURE_VERSION)

    def test_label_matches_actual_result(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=3, away_goals=1),  # home win
            make_raw_fixture(2, 2000, home_id=1, away_id=2, home_goals=0, away_goals=0),  # draw
            make_raw_fixture(3, 3000, home_id=1, away_id=2, home_goals=0, away_goals=2),  # away win
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        rows = {r["fixture_id"]: r for r in build_feature_dataset(db_path)}
        self.assertEqual(rows[1]["label_result"], "H")
        self.assertEqual(rows[1]["label_home_goals"], 3)
        self.assertEqual(rows[1]["label_away_goals"], 1)
        self.assertEqual(rows[2]["label_result"], "D")
        self.assertEqual(rows[3]["label_result"], "A")

    def test_abandoned_fixture_gets_null_label_but_still_a_feature_row(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=None, away_goals=None, status="ABANDONED"),
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        rows = {r["fixture_id"]: r for r in build_feature_dataset(db_path)}
        self.assertIn(2, rows)  # a row is still generated
        self.assertIsNone(rows[2]["label_result"])
        self.assertIsNone(rows[2]["label_home_goals"])

    def test_abandoned_fixture_never_enters_anyones_history(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=None, away_goals=None, status="ABANDONED"),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=1, away_goals=1),
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        rows = {r["fixture_id"]: r for r in build_feature_dataset(db_path)}
        # team 1's history before fixture 3 should count only fixture 1
        # (2 matches played), not the abandoned fixture 2.
        self.assertEqual(rows[3]["home_matches_played_before_target"], 1)

    def test_deterministic_reproducibility(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=2, away_goals=1),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=1, away_goals=1),
            make_raw_fixture(3, 3000, home_id=3, away_id=1, home_goals=0, away_goals=2),
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        rows_a = build_feature_dataset(db_path)
        rows_b = build_feature_dataset(db_path)
        for a, b in zip(rows_a, rows_b):
            a = dict(a); b = dict(b)
            a.pop("generated_at"); b.pop("generated_at")
            self.assertEqual(a, b)

    def test_raw_fixture_table_untouched_by_feature_generation(self):
        import sqlite3
        fixtures = [make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0)]
        db_path = build_matches_db(self.tmp_path, fixtures)
        conn = sqlite3.connect(str(db_path))
        before = conn.execute("SELECT * FROM fixtures").fetchall()
        conn.close()

        build_feature_dataset(db_path)

        conn = sqlite3.connect(str(db_path))
        after = conn.execute("SELECT * FROM fixtures").fetchall()
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        self.assertEqual(before, after)
        self.assertEqual(tables, ["fixtures"])  # no feature table written into matches.db


if __name__ == "__main__":
    unittest.main()
