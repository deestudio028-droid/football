import _pathfix  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from ingestion.db import FixtureDB
from ingestion.normalize import normalize_fixture


def make_record(fixture_id: int, home_goals: int = 1) -> dict:
    return normalize_fixture({
        "id": fixture_id, "home_name": "A", "away_name": "B",
        "competition_id": 423, "season_id": 6484, "unix": 1738513800,
        "home_goals": home_goals, "away_goals": 0, "status": "FT",
    })


class TestFixtureDB(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_upsert_and_count(self):
        db = FixtureDB(self.tmp_path / "matches.db")
        db.upsert_fixture(make_record(1))
        db.upsert_fixture(make_record(2))
        self.assertEqual(db.count_fixtures(), 2)
        self.assertEqual(db.count_fixtures(competition_id=423, season_id=6484), 2)
        db.close()

    def test_upsert_same_fixture_id_deduplicates(self):
        db = FixtureDB(self.tmp_path / "matches.db")
        db.upsert_fixture(make_record(1, home_goals=1))
        db.upsert_fixture(make_record(1, home_goals=99))
        self.assertEqual(db.count_fixtures(), 1)
        row = db.get_fixture(1)
        self.assertEqual(row["home_goals"], 99)
        db.close()

    def test_missing_stats_are_queryable(self):
        db = FixtureDB(self.tmp_path / "matches.db")
        record = normalize_fixture({"id": 5, "home_name": "A", "away_name": "B"})
        db.upsert_fixture(record)
        incomplete = db.fixtures_with_missing_stats(min_missing=1)
        self.assertEqual(len(incomplete), 1)
        self.assertEqual(incomplete[0]["fixture_id"], 5)
        db.close()

    def test_reopening_db_preserves_data(self):
        db_path = self.tmp_path / "matches.db"
        db1 = FixtureDB(db_path)
        db1.upsert_fixture(make_record(42))
        db1.close()

        db2 = FixtureDB(db_path)
        self.assertEqual(db2.count_fixtures(), 1)
        db2.close()

    # -- numeric column typing regression tests (Phase 1.1) -----------

    def test_numeric_stat_fields_stored_as_real_or_integer_not_text(self):
        record = normalize_fixture({
            "id": 152908783, "home_name": "Manchester United", "away_name": "Fulham",
            "competition_id": 423, "season_id": 6484, "unix": 1723834800,
            "home_goals": 1, "away_goals": 0, "status": "FT",
            "stats": {
                "home_xg": 1.7564, "away_xg": 0.6385,
                "home_xgot": 2.1911, "away_xgot": 1.5774,
                "shots": 24, "home_shots": 12, "away_shots": 12,
                "shots_on": 7, "home_possession": 55, "away_possession": 45,
                "corners": 15, "home_fouls": 12, "away_fouls": 10,
                "home_yellow_cards": 2, "away_yellow_cards": 3,
            },
        })
        db = FixtureDB(self.tmp_path / "matches.db")
        db.upsert_fixture(record)

        cur = db._conn.execute(
            "SELECT typeof(stat_home_xg), typeof(stat_away_xgot), typeof(stat_shots), "
            "typeof(stat_home_possession), typeof(home_goals), typeof(unix), "
            "typeof(competition_id) FROM fixtures WHERE fixture_id = ?",
            (152908783,),
        )
        xg_type, xgot_type, shots_type, poss_type, goals_type, unix_type, comp_type = cur.fetchone()
        self.assertEqual(xg_type, "real")
        self.assertEqual(xgot_type, "real")
        self.assertEqual(shots_type, "integer")
        self.assertEqual(poss_type, "integer")
        self.assertEqual(goals_type, "integer")
        self.assertEqual(unix_type, "integer")
        self.assertEqual(comp_type, "integer")
        db.close()

    def test_categorical_text_fields_remain_text(self):
        record = normalize_fixture({
            "id": 1, "home_name": "Arsenal", "away_name": "Chelsea",
            "ht_score": "1-0", "home_formation": "4-3-3", "status": "FT",
        })
        db = FixtureDB(self.tmp_path / "matches.db")
        db.upsert_fixture(record)
        cur = db._conn.execute(
            "SELECT typeof(home_name), typeof(ht_score), typeof(home_formation) "
            "FROM fixtures WHERE fixture_id = 1"
        )
        name_type, ht_type, formation_type = cur.fetchone()
        self.assertEqual(name_type, "text")
        self.assertEqual(ht_type, "text")
        self.assertEqual(formation_type, "text")
        db.close()

    def test_aggregation_works_without_explicit_cast(self):
        # This is the concrete behavior the type fix must guarantee: SUM/AVG
        # over stat columns must return real numeric results directly, not
        # rely on SQLite's implicit numeric-string coercion in aggregates.
        db = FixtureDB(self.tmp_path / "matches.db")
        for i, (shots, xg) in enumerate([(24, 1.7564), (10, 0.9), (18, 2.25)], start=1):
            record = normalize_fixture({
                "id": i, "home_name": "A", "away_name": "B",
                "competition_id": 423, "season_id": 6484,
                "stats": {"shots": shots, "home_xg": xg},
            })
            db.upsert_fixture(record)

        total_shots = db._conn.execute("SELECT SUM(stat_shots) FROM fixtures").fetchone()[0]
        avg_xg = db._conn.execute("SELECT AVG(stat_home_xg) FROM fixtures").fetchone()[0]
        self.assertEqual(total_shots, 52)
        self.assertAlmostEqual(avg_xg, (1.7564 + 0.9 + 2.25) / 3, places=6)
        self.assertIsInstance(total_shots, int)
        self.assertIsInstance(avg_xg, float)
        db.close()

    def test_null_numeric_field_remains_null_not_coerced_to_zero(self):
        # A missing numeric field must stay NULL (and be trackable via
        # missing_fields), never silently become 0 -- that would corrupt
        # any downstream average/sum.
        record = normalize_fixture({"id": 99, "home_name": "A", "away_name": "B"})
        db = FixtureDB(self.tmp_path / "matches.db")
        db.upsert_fixture(record)
        row = db.get_fixture(99)
        self.assertIsNone(row["stat_home_xg"])
        self.assertIsNone(row["home_goals"])
        db.close()


if __name__ == "__main__":
    unittest.main()
