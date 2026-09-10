import _pathfix  # noqa: F401
import unittest

from ingestion.normalize import normalize_fixture, NormalizationError


class TestNormalize(unittest.TestCase):
    def test_full_record_has_no_missing_fields(self):
        # elapsed_seconds/time_added/competition_predictability are left
        # None deliberately -- these are genuinely-null on real completed
        # fixtures (confirmed in the audit sample), so this test expects
        # them to show up as explicitly-tracked missing fields rather than
        # silently vanishing. That is the behavior being verified.
        raw = {
            "id": 215022547,
            "home_name": "Arsenal",
            "away_name": "Manchester City",
            "competition_id": 423,
            "competition_country": "England",
            "competition_name": "Premier League",
            "competition_type": "league",
            "competition_predictability": None,
            "season_id": 6484,
            "season": "2024/2025",
            "season_progress": 61,
            "home_id": 5303,
            "away_id": 5310,
            "winning_team": "home",
            "status": "FT",
            "home_goals": 5,
            "away_goals": 1,
            "ht_score": "1-0",
            "elapsed": 90,
            "elapsed_seconds": None,
            "time_added": None,
            "home_position": 1,
            "away_position": 2,
            "unix": 1738513800,
            "home_played": 23,
            "away_played": 23,
            "home_formation": "4-3-3",
            "away_formation": "4-2-3-1",
            "venue": "Emirates Stadium",
            "has_odds": True,
            "referee_id": 535,
            "date": "2025-02-02T16:30:00.000000Z",
            "ko_human": "Sun 2nd, 16:30",
            "is_friendly": False,
            "is_cup": False,
            "stats": {f: 1 for f in [
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
            ]},
        }
        record = normalize_fixture(raw)
        self.assertEqual(record["fixture_id"], 215022547)
        self.assertEqual(
            record["missing_fields"],
            ["competition_predictability", "elapsed_seconds", "time_added"],
        )
        self.assertFalse(record["is_complete"])
        self.assertEqual(record["stat_home_xg"], 1)
        # Everything else -- all 43 stat fields plus every other top-level
        # field -- was provided and must not be flagged as missing.
        self.assertNotIn("stat_home_shots", record["missing_fields"])
        self.assertNotIn("home_goals", record["missing_fields"])

    def test_missing_stats_block_is_tracked_explicitly(self):
        raw = {"id": 1, "home_name": "A", "away_name": "B"}
        record = normalize_fixture(raw)
        self.assertFalse(record["is_complete"])
        self.assertIn("stats", record["missing_fields"])
        self.assertIn("stat_home_xg", record["missing_fields"])
        self.assertIsNone(record["stat_home_xg"])
        self.assertIn("home_goals", record)
        self.assertIsNone(record["home_goals"])

    def test_requires_id(self):
        with self.assertRaises(NormalizationError):
            normalize_fixture({"home_name": "A"})

    def test_null_id_rejected(self):
        with self.assertRaises(NormalizationError):
            normalize_fixture({"id": None})

    def test_pre_xg_cutover_fixture_has_null_xg_tracked_not_zero(self):
        raw = {
            "id": 23096420, "home_name": "Bihar", "away_name": "Mizoram",
            "stats": {
                "home_xg": None, "away_xg": None, "home_xgot": None, "away_xgot": None,
                "home_shots": 10, "away_shots": 8,
            },
        }
        record = normalize_fixture(raw)
        self.assertIsNone(record["stat_home_xg"])
        self.assertIn("stat_home_xg", record["missing_fields"])
        self.assertEqual(record["stat_home_shots"], 10)
        self.assertNotIn("stat_home_shots", record["missing_fields"])


if __name__ == "__main__":
    unittest.main()
