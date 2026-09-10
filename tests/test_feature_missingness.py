"""Missing-data policy tests (spec §6): NULL stays NULL, minimum
coverage thresholds are enforced, and deferred field families never
appear in the output at all.
"""
import _pathfix  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from _feature_test_helpers import build_matches_db, make_raw_fixture
from features.feature_builder import build_feature_dataset
from features.missingness import describe_missingness
from features.rolling import gated_mean, windowed_mean


class TestMissingnessInPipeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_below_minimum_coverage_is_null_not_zero(self):
        # Only 2 prior matches -- below last5's minimum of 3.
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=2, away_goals=0),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=1, away_goals=1),  # target
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[3])[0]
        self.assertIsNone(row["home_goals_for_per_match_last5"])
        self.assertNotEqual(row["home_goals_for_per_match_last5"], 0)
        self.assertEqual(row["home_goals_for_per_match_last5_n"], 2)  # n still reported

    def test_missing_xg_on_some_matches_reduces_coverage_n_not_n(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0, home_xg=1.0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=1, away_goals=0, home_xg=None),  # pre-cutover style
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=1, away_goals=0, home_xg=1.5),
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=1, away_goals=0, home_xg=2.0),
            make_raw_fixture(5, 5000, home_id=1, away_id=6, home_goals=0, away_goals=0),  # target
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[5])[0]
        self.assertEqual(row["home_xg_for_per_match_last5_n"], 4)          # 4 matches in window
        self.assertEqual(row["home_xg_for_per_match_last5_coverage_n"], 3)  # only 3 had xg
        self.assertAlmostEqual(row["home_xg_for_per_match_last5"], 1.5)    # mean of [1.0, 1.5, 2.0], not counting the null

    def test_deferred_fields_never_appear(self):
        fixtures = [make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0)]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path)[0]
        for banned_substring in ["tackle", "goal_kick", "throw_in", "red_card", "offside", "odds"]:
            hits = [k for k in row if banned_substring in k.lower()]
            self.assertEqual(hits, [], f"deferred field family leaked into output: {hits}")

    def test_insufficient_history_flag_matches_threshold(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=1, away_goals=0),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=0, away_goals=0),  # target, 2 prior -> insufficient (<3)
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[3])[0]
        self.assertTrue(row["home_insufficient_history"])
        self.assertEqual(row["home_matches_played_before_target"], 2)


class TestMissingnessPrimitives(unittest.TestCase):
    def test_windowed_mean_empty_returns_none(self):
        mean, n, cov = windowed_mean([], lambda m: m.get("x"))
        self.assertIsNone(mean)
        self.assertEqual(n, 0)
        self.assertEqual(cov, 0)

    def test_gated_mean_respects_season_minimum(self):
        matches = [{"season_id": 1, "goals_for": 2}]  # only 1 match, season min is 2
        value, n, cov = gated_mean(matches, "season", 1, lambda m: m["goals_for"])
        self.assertIsNone(value)
        self.assertEqual(n, 1)

    def test_describe_missingness(self):
        rows = [{"x": 1}, {"x": None}, {"x": 3}, {"x": None}]
        result = describe_missingness(rows, "x")
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["null"], 2)
        self.assertEqual(result["pct_null"], 50.0)

    def test_describe_missingness_empty(self):
        result = describe_missingness([], "x")
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["pct_null"])


if __name__ == "__main__":
    unittest.main()
