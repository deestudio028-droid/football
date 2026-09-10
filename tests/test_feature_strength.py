"""Tests for the attack/defence shrinkage methodology (spec §4)."""
import _pathfix  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from _feature_test_helpers import build_matches_db, make_raw_fixture
from features.config import SHRINKAGE_K
from features.feature_builder import build_feature_dataset
from features.strength import shrunk_rate


class TestShrunkRatePrimitive(unittest.TestCase):
    def test_zero_matches_equals_league_mean_exactly(self):
        self.assertEqual(shrunk_rate(team_sum=0.0, team_matches=0, league_mean=1.4), 1.4)

    def test_zero_matches_and_no_league_mean_is_none(self):
        self.assertIsNone(shrunk_rate(team_sum=0.0, team_matches=0, league_mean=None))

    def test_large_sample_converges_toward_raw_team_average(self):
        # 1000 matches, team average = 2.0, league mean = 1.0 -> shrinkage
        # should have negligible effect (k=5 vs n=1000).
        team_matches = 1000
        team_sum = 2.0 * team_matches
        result = shrunk_rate(team_sum, team_matches, league_mean=1.0, k=5)
        self.assertAlmostEqual(result, 2.0, places=2)

    def test_small_sample_pulled_toward_league_mean(self):
        # 1 match with 5 goals, league mean 1.0, k=5:
        # (5 + 5*1.0) / (1+5) = 10/6 = 1.667 -- pulled far from the raw 5.0.
        result = shrunk_rate(team_sum=5.0, team_matches=1, league_mean=1.0, k=5)
        self.assertAlmostEqual(result, 10 / 6)
        self.assertLess(result, 5.0)
        self.assertGreater(result, 1.0)

    def test_formula_matches_manual_calculation(self):
        result = shrunk_rate(team_sum=12.0, team_matches=6, league_mean=1.5, k=SHRINKAGE_K)
        expected = (12.0 + SHRINKAGE_K * 1.5) / (6 + SHRINKAGE_K)
        self.assertAlmostEqual(result, expected)


class TestStrengthInPipeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_match_of_league_season_has_null_strength(self):
        fixtures = [make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0, season_id=100)]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path)[0]
        # no prior league data AND no prior team data -> None, not 0
        self.assertIsNone(row["home_attack_strength_score"])
        self.assertIsNone(row["strength_diff"])

    def test_strength_score_uses_only_current_season(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=5, away_goals=0, season_id=100),
            make_raw_fixture(2, 2000, home_id=3, away_id=4, home_goals=1, away_goals=1, season_id=100),
            make_raw_fixture(3, 3000, home_id=1, away_id=3, home_goals=0, away_goals=0, season_id=100),  # target
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[3])[0]
        # team 1 has 1 prior match this season (5 goals for), league mean
        # goals-per-team-match so far = (5+0+1+1)/(2 matches*2) = 1.75
        league_mean = (5 + 0 + 1 + 1) / 4
        expected_attack = (5.0 + SHRINKAGE_K * league_mean) / (1 + SHRINKAGE_K)
        self.assertAlmostEqual(row["home_attack_strength_score"], expected_attack)
        self.assertAlmostEqual(row["league_mean_goals_per_team_match_season"], league_mean)


if __name__ == "__main__":
    unittest.main()
