"""Leakage-safety proofs for the feature pipeline (Step 7 of the Phase 2
feature engineering task). This is one of the most important test
files in the project: every test here tries to construct a scenario
where future information COULD leak into a past fixture's features,
and asserts that it doesn't.
"""
import _pathfix  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from _feature_test_helpers import build_matches_db, make_raw_fixture
from features.feature_builder import build_feature_dataset


class TestFeatureLeakage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_1_target_fixture_never_in_its_own_rolling_history(self):
        # Team 1's target match has a very distinctive goal count (77).
        # If the target ever leaked into its own history window, some
        # rolling "goals_for" feature would reflect it, but a single
        # match can't be its own *history* -- there's nothing before it
        # to roll over except genuinely earlier matches. Assert none of
        # team 1's earlier matches happen to also be 77 goals as a
        # sanity guard, then assert the target's own goals don't appear
        # anywhere in its home-side rolling window average given the
        # small integer window means one contaminating value would be
        # obviously visible.
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=2, away_goals=0),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=3, away_goals=0),
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=77, away_goals=0),  # target
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        rows = build_feature_dataset(db_path, fixture_ids=[4])
        row = rows[0]
        # last5 average of [1, 2, 3] = 2.0 -- if 77 leaked in, average would be huge.
        self.assertAlmostEqual(row["home_goals_for_per_match_last5"], 2.0)
        self.assertEqual(row["home_goals_for_per_match_last5_n"], 3)

    def test_2_future_fixture_cannot_affect_earlier_fixtures_features(self):
        base = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=2, away_goals=1),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=0, away_goals=0),  # T
        ]
        with_future = base + [
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=99, away_goals=0),
        ]
        db_a_dir = self.tmp_path / "a"
        db_a_dir.mkdir(exist_ok=True)
        db_a = build_matches_db(db_a_dir, base)
        db_b_dir = self.tmp_path / "b"
        db_b_dir.mkdir(exist_ok=True)
        db_b = build_matches_db(db_b_dir, with_future)

        row_a = build_feature_dataset(db_a, fixture_ids=[3])[0]
        row_b = build_feature_dataset(db_b, fixture_ids=[3])[0]
        self.assertEqual(row_a["home_goals_for_per_match_last5"], row_b["home_goals_for_per_match_last5"])
        self.assertEqual(row_a["home_points_last5"], row_b["home_points_last5"])
        self.assertEqual(row_a["home_matches_played_before_target"], row_b["home_matches_played_before_target"])

    def test_3_reordering_future_rows_does_not_change_past_features(self):
        past = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=2, away_goals=1),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=0, away_goals=0),  # T
        ]
        future_order_1 = past + [
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=9, away_goals=0),
            make_raw_fixture(5, 5000, home_id=1, away_id=6, home_goals=0, away_goals=9),
        ]
        future_order_2 = past + [
            make_raw_fixture(5, 5000, home_id=1, away_id=6, home_goals=0, away_goals=9),
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=9, away_goals=0),
        ]
        dir1 = self.tmp_path / "o1"; dir1.mkdir()
        dir2 = self.tmp_path / "o2"; dir2.mkdir()
        db1 = build_matches_db(dir1, future_order_1)
        db2 = build_matches_db(dir2, future_order_2)

        row1 = build_feature_dataset(db1, fixture_ids=[3])[0]
        row2 = build_feature_dataset(db2, fixture_ids=[3])[0]
        row1.pop("generated_at"); row2.pop("generated_at")  # wall-clock, not feature content
        self.assertEqual(row1, row2)

    def test_4_removing_future_fixtures_does_not_change_features_at_T(self):
        with_future = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=2, away_goals=1),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=0, away_goals=0),  # T
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=9, away_goals=0),
        ]
        without_future = with_future[:3]
        dir1 = self.tmp_path / "wf"; dir1.mkdir()
        dir2 = self.tmp_path / "nf"; dir2.mkdir()
        db_wf = build_matches_db(dir1, with_future)
        db_nf = build_matches_db(dir2, without_future)

        row_wf = build_feature_dataset(db_wf, fixture_ids=[3])[0]
        row_nf = build_feature_dataset(db_nf, fixture_ids=[3])[0]
        row_wf.pop("generated_at"); row_nf.pop("generated_at")
        self.assertEqual(row_wf, row_nf)

    def test_5_season_aggregate_only_uses_matches_before_T(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=3, away_goals=0, season_id=100),
            make_raw_fixture(2, 2000, home_id=3, away_id=4, home_goals=1, away_goals=1, season_id=100),
            make_raw_fixture(3, 3000, home_id=1, away_id=3, home_goals=2, away_goals=2, season_id=100),  # T
            make_raw_fixture(4, 4000, home_id=5, away_id=6, home_goals=5, away_goals=0, season_id=100),  # after T
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[3])[0]
        # league_home_advantage_season should reflect only fixtures 1 and 2
        # (home_goals-away_goals: 3 and 0) => mean = 1.5, n=2. Fixture 4's
        # home_goals-away_goals=5 must NOT be included.
        self.assertEqual(row["league_home_advantage_season_n"], 2)
        self.assertAlmostEqual(row["league_home_advantage_season"], 1.5)

    def test_6_venue_restricted_stats_exclude_target(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=4, away_goals=0),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=6, away_goals=0),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=100, away_goals=0),  # T, home venue
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[3])[0]
        # mean of [4, 6] = 5.0 -- if T's own 100 leaked in, this would be huge.
        self.assertAlmostEqual(row["home_goals_for_home_venue_season"], 5.0)
        self.assertEqual(row["home_goals_for_home_venue_season_n"], 2)

    def test_7_xg_features_never_use_post_target_xg(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0, home_xg=1.0, away_xg=0.5),
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=1, away_goals=0, home_xg=1.2, away_xg=0.4),
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=1, away_goals=0, home_xg=1.4, away_xg=0.3),
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=0, away_goals=0, home_xg=0.9, away_xg=0.3),  # T
            make_raw_fixture(5, 5000, home_id=1, away_id=6, home_goals=0, away_goals=0, home_xg=999.0, away_xg=0.0),  # future, extreme xg
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[4])[0]
        # mean of [1.0, 1.2, 1.4] = 1.2 -- must not be pulled toward 999.0
        # or include T's own 0.9.
        self.assertAlmostEqual(row["home_xg_for_per_match_last5"], 1.2)
        self.assertEqual(row["home_xg_for_per_match_last5_coverage_n"], 3)

    def test_8_form_features_never_use_post_target_results(self):
        fixtures = [
            make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0),  # win
            make_raw_fixture(2, 2000, home_id=1, away_id=3, home_goals=0, away_goals=1),  # loss
            make_raw_fixture(3, 3000, home_id=1, away_id=4, home_goals=1, away_goals=1),  # draw
            make_raw_fixture(4, 4000, home_id=1, away_id=5, home_goals=2, away_goals=2),  # T, draw
            make_raw_fixture(5, 5000, home_id=1, away_id=6, home_goals=5, away_goals=0),  # future win
        ]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[4])[0]
        # form before T: 1 win, 1 loss, 1 draw -> win_rate 1/3, loss_rate
        # 1/3, points = (3+0+1)/3 = 1.333. Must NOT include T's own draw
        # or the future win.
        self.assertAlmostEqual(row["home_win_rate_last5"], 1 / 3)
        self.assertAlmostEqual(row["home_loss_rate_last5"], 1 / 3)
        self.assertAlmostEqual(row["home_points_last5"], 4 / 3)
        self.assertEqual(row["home_points_last5_n"], 3)

    def test_9_no_odds_columns_exist_at_all(self):
        # Odds were never ingested (spec §J) -- the strongest possible
        # leakage guard for "don't accidentally use closing/future odds"
        # is that no odds-derived column exists in the feature schema to
        # leak through in the first place. This test enforces that stays
        # true rather than assuming it.
        fixtures = [make_raw_fixture(1, 1000, home_id=1, away_id=2, home_goals=1, away_goals=0)]
        db_path = build_matches_db(self.tmp_path, fixtures)
        row = build_feature_dataset(db_path, fixture_ids=[1])[0]
        odds_columns = [k for k in row.keys() if "odds" in k.lower()]
        self.assertEqual(odds_columns, [])

    def test_context_never_mutated_before_feature_read(self):
        # Direct test of the FeatureContext ordering contract itself:
        # history_before() for a team must be empty/short until record()
        # is explicitly called, proving the "read then write" ordering
        # feature_builder relies on is actually enforced by history.py,
        # not just true by accident of call order in feature_builder.
        from features.history import FeatureContext, team_match_record

        ctx = FeatureContext()
        fixture = {
            "fixture_id": 1, "unix": 1000, "competition_id": 1, "season_id": 100,
            "home_id": 1, "away_id": 2, "home_goals": 3, "away_goals": 0, "status": "FT",
        }
        self.assertEqual(ctx.history_before(1), [])
        ctx.record(fixture)
        history = ctx.history_before(1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["fixture_id"], 1)


if __name__ == "__main__":
    unittest.main()
