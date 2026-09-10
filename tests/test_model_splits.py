"""Tests for src/models/splits.py -- the walk-forward temporal split
layer. Covers "SPLIT SAFETY REQUIREMENTS" and the split-related items
in "IMPORTANT TESTS" from the Phase 3 Step 1 task.
"""
import _pathfix  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from _model_test_helpers import build_features_db, make_row

from models.config import FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, WALK_FORWARD_FOLDS
from models.data import load_supervised_dataset
from models.splits import (
    SplitSafetyError,
    final_split,
    iter_walk_forward_folds,
    season_ids_for,
    split_by_seasons,
    verify_temporal_safety,
)


class TestFoldConfiguration(unittest.TestCase):
    def test_exactly_three_walk_forward_folds(self):
        self.assertEqual(len(WALK_FORWARD_FOLDS), 3)

    def test_fold_seasons_match_approved_design(self):
        expected = [
            (("2020/2021", "2021/2022"), ("2022/2023",)),
            (("2020/2021", "2021/2022", "2022/2023"), ("2023/2024",)),
            (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), ("2024/2025",)),
        ]
        actual = [(f.train_seasons, f.validation_seasons) for f in WALK_FORWARD_FOLDS]
        self.assertEqual(actual, expected)

    def test_2025_26_excluded_from_every_walk_forward_fold(self):
        for fold in WALK_FORWARD_FOLDS:
            self.assertNotIn("2025/2026", fold.train_seasons)
            self.assertNotIn("2025/2026", fold.validation_seasons)

    def test_final_test_is_only_2025_26(self):
        self.assertEqual(FINAL_TEST_SEASONS, ("2025/2026",))
        self.assertNotIn("2025/2026", FINAL_TRAIN_SEASONS)


class TestSplitsAgainstRealDataset(unittest.TestCase):
    """Runs against the real, already-generated data/processed/features.db
    -- these are integration-level checks that the actual shipped dataset
    satisfies every safety property, not just synthetic examples.
    """

    @classmethod
    def setUpClass(cls):
        real_db = Path("data/processed/features.db")
        if not real_db.exists():
            raise unittest.SkipTest("data/processed/features.db not present in this environment")
        cls.dataset = load_supervised_dataset(real_db)

    def test_three_folds_all_temporally_safe(self):
        folds = list(iter_walk_forward_folds(self.dataset))
        self.assertEqual(len(folds), 3)
        for fold, train, val in folds:
            self.assertGreater(len(train), 0)
            self.assertGreater(len(val), 0)
            # verify_temporal_safety already ran inside iter_walk_forward_folds
            # and would have raised if unsafe; re-run explicitly here too so
            # this test fails on its own if that internal call is ever removed.
            verify_temporal_safety(train, val)

    def test_no_fixture_overlap_between_any_fold_train_and_validation(self):
        for fold, train, val in iter_walk_forward_folds(self.dataset):
            overlap = set(train.metadata["fixture_id"]) & set(val.metadata["fixture_id"])
            self.assertEqual(overlap, set(), f"{fold.name} has overlapping fixtures: {overlap}")

    def test_every_training_fixture_before_every_validation_fixture(self):
        for fold, train, val in iter_walk_forward_folds(self.dataset):
            self.assertLess(
                train.metadata["unix"].max(), val.metadata["unix"].min(),
                f"{fold.name}: a training fixture is not strictly before validation",
            )

    def test_2025_26_never_appears_in_any_walk_forward_fold_data(self):
        season_2025_26_ids = season_ids_for(("2025/2026",))
        for fold, train, val in iter_walk_forward_folds(self.dataset):
            self.assertTrue(set(train.metadata["season_id"]).isdisjoint(season_2025_26_ids))
            self.assertTrue(set(val.metadata["season_id"]).isdisjoint(season_2025_26_ids))

    def test_final_split_contains_only_2025_26_in_test(self):
        train, test = final_split(self.dataset)
        season_2025_26_ids = season_ids_for(("2025/2026",))
        self.assertTrue(set(test.metadata["season_id"]).issubset(season_2025_26_ids))
        self.assertTrue(set(train.metadata["season_id"]).isdisjoint(season_2025_26_ids))

    def test_final_train_before_final_test(self):
        train, test = final_split(self.dataset)
        self.assertLess(train.metadata["unix"].max(), test.metadata["unix"].min())

    def test_no_fixture_overlap_in_final_split(self):
        train, test = final_split(self.dataset)
        overlap = set(train.metadata["fixture_id"]) & set(test.metadata["fixture_id"])
        self.assertEqual(overlap, set())

    def test_no_validation_fixture_appears_in_final_test(self):
        _, final_test = final_split(self.dataset)
        final_test_ids = set(final_test.metadata["fixture_id"])
        for fold, train, val in iter_walk_forward_folds(self.dataset):
            overlap = set(val.metadata["fixture_id"]) & final_test_ids
            self.assertEqual(overlap, set(), f"{fold.name} validation overlaps final test: {overlap}")

    def test_abandoned_fixture_excluded_from_every_split(self):
        abandoned_id = 420450481
        train, test = final_split(self.dataset)
        self.assertNotIn(abandoned_id, set(train.metadata["fixture_id"]))
        self.assertNotIn(abandoned_id, set(test.metadata["fixture_id"]))
        for fold, ftrain, fval in iter_walk_forward_folds(self.dataset):
            self.assertNotIn(abandoned_id, set(ftrain.metadata["fixture_id"]))
            self.assertNotIn(abandoned_id, set(fval.metadata["fixture_id"]))

    def test_splits_are_deterministic(self):
        folds_a = [(f.name, len(tr), len(va)) for f, tr, va in iter_walk_forward_folds(self.dataset)]
        folds_b = [(f.name, len(tr), len(va)) for f, tr, va in iter_walk_forward_folds(self.dataset)]
        self.assertEqual(folds_a, folds_b)


class TestSplitSafetyEnforcement(unittest.TestCase):
    """Synthetic tests proving verify_temporal_safety actually catches
    unsafe splits, not just that real data happens to pass.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_detects_fixture_overlap(self):
        rows = [make_row(1, 423, 100, 1000), make_row(2, 423, 200, 2000)]
        db_path = build_features_db(self.tmp_path, rows)
        ds = load_supervised_dataset(db_path)
        # Manually construct an "overlap" scenario: same row in both sides.
        train = ds
        evald = ds
        with self.assertRaises(SplitSafetyError):
            verify_temporal_safety(train, evald)

    def test_detects_out_of_order_unix(self):
        # train has a LATER unix than eval -- must be rejected.
        rows = [
            make_row(1, 423, 100, 5000),  # "train" row, later in time
            make_row(2, 423, 200, 1000),  # "eval" row, earlier in time
        ]
        db_path = build_features_db(self.tmp_path, rows)
        ds = load_supervised_dataset(db_path)
        # Build train/eval subsets directly by fixture_id instead of season,
        # since these two rows use fake season_ids not in competitions.json.
        from models.data import SupervisedDataset
        train = SupervisedDataset(
            metadata=ds.metadata[ds.metadata["fixture_id"] == 1].reset_index(drop=True),
            X=ds.X[ds.metadata["fixture_id"] == 1].reset_index(drop=True),
            y=ds.y[ds.metadata["fixture_id"] == 1].reset_index(drop=True),
        )
        evald = SupervisedDataset(
            metadata=ds.metadata[ds.metadata["fixture_id"] == 2].reset_index(drop=True),
            X=ds.X[ds.metadata["fixture_id"] == 2].reset_index(drop=True),
            y=ds.y[ds.metadata["fixture_id"] == 2].reset_index(drop=True),
        )
        with self.assertRaises(SplitSafetyError):
            verify_temporal_safety(train, evald)

    def test_unknown_season_name_raises(self):
        with self.assertRaises(KeyError):
            season_ids_for(("Not A Real Season",))


if __name__ == "__main__":
    unittest.main()
