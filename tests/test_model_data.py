"""Tests for src/models/data.py -- the read-only feature_rows loader.
Covers the "DATA LOADING REQUIREMENTS" and "FEATURE MATRIX RULES" from
the Phase 3 Step 1 task in full.
"""
import _pathfix  # noqa: F401
import sqlite3
import tempfile
import unittest
from pathlib import Path

from _model_test_helpers import build_features_db, make_row

from models.data import (
    FeatureVersionError,
    ModelDataError,
    UnexpectedLabelError,
    get_feature_columns,
    load_feature_rows,
    load_supervised_dataset,
)


class TestFeatureVersionEnforcement(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_single_correct_version_loads_fine(self):
        rows = [make_row(1, 423, 100, 1000), make_row(2, 423, 100, 2000)]
        db_path = build_features_db(self.tmp_path, rows)
        df = load_feature_rows(db_path)
        self.assertEqual(len(df), 2)

    def test_multiple_feature_versions_fails_loudly(self):
        rows = [
            make_row(1, 423, 100, 1000, feature_version="v1.0"),
            make_row(2, 423, 100, 2000, feature_version="v1.1"),
        ]
        db_path = build_features_db(self.tmp_path, rows)
        with self.assertRaises(FeatureVersionError):
            load_feature_rows(db_path)

    def test_wrong_single_version_fails_loudly(self):
        rows = [make_row(1, 423, 100, 1000, feature_version="v0.9")]
        db_path = build_features_db(self.tmp_path, rows)
        with self.assertRaises(FeatureVersionError):
            load_feature_rows(db_path)


class TestLabelValidation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_h_d_a_allowed(self):
        rows = [
            make_row(1, 423, 100, 1000, label_result="H"),
            make_row(2, 423, 100, 2000, label_result="D"),
            make_row(3, 423, 100, 3000, label_result="A"),
        ]
        db_path = build_features_db(self.tmp_path, rows)
        ds = load_supervised_dataset(db_path)
        self.assertEqual(set(ds.y.unique()), {"H", "D", "A"})

    def test_unexpected_label_value_fails_loudly(self):
        rows = [
            make_row(1, 423, 100, 1000, label_result="H"),
            make_row(2, 423, 100, 2000, label_result="WIN"),  # bogus
        ]
        db_path = build_features_db(self.tmp_path, rows)
        with self.assertRaises(UnexpectedLabelError):
            load_supervised_dataset(db_path)

    def test_label_result_is_not_null_is_the_supervised_filter(self):
        rows = [
            make_row(1, 423, 100, 1000, label_result="H"),
            make_row(2, 423, 100, 2000, label_result=None, label_home_goals=None, label_away_goals=None),
        ]
        db_path = build_features_db(self.tmp_path, rows)
        ds = load_supervised_dataset(db_path)
        self.assertEqual(len(ds), 1)
        self.assertEqual(list(ds.metadata["fixture_id"]), [1])

    def test_known_unlabeled_fixture_is_excluded(self):
        # 420450481 is the real ABANDONED fixture ID -- even if it somehow
        # had a non-null label_result in a hypothetical bad dataset, its
        # presence should be treated as suspicious. Here we just confirm
        # the normal (null-label) case correctly excludes it.
        rows = [
            make_row(1, 423, 100, 1000, label_result="H"),
            make_row(420450481, 200, 999, 5000, label_result=None, label_home_goals=None, label_away_goals=None),
        ]
        db_path = build_features_db(self.tmp_path, rows)
        ds = load_supervised_dataset(db_path)
        self.assertNotIn(420450481, set(ds.metadata["fixture_id"]))


class TestFeatureMatrixRules(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _dataset(self):
        rows = [make_row(i, 423, 100, 1000 * i) for i in range(1, 6)]
        db_path = build_features_db(self.tmp_path, rows)
        return load_supervised_dataset(db_path)

    def test_label_columns_excluded_from_X(self):
        ds = self._dataset()
        for col in ["label_home_goals", "label_away_goals", "label_result"]:
            self.assertNotIn(col, ds.X.columns)

    def test_fixture_id_excluded_from_X(self):
        ds = self._dataset()
        self.assertNotIn("fixture_id", ds.X.columns)

    def test_generated_at_excluded_from_X(self):
        ds = self._dataset()
        self.assertNotIn("generated_at", ds.X.columns)

    def test_feature_version_excluded_from_X(self):
        ds = self._dataset()
        self.assertNotIn("feature_version", ds.X.columns)

    def test_season_id_and_unix_not_auto_included_but_available_as_metadata(self):
        ds = self._dataset()
        self.assertNotIn("season_id", ds.X.columns)
        self.assertNotIn("unix", ds.X.columns)
        self.assertIn("season_id", ds.metadata.columns)
        self.assertIn("unix", ds.metadata.columns)

    def test_competition_id_available_as_recommended_feature(self):
        ds = self._dataset()
        self.assertIn("competition_id", ds.X.columns)
        # also still available as metadata for splitting/auditing
        self.assertIn("competition_id", ds.metadata.columns)

    def test_get_feature_columns_matches_dataset_X_columns(self):
        rows = [make_row(1, 423, 100, 1000)]
        db_path = build_features_db(self.tmp_path, rows)
        df = load_feature_rows(db_path)
        cols = get_feature_columns(df)
        ds = load_supervised_dataset(db_path)
        self.assertEqual(cols, list(ds.X.columns))


class TestNoBlanketDropna(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_nullable_feature_column_does_not_remove_rows(self):
        # home_xg_for_per_match_season is null for row 2 (pre-cutover-style)
        # but populated for row 1 -- a blanket dropna() over the whole
        # feature matrix would drop row 2 entirely. It must not.
        rows = [
            make_row(1, 423, 100, 1000, xg=1.2),
            make_row(2, 423, 100, 2000, xg=None),
        ]
        db_path = build_features_db(self.tmp_path, rows)
        ds = load_supervised_dataset(db_path)
        self.assertEqual(len(ds), 2)
        self.assertTrue(ds.X["home_xg_for_per_match_season"].isna().any())
        self.assertFalse(ds.X["home_xg_for_per_match_season"].isna().all())


class TestReadOnlyBehavior(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loading_does_not_modify_features_db(self):
        rows = [make_row(1, 423, 100, 1000), make_row(2, 423, 100, 2000)]
        db_path = build_features_db(self.tmp_path, rows)

        before = db_path.read_bytes()
        load_supervised_dataset(db_path)
        after = db_path.read_bytes()
        self.assertEqual(before, after)

    def test_connection_is_actually_read_only(self):
        # Belt-and-suspenders: the module's own read-only URI connection
        # must reject a write attempt, not just "happen to never write."
        rows = [make_row(1, 423, 100, 1000)]
        db_path = build_features_db(self.tmp_path, rows)
        from models.data import _read_only_connection
        conn = _read_only_connection(db_path)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("DELETE FROM feature_rows")
        conn.close()


if __name__ == "__main__":
    unittest.main()
