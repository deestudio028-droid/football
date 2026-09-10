"""Tests for src/models/train.py -- real LogisticRegression and
HistGradientBoostingClassifier candidates.

This entire module requires scikit-learn. If it is not installed in
the environment running the test suite, every test class here is
skipped (not failed, not worked around) via unittest.skipUnless, and
the skip reason names scikit-learn explicitly so `unittest discover`
output makes the gap obvious rather than silently reporting a smaller
total test count.
"""
import _pathfix  # noqa: F401
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from models import train as train_module
    _IMPORT_ERROR = None
except ImportError as exc:  # scikit-learn not installed in this environment
    train_module = None
    _IMPORT_ERROR = str(exc)

from models.baselines import CLASS_ORDER
from models.config import RECOMMENDED_CONTEXT_FEATURE
from models.evaluate import validate_probabilities

_SKIP_REASON = f"scikit-learn not available in this environment: {_IMPORT_ERROR}"


def _synthetic_frame(n: int, seed: int = 0, competitions=(423, 564)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat_a": rng.normal(size=n),
        "feat_b": rng.normal(size=n),
        RECOMMENDED_CONTEXT_FEATURE: rng.choice(competitions, size=n),
    })


def _synthetic_dataset(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = _synthetic_frame(n, seed=seed)
    y = pd.Series(rng.choice(list(CLASS_ORDER), size=n))
    return X, y


@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class TestLogisticRegressionPreprocessorLeakageSafety(unittest.TestCase):
    def test_fit_statistics_are_unaffected_by_any_later_transform_call(self):
        X_train = _synthetic_frame(50, seed=1)
        # Deliberately wild, out-of-distribution "validation" data -- if
        # transform() ever mutated fitted state, these calls would reveal it.
        X_val_a = _synthetic_frame(20, seed=2) * 1000
        X_val_b = _synthetic_frame(20, seed=3) * -1000

        pre = train_module.LogisticRegressionPreprocessor().fit(X_train)
        stats_before = pre._imputer.statistics_.copy()
        scale_before = pre._scaler.scale_.copy()
        mean_before = pre._scaler.mean_.copy()

        pre.transform(X_val_a)
        pre.transform(X_val_b)

        np.testing.assert_array_equal(pre._imputer.statistics_, stats_before)
        np.testing.assert_array_equal(pre._scaler.scale_, scale_before)
        np.testing.assert_array_equal(pre._scaler.mean_, mean_before)

    def test_imputer_median_computed_only_from_training_rows(self):
        X_train = pd.DataFrame({
            "feat_a": [1.0, 2.0, np.nan, 4.0, 5.0],  # median of [1,2,4,5] = 3.0
            RECOMMENDED_CONTEXT_FEATURE: [423, 423, 423, 564, 564],
        })
        pre = train_module.LogisticRegressionPreprocessor().fit(X_train)
        idx = pre.numeric_columns.index("feat_a")
        self.assertAlmostEqual(pre._imputer.statistics_[idx], 3.0)

    def test_scaler_mean_computed_only_from_training_rows(self):
        X_train = pd.DataFrame({
            "feat_a": [10.0, 20.0, 30.0],
            RECOMMENDED_CONTEXT_FEATURE: [423, 423, 423],
        })
        pre = train_module.LogisticRegressionPreprocessor().fit(X_train)
        idx = pre.numeric_columns.index("feat_a")
        self.assertAlmostEqual(pre._scaler.mean_[idx], 20.0)

    def test_transform_before_fit_raises(self):
        pre = train_module.LogisticRegressionPreprocessor()
        with self.assertRaises(RuntimeError):
            pre.transform(_synthetic_frame(3))

    def test_unseen_competition_category_gets_all_zero_onehot_not_an_error(self):
        X_train = pd.DataFrame({"feat_a": [1.0, 2.0], RECOMMENDED_CONTEXT_FEATURE: [423, 423]})
        pre = train_module.LogisticRegressionPreprocessor().fit(X_train)
        X_val = pd.DataFrame({"feat_a": [3.0], RECOMMENDED_CONTEXT_FEATURE: [999999]})
        encoded = pre.transform(X_val)
        onehot_part = encoded[:, len(pre.numeric_columns):]
        np.testing.assert_array_equal(onehot_part, np.zeros_like(onehot_part))


@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class TestCompetitionCodeEncoder(unittest.TestCase):
    """Fix 2: competition_id must be re-coded to a dense [0, n_categories)
    integer range before reaching HistGradientBoostingClassifier's native
    categorical handling -- the real IDs {200, 419, 423, 477, 499} are
    not contiguous and violate sklearn's documented contract. See
    docs/PHASE3_STEP3_DIAGNOSTIC_REPORT.md, design question 3.
    """

    def test_maps_real_ids_to_contiguous_zero_based_range(self):
        ids = pd.Series([499, 200, 423, 419, 477, 200, 423])
        encoder = train_module.CompetitionCodeEncoder().fit(ids)
        codes = encoder.transform(ids)
        self.assertEqual(set(codes.tolist()), {0, 1, 2, 3, 4})
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes < 5).all())

    def test_mapping_is_sorted_not_insertion_order(self):
        # Fitting on data seen in a different row order must produce the
        # identical mapping -- the mapping is a function of the *set* of
        # training categories, not the order rows happened to arrive in.
        ids_order_a = pd.Series([499, 200, 423, 419, 477])
        ids_order_b = pd.Series([200, 419, 423, 477, 499])
        enc_a = train_module.CompetitionCodeEncoder().fit(ids_order_a)
        enc_b = train_module.CompetitionCodeEncoder().fit(ids_order_b)
        self.assertEqual(enc_a.categories_, enc_b.categories_)
        self.assertEqual(enc_a._code_by_category, enc_b._code_by_category)

    def test_deterministic_across_repeated_fits_on_same_data(self):
        ids = pd.Series([200, 419, 423, 477, 499, 423, 200])
        enc1 = train_module.CompetitionCodeEncoder().fit(ids)
        enc2 = train_module.CompetitionCodeEncoder().fit(ids)
        np.testing.assert_array_equal(enc1.transform(ids), enc2.transform(ids))
        self.assertEqual(enc1.categories_, enc2.categories_)

    def test_validation_transform_uses_training_mapping_not_its_own(self):
        # The whole point: fit ONLY on train, then transform() must reuse
        # those exact codes for validation data, even if validation's
        # category distribution/order differs completely.
        train_ids = pd.Series([200, 419, 423, 477, 499])
        encoder = train_module.CompetitionCodeEncoder().fit(train_ids)
        train_codes = encoder.transform(train_ids)

        val_ids = pd.Series([499, 499, 200])  # different order/frequency
        val_codes = encoder.transform(val_ids)

        # 499's code in val must match 499's code as learned from train.
        code_499_from_train = train_codes[list(train_ids).index(499)]
        self.assertTrue((val_codes[val_ids == 499] == code_499_from_train).all())

    def test_unseen_category_at_transform_time_maps_to_negative_one(self):
        train_ids = pd.Series([200, 419, 423])
        encoder = train_module.CompetitionCodeEncoder().fit(train_ids)
        val_ids = pd.Series([200, 999999])  # 999999 never seen in training
        codes = encoder.transform(val_ids)
        self.assertEqual(codes[1], -1)
        self.assertGreaterEqual(codes[0], 0)

    def test_unseen_category_code_never_collides_with_a_real_seen_category(self):
        # -1 is reserved and cannot equal any valid learned code (which
        # are always >= 0), so an unseen category can never be silently
        # mistaken for a specific real league.
        train_ids = pd.Series([200, 419, 423, 477, 499])
        encoder = train_module.CompetitionCodeEncoder().fit(train_ids)
        real_codes = set(encoder.transform(train_ids).tolist())
        self.assertNotIn(-1, real_codes)

    def test_transform_before_fit_raises(self):
        encoder = train_module.CompetitionCodeEncoder()
        with self.assertRaises(RuntimeError):
            encoder.transform(pd.Series([200, 419]))

    def test_hgb_pipeline_receives_valid_categorical_codes_not_raw_ids(self):
        # End-to-end: train_hist_gradient_boosting must internally re-code
        # competition_id before fitting -- verified here by fitting the
        # returned encoder against the same real-world-shaped IDs used
        # throughout this test class and confirming every code handed to
        # the model would fall in sklearn's required [0, n_categories) range.
        rng = np.random.default_rng(7)
        n = 100
        X_train = pd.DataFrame({
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            RECOMMENDED_CONTEXT_FEATURE: rng.choice([200, 419, 423, 477, 499], size=n),
        })
        y_train = pd.Series(rng.choice(list(CLASS_ORDER), size=n))
        X_val = pd.DataFrame({
            "feat_a": rng.normal(size=20),
            "feat_b": rng.normal(size=20),
            RECOMMENDED_CONTEXT_FEATURE: rng.choice([200, 419, 423, 477, 499], size=20),
        })

        _, encoder, P_val = train_module.train_hist_gradient_boosting(X_train, y_train, X_val)
        self.assertIsNotNone(encoder)
        train_codes = encoder.transform(X_train[RECOMMENDED_CONTEXT_FEATURE])
        val_codes = encoder.transform(X_val[RECOMMENDED_CONTEXT_FEATURE])
        self.assertTrue((train_codes >= 0).all())  # every training id was seen at fit time
        self.assertTrue(train_codes.max() < 5)
        self.assertTrue((val_codes >= -1).all())
        self.assertTrue(val_codes.max() < 5)
        validate_probabilities(P_val)


@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class TestReorderProba(unittest.TestCase):
    """sklearn returns predict_proba columns in alphabetical class order
    (A, D, H), not our CLASS_ORDER (H, D, A) -- this is the single
    easiest place to introduce a silent, high-impact bug (every
    downstream probability would be mislabeled), so it gets its own
    focused tests independent of any real model fit.
    """

    def test_reorders_alphabetical_sklearn_output_to_class_order(self):
        class FakeModel:
            classes_ = np.array(["A", "D", "H"])

            def predict_proba(self, X):
                return np.array([[0.2, 0.3, 0.5]])  # P(A)=.2, P(D)=.3, P(H)=.5

        P = train_module._reorder_proba(FakeModel(), X_encoded=None)
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])
        np.testing.assert_allclose(P, [[0.5, 0.3, 0.2]])

    def test_missing_class_in_model_raises_rather_than_silently_misaligning(self):
        class FakeModel:
            classes_ = np.array(["D", "H"])  # no A ever seen in training

            def predict_proba(self, X):
                return np.array([[0.4, 0.6]])

        with self.assertRaises(RuntimeError):
            train_module._reorder_proba(FakeModel(), X_encoded=None)


@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class TestTrainingProducesValidProbabilities(unittest.TestCase):
    def test_logistic_regression_predict_proba_is_valid(self):
        X_train, y_train = _synthetic_dataset(200, seed=1)
        X_val, _ = _synthetic_dataset(50, seed=2)
        _, _, P_val = train_module.train_logistic_regression(X_train, y_train, X_val)
        validate_probabilities(P_val)
        self.assertEqual(P_val.shape, (50, 3))

    def test_hist_gradient_boosting_predict_proba_is_valid(self):
        X_train, y_train = _synthetic_dataset(200, seed=1)
        X_val, _ = _synthetic_dataset(50, seed=2)
        _, _, P_val = train_module.train_hist_gradient_boosting(X_train, y_train, X_val)
        validate_probabilities(P_val)
        self.assertEqual(P_val.shape, (50, 3))

    def test_hist_gradient_boosting_handles_missing_values_without_manual_imputation(self):
        X_train, y_train = _synthetic_dataset(200, seed=1)
        X_train.loc[X_train.index[:10], "feat_a"] = np.nan
        X_val, _ = _synthetic_dataset(50, seed=2)
        X_val.loc[X_val.index[:5], "feat_a"] = np.nan
        _, _, P_val = train_module.train_hist_gradient_boosting(X_train, y_train, X_val)
        validate_probabilities(P_val)

    def test_logistic_regression_handles_missing_values_via_fold_only_imputation(self):
        X_train, y_train = _synthetic_dataset(200, seed=1)
        X_train.loc[X_train.index[:10], "feat_a"] = np.nan
        X_val, _ = _synthetic_dataset(50, seed=2)
        X_val.loc[X_val.index[:5], "feat_a"] = np.nan
        _, _, P_val = train_module.train_logistic_regression(X_train, y_train, X_val)
        validate_probabilities(P_val)


@unittest.skipUnless(train_module is not None, _SKIP_REASON)
class TestWalkForwardAgainstRealDataset(unittest.TestCase):
    """Runs the 3 approved walk-forward folds against the real
    data/processed/features.db, if present in this environment. Skips
    (never fails) when the dataset isn't available -- the actual
    fold-by-fold numeric results belong in the Step 2 report, produced
    on the machine where both the real dataset and scikit-learn are
    available together.
    """

    @classmethod
    def setUpClass(cls):
        real_db = Path("data/processed/features.db")
        if not real_db.exists():
            raise unittest.SkipTest("data/processed/features.db not present in this environment")
        from models.data import load_supervised_dataset
        cls.dataset = load_supervised_dataset(real_db)

    def test_all_three_folds_produce_valid_probabilities_for_both_candidates(self):
        # Scoped to the approved 15-column feature set, matching what
        # run_experiments.py actually feeds each model -- the full
        # 264-column feature_rows table includes xG feature groups that
        # are structurally all-null in early folds (pre-xG-cutover) and
        # will crash HistGradientBoostingClassifier's binning step if
        # passed directly. See docs/PHASE3_STEP3_DIAGNOSTIC_REPORT.md.
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        from models.splits import iter_walk_forward_folds

        for fold, train_ds, val_ds in iter_walk_forward_folds(self.dataset):
            with self.subTest(fold=fold.name):
                train_X = train_ds.X[list(APPROVED_FEATURE_COLUMNS_V1)]
                val_X = val_ds.X[list(APPROVED_FEATURE_COLUMNS_V1)]

                _, _, P_lr = train_module.train_logistic_regression(train_X, train_ds.y, val_X)
                validate_probabilities(P_lr)
                self.assertEqual(len(P_lr), len(val_ds))

                _, _, P_hgb = train_module.train_hist_gradient_boosting(train_X, train_ds.y, val_X)
                validate_probabilities(P_hgb)
                self.assertEqual(len(P_hgb), len(val_ds))


if __name__ == "__main__":
    unittest.main()
