"""Gate 4 — Preprocessing reproducibility.

Gate 4 requirement (docs/PHASE5_V2_MODEL_SPEC_DRAFT.md):
    "Imputation/scaling/encoding fit on training partition only and
    reproducible." Verification: code audit + fit-isolation tests
    (fold-swap style). Pass: statistics depend only on training rows;
    identical across repeated fits. Stop: any statistic influenced by
    validation/test data.

Two halves:
  - Static code audit (runs everywhere, no scikit-learn needed): proves
    structurally that fitting is confined to fit(X_train), transform()
    never refits, and no forbidden operation exists.
  - Empirical fold-swap and reproducibility tests (require scikit-learn,
    since V1's preprocessor is built on SimpleImputer/StandardScaler):
    these fit the REAL V1 preprocessor and prove statistics are
    bit-identical when non-training rows change arbitrarily.

Trains no model, tunes nothing, calibrates nothing. Never uses 2025/26
as a model evaluation or selection input.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models.candidate_contract import CANDIDATE_FEATURE_COLUMNS

try:
    from models import train as train_module
    _IMPORT_ERROR = None
except ImportError as exc:
    train_module = None
    _IMPORT_ERROR = str(exc)

_SKLEARN_SKIP = f"scikit-learn not available in this environment: {_IMPORT_ERROR}"
_TRAIN_SRC = Path("src/models/train.py")

KNOWN_LEAGUES = [200, 419, 423, 477, 499]


def _preproc_class_source() -> str:
    src = _TRAIN_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "LogisticRegressionPreprocessor")
    return ast.get_source_segment(src, cls)


def _method_node(name: str) -> ast.FunctionDef:
    src = _TRAIN_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "LogisticRegressionPreprocessor")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls_in(node) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            out.append(getattr(n.func, "attr", None) or getattr(n.func, "id", None))
    return [c for c in out if c]


# =====================================================================
# Static code audit — runs without scikit-learn
# =====================================================================
class TestGate4StaticCodeAudit(unittest.TestCase):
    def test_fit_takes_only_a_training_frame(self):
        fit = _method_node("fit")
        self.assertEqual([a.arg for a in fit.args.args], ["self", "X_train"])

    def test_fit_never_references_validation_or_test_frames(self):
        fit = _method_node("fit")
        names = {n.id for n in ast.walk(fit) if isinstance(n, ast.Name)}
        for forbidden in ("X_val", "X_test", "X_validation"):
            self.assertNotIn(forbidden, names)

    def test_all_fitting_happens_inside_fit_only(self):
        fit_calls = _calls_in(_method_node("fit"))
        self.assertIn("fit_transform", fit_calls)  # imputer
        self.assertIn("fit", fit_calls)            # scaler

    def test_transform_never_fits(self):
        tf_calls = _calls_in(_method_node("transform"))
        for forbidden in ("fit", "fit_transform", "partial_fit"):
            self.assertNotIn(forbidden, tf_calls, f"transform() must not call {forbidden}")

    def test_transform_only_applies_stored_statistics(self):
        tf = ast.get_source_segment(_TRAIN_SRC.read_text(encoding="utf-8"), _method_node("transform"))
        self.assertIn("self._imputer.transform(", tf)
        self.assertIn("self._scaler.transform(", tf)
        self.assertIn("self.competition_categories", tf)

    def test_transform_does_not_recompute_categories(self):
        tf = ast.get_source_segment(_TRAIN_SRC.read_text(encoding="utf-8"), _method_node("transform"))
        self.assertNotIn("unique()", tf)
        self.assertNotIn("sorted(", tf)

    def test_no_zero_fill_anywhere_in_preprocessor(self):
        src = _preproc_class_source()
        for forbidden in ("fillna", "nan_to_num", "replace(np.nan"):
            self.assertNotIn(forbidden, src)

    def test_no_blanket_dropna_on_the_feature_frame(self):
        # The ONLY dropna present is on a single Series
        # (X_train[competition_id]) to enumerate categories -- it drops no
        # rows and the imputer still receives the full X_train. A blanket
        # frame-level dropna() would be a spec violation; this is not one.
        src = _TRAIN_SRC.read_text(encoding="utf-8")
        cls_src = _preproc_class_source()
        tree = ast.parse(src)
        cls = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == "LogisticRegressionPreprocessor")
        receivers = []
        for n in ast.walk(cls):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "dropna":
                receivers.append(ast.get_source_segment(src, n.func.value))
        self.assertEqual(receivers, ["X_train[RECOMMENDED_CONTEXT_FEATURE]"])
        # And the imputer is fit on the unfiltered training frame.
        self.assertIn("fit_transform(X_train[self.numeric_columns])", cls_src)

    def test_no_cross_partition_concatenation(self):
        src = _preproc_class_source()
        for forbidden in ("concat", "concatenate", "merge(", "join("):
            self.assertNotIn(forbidden, src)

    def test_no_module_level_or_global_statistic_state(self):
        src = _preproc_class_source()
        self.assertNotIn("global ", src)

    def test_median_strategy_and_standard_scaler_are_the_frozen_choices(self):
        src = _preproc_class_source()
        self.assertIn('SimpleImputer(strategy="median")', src)
        self.assertIn("StandardScaler()", src)

    def test_transform_before_fit_is_guarded(self):
        tf = ast.get_source_segment(_TRAIN_SRC.read_text(encoding="utf-8"), _method_node("transform"))
        self.assertIn("RuntimeError", tf)
        self.assertIn("_fitted", tf)


# =====================================================================
# Empirical fold-swap + reproducibility — require scikit-learn
# =====================================================================
def _frame(n, seed, league_pool=None, offset=0.0, scale=1.0, nan_rows=0):
    """Synthetic frame with the exact contract columns."""
    rng = np.random.default_rng(seed)
    pool = league_pool or KNOWN_LEAGUES
    data = {}
    for c in CANDIDATE_FEATURE_COLUMNS:
        if c == "competition_id":
            continue
        vals = rng.normal(size=n) * scale + offset
        data[c] = vals
    data["competition_id"] = rng.choice(pool, size=n)
    df = pd.DataFrame(data)[list(CANDIDATE_FEATURE_COLUMNS)]
    if nan_rows:
        num_cols = [c for c in CANDIDATE_FEATURE_COLUMNS if c != "competition_id"]
        df.loc[df.index[:nan_rows], num_cols[0]] = np.nan
    return df


def _stats(p):
    """Snapshot of every fitted statistic."""
    return {
        "numeric_columns": list(p.numeric_columns),
        "competition_categories": list(p.competition_categories),
        "imputer_statistics": p._imputer.statistics_.copy(),
        "scaler_mean": p._scaler.mean_.copy(),
        "scaler_scale": p._scaler.scale_.copy(),
        "scaler_var": p._scaler.var_.copy(),
    }


def _assert_stats_identical(case, a, b, msg):
    case.assertEqual(a["numeric_columns"], b["numeric_columns"], msg)
    case.assertEqual(a["competition_categories"], b["competition_categories"], msg)
    np.testing.assert_array_equal(a["imputer_statistics"], b["imputer_statistics"], err_msg=msg)
    np.testing.assert_array_equal(a["scaler_mean"], b["scaler_mean"], err_msg=msg)
    np.testing.assert_array_equal(a["scaler_scale"], b["scaler_scale"], err_msg=msg)
    np.testing.assert_array_equal(a["scaler_var"], b["scaler_var"], err_msg=msg)


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP)
class TestGate4FoldSwapIsolation(unittest.TestCase):
    """Fold-swap: hold the TRAINING partition fixed, vary the
    validation/test partition materially, and prove every fitted
    statistic is bit-identical."""

    def setUp(self):
        self.X_train = _frame(400, seed=1, nan_rows=20)

    def _fit(self):
        return train_module.LogisticRegressionPreprocessor().fit(self.X_train)

    def test_statistics_identical_when_validation_partition_is_swapped(self):
        p1 = self._fit()
        before = _stats(p1)
        # Three materially different validation sets: shifted mean,
        # inflated variance, and heavy missingness.
        for label, X_val in (
            ("shifted", _frame(200, seed=2, offset=1000.0)),
            ("inflated", _frame(200, seed=3, scale=5000.0)),
            ("missing-heavy", _frame(200, seed=4, nan_rows=190)),
        ):
            p1.transform(X_val)
            _assert_stats_identical(self, before, _stats(p1),
                                    f"validation partition '{label}' altered fitted statistics")

    def test_statistics_identical_across_two_preprocessors_seeing_different_validation(self):
        pa = self._fit()
        pb = self._fit()
        pa.transform(_frame(150, seed=10, offset=-5000.0))
        pb.transform(_frame(150, seed=11, scale=1e6))
        _assert_stats_identical(self, _stats(pa), _stats(pb),
                                "different validation partitions produced different statistics")

    def test_extreme_validation_values_do_not_shift_imputer_medians(self):
        p = self._fit()
        med_before = p._imputer.statistics_.copy()
        wild = _frame(300, seed=12, offset=1e9, nan_rows=150)
        p.transform(wild)
        np.testing.assert_array_equal(med_before, p._imputer.statistics_)

    def test_unseen_validation_league_does_not_extend_learned_categories(self):
        p = self._fit()
        cats_before = list(p.competition_categories)
        X_val = _frame(50, seed=13, league_pool=[999_001, 999_002])
        out = p.transform(X_val)
        self.assertEqual(list(p.competition_categories), cats_before)
        # Unseen leagues must produce an all-zero one-hot block, not a new column.
        onehot = out[:, len(p.numeric_columns):]
        self.assertEqual(onehot.shape[1], len(cats_before))
        np.testing.assert_array_equal(onehot, np.zeros_like(onehot))

    def test_transform_output_width_is_fixed_by_training_categories(self):
        p = self._fit()
        expected_width = len(p.numeric_columns) + len(p.competition_categories)
        for X_val in (_frame(20, seed=14, league_pool=[200]),
                      _frame(20, seed=15, league_pool=KNOWN_LEAGUES + [777])):
            self.assertEqual(p.transform(X_val).shape[1], expected_width)

    def test_training_rows_alone_determine_statistics(self):
        # Positive control: changing the TRAINING data MUST change stats,
        # otherwise the isolation tests above would be vacuous.
        p1 = train_module.LogisticRegressionPreprocessor().fit(self.X_train)
        p2 = train_module.LogisticRegressionPreprocessor().fit(_frame(400, seed=99, offset=50.0))
        self.assertFalse(
            np.array_equal(p1._scaler.mean_, p2._scaler.mean_),
            "different training data produced identical statistics -- isolation tests would be vacuous",
        )


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP)
class TestGate4Reproducibility(unittest.TestCase):
    """Repeated fits on identical training data must be bit-identical."""

    def test_repeated_fits_on_identical_data_are_bit_identical(self):
        X = _frame(500, seed=7, nan_rows=30)
        runs = [_stats(train_module.LogisticRegressionPreprocessor().fit(X)) for _ in range(5)]
        for i, r in enumerate(runs[1:], start=2):
            _assert_stats_identical(self, runs[0], r, f"fit #{i} differed from fit #1")

    def test_repeated_transform_output_is_bit_identical(self):
        X = _frame(300, seed=8, nan_rows=15)
        X_val = _frame(120, seed=9, nan_rows=10)
        p = train_module.LogisticRegressionPreprocessor().fit(X)
        a, b = p.transform(X_val), p.transform(X_val)
        np.testing.assert_array_equal(a, b)

    def test_row_order_of_training_data_does_not_change_statistics(self):
        X = _frame(400, seed=21, nan_rows=25)
        shuffled = X.iloc[::-1].reset_index(drop=True)
        pa = train_module.LogisticRegressionPreprocessor().fit(X)
        pb = train_module.LogisticRegressionPreprocessor().fit(shuffled)
        self.assertEqual(pa.competition_categories, pb.competition_categories)
        np.testing.assert_allclose(pa._imputer.statistics_, pb._imputer.statistics_)
        np.testing.assert_allclose(pa._scaler.mean_, pb._scaler.mean_)

    def test_categories_are_sorted_hence_order_independent(self):
        a = _frame(100, seed=31, league_pool=[499, 200, 423])
        b = a.copy()
        b["competition_id"] = list(a["competition_id"])[::-1]
        pa = train_module.LogisticRegressionPreprocessor().fit(a)
        pb = train_module.LogisticRegressionPreprocessor().fit(b)
        self.assertEqual(pa.competition_categories, sorted(pa.competition_categories))
        self.assertEqual(pa.competition_categories, pb.competition_categories)

    def test_hand_computed_median_and_mean(self):
        # Anchors the semantics: median imputation, then standardization.
        cols = list(CANDIDATE_FEATURE_COLUMNS)
        n = 5
        data = {c: [1.0, 2.0, 3.0, 4.0, 5.0] for c in cols if c != "competition_id"}
        data["competition_id"] = [200, 200, 419, 419, 423]
        X = pd.DataFrame(data)[cols]
        target = [c for c in cols if c != "competition_id"][0]
        X.loc[X.index[0], target] = np.nan          # median of [2,3,4,5] = 3.5
        p = train_module.LogisticRegressionPreprocessor().fit(X)
        idx = p.numeric_columns.index(target)
        self.assertAlmostEqual(p._imputer.statistics_[idx], 3.5)
        # After imputation the column is [3.5,2,3,4,5] -> mean 3.5
        self.assertAlmostEqual(p._scaler.mean_[idx], 3.5)
        self.assertEqual(p.competition_categories, [200, 419, 423])


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP)
class TestGate4RealFoldSwapOnAuthoritativeData(unittest.TestCase):
    """Fold-swap using the real walk-forward partitions. 2025/26 is not
    used here at all -- only folds 1-3."""

    @classmethod
    def setUpClass(cls):
        db = Path("data/processed/features.db")
        if not db.exists():
            raise unittest.SkipTest("features.db not present in this environment")
        from models.data import load_supervised_dataset
        from models.splits import iter_walk_forward_folds
        cls.dataset = load_supervised_dataset(db)
        cls.folds = list(iter_walk_forward_folds(cls.dataset))

    def test_swapping_validation_fold_leaves_statistics_identical(self):
        from models.candidate_contract import select_candidate_features
        _, train_ds, val_ds = self.folds[0]
        X_train = select_candidate_features(train_ds.X)
        p = train_module.LogisticRegressionPreprocessor().fit(X_train)
        before = _stats(p)
        # Transform every other fold's validation partition through it.
        for _, _, other_val in self.folds:
            p.transform(select_candidate_features(other_val.X))
        _assert_stats_identical(self, before, _stats(p),
                               "real validation partitions altered fitted statistics")

    def test_statistics_reproducible_on_real_training_partition(self):
        from models.candidate_contract import select_candidate_features
        _, train_ds, _ = self.folds[2]
        X_train = select_candidate_features(train_ds.X)
        a = _stats(train_module.LogisticRegressionPreprocessor().fit(X_train))
        b = _stats(train_module.LogisticRegressionPreprocessor().fit(X_train))
        _assert_stats_identical(self, a, b, "repeated fit on real data differed")


class TestGate4NoSideEffects(unittest.TestCase):
    def test_no_model_trained_in_this_gate(self):
        this = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(this)
        bad = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in {"predict", "predict_proba"}]
        self.assertEqual(bad, [])
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        self.assertNotIn("LogisticRegression", imported)

    def test_no_final_split_reachable_from_this_gate(self):
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for forbidden in ("final_split", "FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, imported)

    def test_model_version_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_no_gate4_artifact_created(self):
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate4", p.name.lower())

    def test_v1_preprocessing_and_locked_artifacts_unchanged(self):
        expected = {
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
            "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
            "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
            "src/models/ablation.py": "9bf8bf4a1f332f8bb47d640a72384ed7",
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
        }
        for rel, exp in expected.items():
            actual = hashlib.md5(Path(rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"locked artifact modified: {rel}")


if __name__ == "__main__":
    unittest.main()
