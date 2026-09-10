"""Gate 5 — Deterministic training.

Gate 5 requirement (docs/PHASE5_V2_MODEL_SPEC_DRAFT.md):
    "Identical inputs produce identical model and predictions."
    Verification: train twice, compare predictions byte-for-byte.
    Pass: byte-identical outputs. Stop: any nondeterminism.

Acceptance criterion is **byte identity** (`ndarray.tobytes()`), not
floating-point closeness. Nothing here uses `assertAlmostEqual` or
`allclose` on the determinism comparisons.

Three determinism surfaces are examined separately, because a failure in
any one would have a different cause:
  1. preprocessing  -- fitted statistics and encoded matrices
  2. estimator fit  -- learned coefficients and intercepts
  3. prediction     -- probability arrays returned to callers

FINAL-TEST SAFETY: 2025/26 is never selected. The training partition is
obtained via `split_by_seasons(dataset, FINAL_TRAIN_SEASONS, ())` -- an
empty eval-season tuple -- so no 2025/26 row is ever loaded into a
partition here. `final_split` is deliberately NOT imported (asserted).
No metric is computed anywhere in this module: Gate 5 compares bytes,
it does not evaluate.

Trains the candidate only to prove reproducibility. Creates no
production artifact, bumps no version, tunes nothing, calibrates nothing.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models.candidate_contract import (
    CANDIDATE_FEATURE_COLUMNS,
    select_candidate_features,
    verify_feature_contract,
)
from models.config import FINAL_TRAIN_SEASONS

try:
    from models import train as train_module
    _IMPORT_ERROR = None
except ImportError as exc:
    train_module = None
    _IMPORT_ERROR = str(exc)

_SKLEARN_SKIP = f"scikit-learn not available in this environment: {_IMPORT_ERROR}"
_REAL_DB = Path("data/processed/features.db")
_DATA_SKIP = "data/processed/features.db not present in this environment"

# Frozen estimator configuration (spec §4). Asserted against train.py.
FROZEN_ESTIMATOR_KWARGS = {"max_iter": 2000, "C": 1.0, "random_state": 0}


def _load_frozen_training_partition():
    """The exact FINAL_TRAIN_SEASONS partition, obtained without ever
    selecting a 2025/26 row (empty eval-season tuple)."""
    from models.data import load_supervised_dataset
    from models.splits import split_by_seasons

    dataset = load_supervised_dataset(_REAL_DB)
    train_ds, empty = split_by_seasons(dataset, FINAL_TRAIN_SEASONS, ())
    assert len(empty) == 0, "eval partition must be empty -- 2025/26 must not be selected"
    return train_ds


def _byte_signature(arr: np.ndarray) -> tuple:
    """Strict byte-level identity signature: dtype, shape, C-contiguity
    and the raw buffer. Two arrays with the same signature are
    bit-identical, not merely numerically close."""
    a = np.ascontiguousarray(arr)
    return (str(a.dtype), a.shape, hashlib.sha256(a.tobytes()).hexdigest())


class TestGate5FrozenConfiguration(unittest.TestCase):
    """The determinism claim only means something if the configuration
    under test is the frozen one."""

    def test_contract_is_the_frozen_eighty_columns(self):
        r = verify_feature_contract()
        self.assertTrue(r["verified"])
        self.assertEqual(r["n_features"], 80)
        self.assertTrue(r["order_preserved"])
        self.assertTrue(r["omitted_v1_feature_absent"])

    def test_omitted_v1_feature_still_absent(self):
        self.assertNotIn("league_home_advantage_season", CANDIDATE_FEATURE_COLUMNS)

    def test_estimator_kwargs_in_train_py_are_frozen(self):
        tree = ast.parse(Path("src/models/train.py").read_text(encoding="utf-8"))
        kwargs = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LogisticRegression":
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        self.assertEqual(kwargs, FROZEN_ESTIMATOR_KWARGS)

    def test_random_state_is_zero(self):
        self.assertEqual(FROZEN_ESTIMATOR_KWARGS["random_state"], 0)

    def test_class_order_frozen(self):
        from models.baselines import CLASS_ORDER
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])

    def test_final_train_seasons_frozen(self):
        self.assertEqual(
            FINAL_TRAIN_SEASONS,
            ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"),
        )


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP)
@unittest.skipUnless(_REAL_DB.exists(), _DATA_SKIP)
class TestGate5DeterministicTrainingOnFrozenPartition(unittest.TestCase):
    """Train the candidate TWICE on the identical frozen training
    partition and compare all three determinism surfaces byte-for-byte."""

    @classmethod
    def setUpClass(cls):
        cls.train_ds = _load_frozen_training_partition()
        cls.X = select_candidate_features(cls.train_ds.X)
        cls.y = cls.train_ds.y
        # Prediction surface: the training rows themselves. This is a
        # byte-comparison only -- no metric is computed, so predicting
        # in-sample carries no evaluation meaning and keeps 2025/26 out.
        cls.X_pred = cls.X

    def _run(self):
        return train_module.train_logistic_regression(self.X, self.y, self.X_pred)

    def test_training_partition_is_the_expected_frozen_size(self):
        self.assertEqual(len(self.train_ds), 8983)
        self.assertEqual(self.X.shape[1], 80)
        self.assertEqual(list(self.X.columns), list(CANDIDATE_FEATURE_COLUMNS))

    def test_predictions_are_byte_identical_across_two_identical_runs(self):
        _, _, P_a = self._run()
        _, _, P_b = self._run()
        self.assertEqual(P_a.shape, P_b.shape, "prediction shape differed")
        self.assertEqual(
            _byte_signature(P_a), _byte_signature(P_b),
            "predictions from two identical training runs are NOT byte-identical",
        )
        # Strictest possible restatement: raw buffers equal.
        self.assertEqual(P_a.tobytes(), P_b.tobytes())

    def test_preprocessing_statistics_are_byte_identical(self):
        _, pre_a, _ = self._run()
        _, pre_b, _ = self._run()
        self.assertEqual(pre_a.numeric_columns, pre_b.numeric_columns)
        self.assertEqual(pre_a.competition_categories, pre_b.competition_categories)
        for name in ("statistics_",):
            self.assertEqual(
                _byte_signature(getattr(pre_a._imputer, name)),
                _byte_signature(getattr(pre_b._imputer, name)),
                f"imputer.{name} not byte-identical",
            )
        for name in ("mean_", "scale_", "var_"):
            self.assertEqual(
                _byte_signature(getattr(pre_a._scaler, name)),
                _byte_signature(getattr(pre_b._scaler, name)),
                f"scaler.{name} not byte-identical",
            )

    def test_encoded_design_matrix_is_byte_identical(self):
        _, pre_a, _ = self._run()
        _, pre_b, _ = self._run()
        self.assertEqual(
            _byte_signature(pre_a.transform(self.X)),
            _byte_signature(pre_b.transform(self.X)),
            "encoded design matrix not byte-identical",
        )

    def test_estimator_coefficients_are_byte_identical(self):
        model_a, _, _ = self._run()
        model_b, _, _ = self._run()
        self.assertEqual(_byte_signature(model_a.coef_), _byte_signature(model_b.coef_),
                         "coef_ not byte-identical")
        self.assertEqual(_byte_signature(model_a.intercept_), _byte_signature(model_b.intercept_),
                         "intercept_ not byte-identical")
        self.assertEqual(list(model_a.classes_), list(model_b.classes_))
        self.assertEqual(model_a.n_iter_.tolist(), model_b.n_iter_.tolist(),
                         "solver iteration counts differed -- possible nondeterminism")

    def test_three_consecutive_runs_all_byte_identical(self):
        sigs = []
        for _ in range(3):
            _, _, P = self._run()
            sigs.append(_byte_signature(P))
        self.assertEqual(len(set(sigs)), 1, f"runs produced {len(set(sigs))} distinct outputs")

    def test_prediction_shape_and_row_alignment(self):
        _, _, P = self._run()
        self.assertEqual(P.shape, (len(self.X_pred), 3))
        # fixture_id alignment: row i of P corresponds to row i of the
        # partition metadata, in the same order.
        self.assertEqual(len(P), len(self.train_ds.metadata))

    def test_fixture_id_alignment_is_stable_across_runs(self):
        fids = list(self.train_ds.metadata["fixture_id"])
        _, _, P_a = self._run()
        _, _, P_b = self._run()
        frame_a = pd.DataFrame(P_a, index=fids)
        frame_b = pd.DataFrame(P_b, index=fids)
        pd.testing.assert_frame_equal(frame_a, frame_b, check_exact=True)


@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP)
class TestGate5PositiveControls(unittest.TestCase):
    """Determinism assertions would be vacuous if the pipeline could not
    produce different output at all. These prove it can."""

    def _synthetic(self, n, seed, offset=0.0):
        rng = np.random.default_rng(seed)
        data = {c: rng.normal(size=n) + offset for c in CANDIDATE_FEATURE_COLUMNS if c != "competition_id"}
        data["competition_id"] = rng.choice([200, 419, 423, 477, 499], size=n)
        X = pd.DataFrame(data)[list(CANDIDATE_FEATURE_COLUMNS)]
        y = pd.Series(rng.choice(["H", "D", "A"], size=n))
        return X, y

    def test_changing_training_data_changes_predictions(self):
        X1, y1 = self._synthetic(300, seed=1)
        X2, y2 = self._synthetic(300, seed=2, offset=3.0)
        X_pred, _ = self._synthetic(50, seed=99)
        _, _, P1 = train_module.train_logistic_regression(X1, y1, X_pred)
        _, _, P2 = train_module.train_logistic_regression(X2, y2, X_pred)
        self.assertNotEqual(
            _byte_signature(P1), _byte_signature(P2),
            "different training data produced byte-identical output -- determinism test is vacuous",
        )

    def test_changing_prediction_input_changes_predictions(self):
        X, y = self._synthetic(300, seed=3)
        Xa, _ = self._synthetic(40, seed=4)
        Xb, _ = self._synthetic(40, seed=5, offset=2.0)
        _, _, Pa = train_module.train_logistic_regression(X, y, Xa)
        _, _, Pb = train_module.train_logistic_regression(X, y, Xb)
        self.assertNotEqual(_byte_signature(Pa), _byte_signature(Pb))

    def test_byte_signature_detects_a_single_bit_difference(self):
        # Guards the comparison helper itself.
        a = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        b = a.copy()
        b[2] = np.nextafter(b[2], 1.0)  # smallest representable change
        self.assertNotEqual(_byte_signature(a), _byte_signature(b))
        self.assertEqual(_byte_signature(a), _byte_signature(a.copy()))

    def test_byte_signature_distinguishes_dtype_and_shape(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        self.assertNotEqual(_byte_signature(a), _byte_signature(a.astype(np.float32)))
        self.assertNotEqual(_byte_signature(a), _byte_signature(a.reshape(2, 1)))


class TestGate5GovernanceBoundary(unittest.TestCase):
    """Gate 5 verifies reproducibility only. It must not touch V1,
    2025/26, versions, artifacts, tuning, or calibration."""

    @staticmethod
    def _this_tree():
        return ast.parse(Path(__file__).read_text(encoding="utf-8"))

    def _imported(self):
        names = set()
        for n in ast.walk(self._this_tree()):
            if isinstance(n, ast.Import):
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                names.add(n.module or "")
                names.update(a.name for a in n.names)
        return names

    def test_final_split_is_never_imported_or_called(self):
        imported = self._imported()
        self.assertNotIn("final_split", imported)
        self.assertNotIn("FINAL_TEST_SEASONS", imported)
        calls = [
            n for n in ast.walk(self._this_tree())
            if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == "final_split"
        ]
        self.assertEqual(calls, [])

    def test_no_metric_is_computed_in_this_gate(self):
        # Gate 5 compares bytes; it must not evaluate.
        imported = self._imported()
        for forbidden in ("evaluate", "log_loss", "brier_score", "accuracy"):
            self.assertNotIn(forbidden, imported)

    def test_no_calibration_or_tuning_introduced(self):
        # AST-based: a substring search over this file would match the
        # forbidden names inside this very assertion.
        imported = self._imported()
        names = {n.id for n in ast.walk(self._this_tree()) if isinstance(n, ast.Name)}
        for forbidden in ("CalibratedClassifierCV", "GridSearchCV", "RandomizedSearchCV", "C_GRID"):
            self.assertNotIn(forbidden, imported)
            self.assertNotIn(forbidden, names)

    def test_no_approximate_equality_used_for_determinism(self):
        # Gate 5 requires byte identity, not tolerance-based closeness.
        tree = self._this_tree()
        loose = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                name = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
                if name in {"assertAlmostEqual", "allclose", "assert_allclose"}:
                    loose.append(name)
        self.assertEqual(loose, [], f"determinism must not be asserted with tolerance: {loose}")

    def test_model_version_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_no_production_artifact_written(self):
        # AST-based for the same self-reference reason: assert this module
        # makes no write/serialisation CALL and imports no serialiser.
        imported = self._imported()
        for forbidden in ("joblib", "pickle", "cloudpickle"):
            self.assertNotIn(forbidden, imported)
        write_calls = [
            n for n in ast.walk(self._this_tree())
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in {"write_text", "write_bytes", "to_csv", "to_json", "dump", "savez", "save"}
        ]
        self.assertEqual(write_calls, [], "Gate 5 must not write any artifact")

    def test_no_gate5_artifact_created(self):
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate5", p.name.lower())

    def test_no_v2_module_or_artifact_created(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_v1_and_locked_artifacts_unchanged(self):
        expected = {
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
            "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
            "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
            "src/models/ablation.py": "9bf8bf4a1f332f8bb47d640a72384ed7",
            "src/models/baselines.py": "42e64e3a0ba8c4bf8cf264f80cdcd208",
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
            "data/audit/phase5a_dropped_feature_comparison.json": "3ceb7090f9e19c106d88eaa7a8848818",
        }
        for rel, exp in expected.items():
            actual = hashlib.md5(Path(rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"locked artifact modified: {rel}")


if __name__ == "__main__":
    unittest.main()
