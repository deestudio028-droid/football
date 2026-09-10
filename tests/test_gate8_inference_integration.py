"""Gate 8 — Inference integration.

Gate 8 requirement (docs/PHASE5_V2_MODEL_SPEC_DRAFT.md):
    "Candidate callable through the intended prediction path."
    Verification: integration test on a held-out-shaped batch.
    Pass: correct shape, ordering, `fixture_id` alignment, valid
    probabilities. Stop: interface mismatch or silent column substitution.

The intended prediction path is `candidate_contract.predict_candidate`,
added under explicit authorization as a minimal additive inference
boundary enforcing spec §7.

HELD-OUT-SHAPED BATCH: synthetic contract-shaped rows only, per
authorization. No 2025/26 or any final-test-season row is used. The
synthetic batch exercises the real interface; it is never used for
fitting, tuning, calibration, or metric selection, and no metric is
computed anywhere in this module.

Fitting a model is required to have something to predict WITH, so the
integration tests are scikit-learn-gated. Contract/interface tests that
need no fitted model run everywhere.

BATCH SIZE: §7 describes the input as a frame "for fixtures" and §10's
pre-inference checks are only defined for a batch with at least one row.
One row is therefore the documented minimum; zero-row input is outside
the contract (see
`test_empty_batch_is_outside_the_documented_contract`).
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models.candidate_contract import (
    CANDIDATE_FEATURE_COLUMNS,
    CLASS_ORDER_NAMES,
    CandidatePrediction,
    InferenceContractViolation,
    predict_candidate,
)
from models.config import X_EXCLUDED_COLUMNS
from models.evaluate import MetricInputError, validate_probabilities

try:
    from models import train as train_module
    _IMPORT_ERROR = None
except ImportError as exc:
    train_module = None
    _IMPORT_ERROR = str(exc)

_SKLEARN_SKIP = f"scikit-learn not available in this environment: {_IMPORT_ERROR}"
KNOWN_LEAGUES = [200, 419, 423, 477, 499]


# ---------------------------------------------------------------------
# Synthetic held-out-shaped batch builders
# ---------------------------------------------------------------------
def _synthetic_batch(n: int, seed: int = 0, with_fixture_id: bool = False) -> pd.DataFrame:
    """Contract-shaped synthetic rows. Never real match data."""
    rng = np.random.default_rng(seed)
    data = {}
    for c in CANDIDATE_FEATURE_COLUMNS:
        if c == "competition_id":
            continue
        data[c] = rng.normal(size=n)
    data["competition_id"] = rng.choice(KNOWN_LEAGUES, size=n)
    df = pd.DataFrame(data)[list(CANDIDATE_FEATURE_COLUMNS)]
    if with_fixture_id:
        df["fixture_id"] = [900_000 + i for i in range(n)]
    return df


def _synthetic_labels(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed + 500)
    return pd.Series(rng.choice(["H", "D", "A"], size=n))


def _fixture_ids(n: int, start: int = 900_000):
    return [start + i for i in range(n)]


# =====================================================================
# Interface contract tests that need no fitted model
# =====================================================================
class TestGate8InterfaceSurface(unittest.TestCase):
    def test_prediction_path_is_exported_and_callable(self):
        self.assertTrue(callable(predict_candidate))
        self.assertEqual(predict_candidate.__module__, "models.candidate_contract")

    def test_class_order_names_match_frozen_class_order(self):
        from models.baselines import CLASS_ORDER
        self.assertEqual(list(CLASS_ORDER_NAMES), CLASS_ORDER)
        self.assertEqual(CLASS_ORDER_NAMES, ("H", "D", "A"))

    def test_fixture_id_is_not_a_contract_feature(self):
        self.assertNotIn("fixture_id", CANDIDATE_FEATURE_COLUMNS)
        self.assertIn("fixture_id", X_EXCLUDED_COLUMNS)

    def test_prediction_path_never_fits(self):
        import inspect
        src = inspect.getsource(predict_candidate)
        tree = ast.parse(src)
        fit_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in {"fit", "fit_transform", "partial_fit"}
        ]
        self.assertEqual(fit_calls, [], "the inference path must never fit")

    def test_prediction_path_reuses_frozen_utilities(self):
        import inspect
        src = inspect.getsource(predict_candidate)
        # Reuses V1's reordering and the existing validator rather than
        # reimplementing either.
        self.assertIn("_reorder_proba", src)
        self.assertIn("validate_probabilities", src)
        self.assertIn("select_candidate_features", src)

    def test_candidate_prediction_result_is_immutable(self):
        pred = CandidatePrediction(
            fixture_ids=(1, 2),
            probabilities=np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]),
            class_order=CLASS_ORDER_NAMES,
            n_rows=2,
        )
        with self.assertRaises(Exception):
            pred.n_rows = 99  # frozen dataclass

    def test_as_frame_columns_follow_class_order_positionally(self):
        pred = CandidatePrediction(
            fixture_ids=(11, 22),
            probabilities=np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]),
            class_order=CLASS_ORDER_NAMES,
            n_rows=2,
        )
        frame = pred.as_frame()
        self.assertEqual(list(frame.columns), ["fixture_id", "p_home", "p_draw", "p_away"])
        self.assertEqual(list(frame["fixture_id"]), [11, 22])
        self.assertAlmostEqual(frame["p_home"].iloc[0], 0.5)
        self.assertAlmostEqual(frame["p_draw"].iloc[0], 0.3)
        self.assertAlmostEqual(frame["p_away"].iloc[0], 0.2)


# =====================================================================
# Integration tests on a synthetic held-out-shaped batch
# =====================================================================
@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP)
class TestGate8InferenceIntegration(unittest.TestCase):
    """Fits once on synthetic rows purely to obtain a model/preprocessor
    to exercise the interface. No metric is computed; nothing is tuned."""

    @classmethod
    def setUpClass(cls):
        X_fit = _synthetic_batch(300, seed=1)
        y_fit = _synthetic_labels(300, seed=1)
        # V1's frozen training path, unmodified.
        cls.model, cls.preprocessor, _ = train_module.train_logistic_regression(
            X_fit, y_fit, X_fit
        )

    def _predict(self, n=40, seed=7, with_fixture_id=False):
        X = _synthetic_batch(n, seed=seed, with_fixture_id=with_fixture_id)
        fids = _fixture_ids(n)
        return X, fids, predict_candidate(self.model, self.preprocessor, X, fids)

    # ---- shape ----
    def test_output_shape_is_n_by_three(self):
        n = 40
        _, _, pred = self._predict(n=n)
        self.assertEqual(pred.probabilities.shape, (n, 3))
        self.assertEqual(pred.n_rows, n)

    def test_single_row_batch(self):
        _, _, pred = self._predict(n=1)
        self.assertEqual(pred.probabilities.shape, (1, 3))

    def test_large_batch(self):
        _, _, pred = self._predict(n=500, seed=11)
        self.assertEqual(pred.probabilities.shape, (500, 3))

    # ---- class order ----
    def test_class_order_is_exactly_h_d_a(self):
        _, _, pred = self._predict()
        self.assertEqual(pred.class_order, ("H", "D", "A"))

    def test_column_order_matches_the_estimator_classes_mapping(self):
        # The reordering must map sklearn's alphabetical classes_ (A,D,H)
        # onto CLASS_ORDER (H,D,A). Verify positionally against the
        # estimator's own raw output.
        X, fids, pred = self._predict(n=25, seed=13)
        design = X[list(CANDIDATE_FEATURE_COLUMNS)]
        raw = self.model.predict_proba(self.preprocessor.transform(design))
        class_to_col = {c: i for i, c in enumerate(self.model.classes_)}
        for i, cls in enumerate(pred.class_order):
            np.testing.assert_allclose(
                pred.probabilities[:, i], raw[:, class_to_col[cls]],
                err_msg=f"column {i} does not correspond to class {cls}",
            )

    # ---- fixture_id alignment ----
    def test_fixture_ids_returned_in_input_order(self):
        _, fids, pred = self._predict(n=30, seed=17)
        self.assertEqual(list(pred.fixture_ids), list(fids))

    def test_fixture_id_row_alignment_is_row_for_row(self):
        n = 30
        X = _synthetic_batch(n, seed=19)
        fids = _fixture_ids(n)
        pred = predict_candidate(self.model, self.preprocessor, X, fids)
        # Predicting each row individually must reproduce the same row.
        for i in range(0, n, 7):
            single = predict_candidate(
                self.model, self.preprocessor, X.iloc[[i]], [fids[i]]
            )
            self.assertEqual(single.fixture_ids, (fids[i],))
            np.testing.assert_allclose(single.probabilities[0], pred.probabilities[i])

    def test_row_permutation_permutes_predictions_correspondingly(self):
        n = 20
        X = _synthetic_batch(n, seed=23)
        fids = _fixture_ids(n)
        base = predict_candidate(self.model, self.preprocessor, X, fids)

        order = list(range(n))[::-1]
        Xr = X.iloc[order].reset_index(drop=True)
        fidr = [fids[i] for i in order]
        permuted = predict_candidate(self.model, self.preprocessor, Xr, fidr)

        self.assertEqual(list(permuted.fixture_ids), fidr)
        for new_pos, old_pos in enumerate(order):
            np.testing.assert_allclose(
                permuted.probabilities[new_pos], base.probabilities[old_pos]
            )

    def test_fixture_id_count_mismatch_fails_loudly(self):
        X = _synthetic_batch(10, seed=29)
        with self.assertRaises(InferenceContractViolation):
            predict_candidate(self.model, self.preprocessor, X, _fixture_ids(9))
        with self.assertRaises(InferenceContractViolation):
            predict_candidate(self.model, self.preprocessor, X, _fixture_ids(11))

    # ---- probability validity ----
    def test_returned_probabilities_pass_validation(self):
        _, _, pred = self._predict(n=120, seed=31)
        validate_probabilities(pred.probabilities)

    def test_returned_probabilities_are_finite_nonnegative_and_sum_to_one(self):
        _, _, pred = self._predict(n=120, seed=37)
        P = pred.probabilities
        self.assertTrue(np.isfinite(P).all())
        self.assertTrue((P >= 0).all())
        np.testing.assert_allclose(P.sum(axis=1), np.ones(len(P)), atol=1e-6)

    # ---- missing column / substitution ----
    def test_missing_contract_column_fails_loudly(self):
        X = _synthetic_batch(10, seed=41).drop(columns=["strength_diff"])
        with self.assertRaises(RuntimeError) as ctx:
            predict_candidate(self.model, self.preprocessor, X, _fixture_ids(10))
        self.assertIn("strength_diff", str(ctx.exception))

    def test_several_missing_contract_columns_fail_loudly(self):
        X = _synthetic_batch(10, seed=43).drop(
            columns=["home_points_last5", "away_points_last5"]
        )
        with self.assertRaises(RuntimeError):
            predict_candidate(self.model, self.preprocessor, X, _fixture_ids(10))

    def test_no_silent_substitution_when_extra_columns_present(self):
        # Extra columns -- including fixture_id and the omitted V1 feature
        # -- must be ignored for modelling, never substituted in.
        n = 25
        X_plain = _synthetic_batch(n, seed=47)
        X_extra = X_plain.copy()
        X_extra["fixture_id"] = _fixture_ids(n)
        X_extra["league_home_advantage_season"] = 1.234
        X_extra["totally_unrelated_column"] = 9.99

        a = predict_candidate(self.model, self.preprocessor, X_plain, _fixture_ids(n))
        b = predict_candidate(self.model, self.preprocessor, X_extra, _fixture_ids(n))
        np.testing.assert_array_equal(a.probabilities, b.probabilities)

    def test_fixture_id_values_do_not_influence_predictions(self):
        # Proves fixture_id is bookkeeping only: wildly different ids,
        # identical features -> identical probabilities.
        n = 25
        X = _synthetic_batch(n, seed=53)
        a = predict_candidate(self.model, self.preprocessor, X, _fixture_ids(n, start=1))
        b = predict_candidate(self.model, self.preprocessor, X, _fixture_ids(n, start=10**9))
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
        self.assertNotEqual(a.fixture_ids, b.fixture_ids)

    def test_fixture_id_column_in_input_is_not_used_as_a_feature(self):
        # Changing the fixture_id COLUMN inside X must not change output.
        n = 25
        X1 = _synthetic_batch(n, seed=59, with_fixture_id=True)
        X2 = X1.copy()
        X2["fixture_id"] = X2["fixture_id"] * -7  # arbitrary mutation
        a = predict_candidate(self.model, self.preprocessor, X1, _fixture_ids(n))
        b = predict_candidate(self.model, self.preprocessor, X2, _fixture_ids(n))
        np.testing.assert_array_equal(a.probabilities, b.probabilities)

    def test_column_reordering_in_input_does_not_change_predictions(self):
        # The contract order is imposed by the interface, so a caller
        # supplying shuffled columns must get the same answer -- not a
        # silently mis-ordered design matrix.
        n = 25
        X = _synthetic_batch(n, seed=61)
        X_shuffled = X[list(reversed(list(X.columns)))]
        a = predict_candidate(self.model, self.preprocessor, X, _fixture_ids(n))
        b = predict_candidate(self.model, self.preprocessor, X_shuffled, _fixture_ids(n))
        np.testing.assert_array_equal(a.probabilities, b.probabilities)

    # ---- input not mutated ----
    def test_input_frame_is_not_mutated(self):
        X = _synthetic_batch(20, seed=67, with_fixture_id=True)
        before = X.copy()
        predict_candidate(self.model, self.preprocessor, X, _fixture_ids(20))
        pd.testing.assert_frame_equal(X, before)

    # ---- minimum supported batch size ----
    def test_single_row_is_the_minimum_supported_batch(self):
        # §7 defines the input as a frame "for fixtures", and §10's
        # pre-inference checks ("no contract column entirely null for the
        # batch", "null rate per family within bounds") are only defined
        # for a batch containing at least one row. One row is therefore
        # the documented minimum.
        _, fids, pred = self._predict(n=1, seed=71)
        self.assertEqual(pred.n_rows, 1)
        self.assertEqual(pred.probabilities.shape, (1, 3))
        self.assertEqual(list(pred.fixture_ids), list(fids))
        validate_probabilities(pred.probabilities)

    def test_empty_batch_is_outside_the_documented_contract(self):
        """Characterization, not a requirement.

        An earlier version of this test asserted that a 0-row batch
        returns an empty aligned result. That was an INVENTED
        requirement: neither the Gate 8 row nor §7 nor §10 mentions
        empty batches, and §10's checks are undefined for zero rows.

        Empty input is therefore outside the contract. It surfaces an
        error originating in the frozen V1 `SimpleImputer`
        ("Found array with 0 sample(s) ... a minimum of 1 is required"),
        not in the inference boundary -- so this is a boundary of the
        frozen V1 preprocessing behaviour, not a defect in
        `predict_candidate`, and no production change was made.

        Asserted loosely (any exception) so the test documents the
        boundary without pinning a third-party error type or message.
        If empty-batch tolerance is ever wanted operationally, that is a
        NEW requirement needing explicit authorization and a spec change.
        """
        X = _synthetic_batch(0, seed=71)
        with self.assertRaises(Exception):
            predict_candidate(self.model, self.preprocessor, X, [])

    def test_validator_itself_still_accepts_a_zero_row_matrix(self):
        # Separate concern, unchanged by the above: `validate_probabilities`
        # has defined behaviour for (0, 3) -- verified in Gate 6. The
        # inference path's minimum batch size is independent of that.
        validate_probabilities(np.empty((0, 3)))

    def test_as_frame_roundtrip_on_a_real_prediction(self):
        n = 15
        _, fids, pred = self._predict(n=n, seed=73)
        frame = pred.as_frame()
        self.assertEqual(len(frame), n)
        self.assertEqual(list(frame["fixture_id"]), list(fids))
        np.testing.assert_allclose(
            frame[["p_home", "p_draw", "p_away"]].to_numpy(), pred.probabilities
        )


# =====================================================================
# Governance / side-effect verification
# =====================================================================
class TestGate8GovernanceBoundary(unittest.TestCase):
    _CONTRACT_SRC = Path("src/models/candidate_contract.py")

    @staticmethod
    def _this_tree():
        return ast.parse(Path(__file__).read_text(encoding="utf-8"))

    def _imported(self, tree):
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                names.add(n.module or "")
                names.update(a.name for a in n.names)
        return names

    def test_contract_module_still_imports_no_ml_library_at_top_level(self):
        tree = ast.parse(self._CONTRACT_SRC.read_text(encoding="utf-8"))
        top_level = set()
        for n in tree.body:
            if isinstance(n, ast.Import):
                top_level.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                top_level.add(n.module or "")
        self.assertNotIn("sklearn", top_level)

    def test_contract_module_never_fits(self):
        tree = ast.parse(self._CONTRACT_SRC.read_text(encoding="utf-8"))
        fits = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in {"fit", "fit_transform", "partial_fit"}
        ]
        self.assertEqual(fits, [])

    def test_contract_module_cannot_reach_final_test_split(self):
        tree = ast.parse(self._CONTRACT_SRC.read_text(encoding="utf-8"))
        imported = self._imported(tree)
        for forbidden in ("final_split", "FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, imported)

    def test_contract_module_writes_no_artifact(self):
        tree = ast.parse(self._CONTRACT_SRC.read_text(encoding="utf-8"))
        writes = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in {
                "write_text", "write_bytes", "to_csv", "to_json", "dump", "save"
            }
        ]
        self.assertEqual(writes, [])

    def test_this_gate_uses_no_real_match_data(self):
        imported = self._imported(self._this_tree())
        for forbidden in ("models.data", "load_supervised_dataset",
                          "models.splits", "final_split"):
            self.assertNotIn(forbidden, imported)

    def test_this_gate_computes_no_metric(self):
        imported = self._imported(self._this_tree())
        for forbidden in ("log_loss", "brier_score", "evaluate", "accuracy"):
            self.assertNotIn(forbidden, imported)

    def test_this_gate_imports_no_calibration_or_tuning(self):
        imported = self._imported(self._this_tree())
        for forbidden in ("models.calibration", "CalibratedClassifierCV",
                          "GridSearchCV", "RandomizedSearchCV"):
            self.assertNotIn(forbidden, imported)

    def test_model_version_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_contract_still_eighty_columns_in_frozen_order(self):
        from models.ablation import MODEL_B_COLUMNS
        self.assertEqual(len(CANDIDATE_FEATURE_COLUMNS), 80)
        self.assertEqual(tuple(CANDIDATE_FEATURE_COLUMNS), tuple(MODEL_B_COLUMNS))
        self.assertNotIn("league_home_advantage_season", CANDIDATE_FEATURE_COLUMNS)

    def test_no_v2_module_or_artifact(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_no_gate8_artifact_created(self):
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate8", p.name.lower())

    def test_v1_and_locked_artifacts_unchanged(self):
        expected = {
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
            "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
            "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
            "src/models/ablation.py": "9bf8bf4a1f332f8bb47d640a72384ed7",
            "src/models/baselines.py": "42e64e3a0ba8c4bf8cf264f80cdcd208",
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
            "data/audit/phase4c_predictions/model_b_logistic_regression_final_test.csv":
                "a8bb4493f1e9ff4fd1ae1b504d2319cf",
            "data/audit/phase5a_dropped_feature_comparison.json": "3ceb7090f9e19c106d88eaa7a8848818",
        }
        for rel, exp in expected.items():
            actual = hashlib.md5(Path(rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"locked artifact modified: {rel}")


if __name__ == "__main__":
    unittest.main()
