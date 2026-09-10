"""Gate 7 — Regression against Phase 4C.

Gate 7 requirement (docs/PHASE5_V2_MODEL_SPEC_DRAFT.md):
    "Implementation reproduces the audited Phase 4C Model B result."
    Compare implementation output on the 2025/26 rows to
    `phase4c_final_comparison.json` (verification only -- not a new
    evaluation, not a new selection input).
    Pass: log loss reproduces 0.9960487649063601 to <1e-9.
    Stop: any deviation >= 1e-9.

This is a REPRODUCTION / REGRESSION check. The 2025/26 rows are used
ONLY as a frozen regression fixture. Nothing here selects, tunes,
calibrates, or optimises anything, and no result feeds back into the
candidate definition.

IMPORTANT SCOPE NOTE: recomputing log loss from the stored Phase 4C CSV
supports the *reference* checks but does NOT satisfy Gate 7 on its own --
Gate 7 asks whether the CURRENT implementation can reproduce the frozen
number by actually training. That empirical reproduction requires
scikit-learn (V1's estimator/preprocessor). Where scikit-learn is
unavailable the reproduction test is dependency-gated and Gate 7 must be
reported INCONCLUSIVE, never PASS.

No replica implementation is substituted: reproduction goes through
`train.train_logistic_regression` unchanged.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models.ablation import MODEL_B_COLUMNS, XG_CORE_COLUMNS
from models.baselines import CLASS_ORDER
from models.candidate_contract import (
    CANDIDATE_FEATURE_COLUMNS,
    select_candidate_features,
)
from models.config import FINAL_TRAIN_SEASONS, X_EXCLUDED_COLUMNS
from models.evaluate import MetricInputError, log_loss, validate_probabilities

try:
    from models import train as train_module
    _IMPORT_ERROR = None
except ImportError as exc:
    train_module = None
    _IMPORT_ERROR = str(exc)

_SKLEARN_SKIP = f"scikit-learn not available in this environment: {_IMPORT_ERROR}"

# ---- Frozen Phase 4C reference values (authoritative artifact) -------
PHASE4C_COMPARISON = Path("data/audit/phase4c_final_comparison.json")
PHASE4C_MANIFEST = Path("data/audit/phase4c_prerun_manifest.json")
MODEL_B_ARM = "model_b_logistic_regression"
MODEL_B_CSV = Path("data/audit/phase4c_predictions/model_b_logistic_regression_final_test.csv")

REFERENCE_LOG_LOSS = 0.9960487649063601
REGRESSION_TOLERANCE = 1e-9
EXPECTED_N_TEST = 1751
EXPECTED_N_TRAIN = 8983
PROB_COLUMNS = ["p_home", "p_draw", "p_away"]
FROZEN_ESTIMATOR_KWARGS = {"max_iter": 2000, "C": 1.0, "random_state": 0}

_REAL_DB = Path("data/processed/features.db")
_DATA_SKIP = "data/processed/features.db not present in this environment"


def _artifacts_present() -> bool:
    return PHASE4C_COMPARISON.exists() and MODEL_B_CSV.exists()


# =====================================================================
# A. Reference artifact integrity
# =====================================================================
@unittest.skipUnless(_artifacts_present(), "Phase 4C artifacts absent")
class TestGate7ReferenceArtifactIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads(PHASE4C_COMPARISON.read_text(encoding="utf-8"))
        cls.df = pd.read_csv(MODEL_B_CSV)

    def test_comparison_artifact_exists_and_parses(self):
        self.assertIn("arms", self.d)
        self.assertIn(MODEL_B_ARM, self.d["arms"])

    def test_reference_log_loss_is_exactly_the_frozen_value(self):
        self.assertEqual(self.d["arms"][MODEL_B_ARM]["log_loss"], REFERENCE_LOG_LOSS)

    def test_reference_value_used_by_this_gate_matches_the_artifact(self):
        # Guards against a hard-coded constant drifting from the artifact.
        self.assertEqual(REFERENCE_LOG_LOSS, self.d["arms"][MODEL_B_ARM]["log_loss"])

    def test_model_b_prediction_artifact_exists(self):
        self.assertTrue(MODEL_B_CSV.exists())

    def test_reference_row_count_is_1751(self):
        self.assertEqual(len(self.df), EXPECTED_N_TEST)
        self.assertEqual(self.d["arms"][MODEL_B_ARM]["n_test"], EXPECTED_N_TEST)

    def test_reference_train_row_count_is_8983(self):
        self.assertEqual(self.d["arms"][MODEL_B_ARM]["n_train"], EXPECTED_N_TRAIN)

    def test_fixture_ids_are_unique(self):
        self.assertEqual(int(self.df["fixture_id"].duplicated().sum()), 0)

    def test_probability_columns_present_in_class_order_positions(self):
        for c in PROB_COLUMNS:
            self.assertIn(c, self.df.columns)
        self.assertEqual(PROB_COLUMNS[CLASS_ORDER.index("H")], "p_home")
        self.assertEqual(PROB_COLUMNS[CLASS_ORDER.index("D")], "p_draw")
        self.assertEqual(PROB_COLUMNS[CLASS_ORDER.index("A")], "p_away")

    def test_reference_probabilities_are_valid(self):
        validate_probabilities(self.df[PROB_COLUMNS].to_numpy())

    def test_reference_labels_are_h_d_a_only(self):
        self.assertTrue(set(self.df["y_true"]).issubset({"H", "D", "A"}))

    def test_stored_csv_recomputes_to_the_reference_log_loss(self):
        # Supports the reference checks. NOT sufficient for Gate 7 PASS --
        # see the module docstring.
        recomputed = log_loss(self.df["y_true"].to_numpy(), self.df[PROB_COLUMNS].to_numpy())
        self.assertLess(abs(recomputed - REFERENCE_LOG_LOSS), REGRESSION_TOLERANCE)

    def test_manifest_records_the_frozen_configuration(self):
        m = json.loads(PHASE4C_MANIFEST.read_text(encoding="utf-8"))
        fs = m["frozen_spec"]
        self.assertEqual(fs["estimator_kwargs"], FROZEN_ESTIMATOR_KWARGS)
        self.assertEqual(fs["class_order"], CLASS_ORDER)
        self.assertEqual(fs["model_b_n_features"], 80)
        self.assertEqual(list(fs["final_train_seasons"]), list(FINAL_TRAIN_SEASONS))
        self.assertEqual(m["authorized_decisions"]["C"], 1.0)


# =====================================================================
# B. Candidate contract identity
# =====================================================================
class TestGate7CandidateContractIdentity(unittest.TestCase):
    def test_contract_matches_model_b_columns_in_exact_order(self):
        self.assertEqual(tuple(CANDIDATE_FEATURE_COLUMNS), tuple(MODEL_B_COLUMNS))

    def test_contract_is_exactly_eighty_columns(self):
        self.assertEqual(len(CANDIDATE_FEATURE_COLUMNS), 80)

    def test_omitted_v1_feature_remains_absent(self):
        self.assertNotIn("league_home_advantage_season", CANDIDATE_FEATURE_COLUMNS)

    def test_no_xg_columns(self):
        self.assertTrue(set(CANDIDATE_FEATURE_COLUMNS).isdisjoint(XG_CORE_COLUMNS))

    def test_no_label_or_bookkeeping_columns(self):
        self.assertEqual(set(CANDIDATE_FEATURE_COLUMNS) & set(X_EXCLUDED_COLUMNS), set())

    @unittest.skipUnless(_artifacts_present(), "Phase 4C artifacts absent")
    def test_phase4c_recorded_columns_match_current_contract_exactly(self):
        # The strongest identity check available: the audited run recorded
        # the exact column list it used, including order.
        d = json.loads(PHASE4C_COMPARISON.read_text(encoding="utf-8"))
        recorded = d["arms"][MODEL_B_ARM]["feature_columns"]
        self.assertEqual(len(recorded), 80)
        self.assertEqual(tuple(recorded), tuple(CANDIDATE_FEATURE_COLUMNS),
                         "current contract differs from the columns Phase 4C actually used")


# =====================================================================
# C. Training / configuration identity
# =====================================================================
class TestGate7ConfigurationIdentity(unittest.TestCase):
    def _train_py_logreg_kwargs(self):
        tree = ast.parse(Path("src/models/train.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LogisticRegression":
                return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        self.fail("no LogisticRegression construction found in train.py")

    def test_estimator_kwargs_unchanged(self):
        self.assertEqual(self._train_py_logreg_kwargs(), FROZEN_ESTIMATOR_KWARGS)

    def test_C_is_one(self):
        self.assertEqual(self._train_py_logreg_kwargs()["C"], 1.0)

    def test_max_iter_is_2000(self):
        self.assertEqual(self._train_py_logreg_kwargs()["max_iter"], 2000)

    def test_random_state_is_zero(self):
        self.assertEqual(self._train_py_logreg_kwargs()["random_state"], 0)

    def test_class_order_unchanged(self):
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])

    def test_no_calibration_in_the_training_path(self):
        src = inspect.getsource(train_module) if train_module else Path("src/models/train.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        self.assertNotIn("CalibratedClassifierCV", imported)

    def test_preprocessing_path_is_v1s_preprocessor(self):
        src = Path("src/models/train.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "train_logistic_regression")
        body = ast.get_source_segment(src, fn)
        self.assertIn("LogisticRegressionPreprocessor().fit(X_train)", body)
        self.assertIn("_reorder_proba(model, X_val_enc)", body)

    def test_final_train_seasons_unchanged(self):
        self.assertEqual(
            FINAL_TRAIN_SEASONS,
            ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"),
        )


# =====================================================================
# E. Prediction alignment against the dataset's 2025/26 rows
# =====================================================================
@unittest.skipUnless(_artifacts_present() and _REAL_DB.exists(), _DATA_SKIP)
class TestGate7PredictionAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.data import load_supervised_dataset
        from models.splits import final_split
        cls.df = pd.read_csv(MODEL_B_CSV)
        dataset = load_supervised_dataset(_REAL_DB)
        _, cls.test_ds = final_split(dataset)

    def test_row_count_matches(self):
        self.assertEqual(len(self.df), EXPECTED_N_TEST)
        self.assertEqual(len(self.test_ds), EXPECTED_N_TEST)

    def test_fixture_id_sets_are_identical(self):
        self.assertEqual(set(self.df["fixture_id"]), set(self.test_ds.metadata["fixture_id"]))

    def test_fixture_id_ordering_is_identical(self):
        self.assertEqual(list(self.df["fixture_id"]), list(self.test_ds.metadata["fixture_id"]))

    def test_y_true_identical(self):
        truth = self.test_ds.metadata[["fixture_id"]].copy()
        truth["y"] = self.test_ds.y.values
        merged = self.df.merge(truth, on="fixture_id", how="left")
        self.assertTrue((merged["y_true"] == merged["y"]).all())

    def test_no_duplicates_and_no_missing_rows(self):
        self.assertEqual(int(self.df["fixture_id"].duplicated().sum()), 0)
        truth = self.test_ds.metadata[["fixture_id"]].copy()
        truth["y"] = self.test_ds.y.values
        merged = self.df.merge(truth, on="fixture_id", how="left")
        self.assertEqual(int(merged["y"].isna().sum()), 0)

    def test_all_contract_columns_available_for_the_fixture_rows(self):
        X = select_candidate_features(self.test_ds.X)
        self.assertEqual(X.shape, (EXPECTED_N_TEST, 80))
        self.assertEqual(list(X.columns), list(CANDIDATE_FEATURE_COLUMNS))


# =====================================================================
# D. Empirical regression reproduction (dependency-gated)
# =====================================================================
@unittest.skipUnless(train_module is not None, _SKLEARN_SKIP)
@unittest.skipUnless(_artifacts_present() and _REAL_DB.exists(), _DATA_SKIP)
class TestGate7EmpiricalRegressionReproduction(unittest.TestCase):
    """Phase 4C regression reproduction.

    Trains the frozen candidate ONCE on the frozen training partition and
    predicts the exact Phase 4C fixture rows, then compares log loss to
    the frozen reference. This is reproduction verification, NOT a new
    model evaluation and NOT a selection input.
    """

    @classmethod
    def setUpClass(cls):
        from models.data import load_supervised_dataset
        from models.splits import final_split
        dataset = load_supervised_dataset(_REAL_DB)
        cls.train_ds, cls.test_ds = final_split(dataset)
        cls.X_train = select_candidate_features(cls.train_ds.X)
        cls.X_test = select_candidate_features(cls.test_ds.X)
        # Single training run through the unmodified V1 path.
        _, _, cls.P = train_module.train_logistic_regression(
            cls.X_train, cls.train_ds.y, cls.X_test
        )
        cls.reproduced_log_loss = log_loss(cls.test_ds.y.to_numpy(), cls.P)

    def test_training_partition_size_matches_phase4c(self):
        self.assertEqual(len(self.train_ds), EXPECTED_N_TRAIN)

    def test_prediction_matrix_shape(self):
        self.assertEqual(self.P.shape, (EXPECTED_N_TEST, 3))

    def test_reproduced_probabilities_are_valid(self):
        validate_probabilities(self.P)

    def test_reproduced_log_loss_matches_reference_within_1e_minus_9(self):
        delta = abs(self.reproduced_log_loss - REFERENCE_LOG_LOSS)
        self.assertLess(
            delta, REGRESSION_TOLERANCE,
            f"Phase 4C regression reproduction FAILED: reproduced="
            f"{self.reproduced_log_loss!r} reference={REFERENCE_LOG_LOSS!r} delta={delta:.3e} "
            f"(tolerance {REGRESSION_TOLERANCE}). Do not adjust the tolerance -- diagnose.",
        )

    def test_reproduced_probabilities_match_the_stored_artifact(self):
        # Stronger than the scalar metric: per-row probability agreement.
        stored = pd.read_csv(MODEL_B_CSV)
        stored_P = stored[PROB_COLUMNS].to_numpy()
        self.assertEqual(stored_P.shape, self.P.shape)
        max_abs = float(np.abs(stored_P - self.P).max())
        self.assertLess(max_abs, 1e-9,
                        f"per-row probabilities diverge from the stored artifact: max|Δ|={max_abs:.3e}")

    def test_reproduced_argmax_classes_match_the_stored_artifact(self):
        stored = pd.read_csv(MODEL_B_CSV)
        reproduced_pred = np.array(CLASS_ORDER)[self.P.argmax(axis=1)]
        self.assertTrue((reproduced_pred == stored["y_pred"].to_numpy()).all())


# =====================================================================
# F. No mutation / governance
# =====================================================================
class TestGate7NoMutation(unittest.TestCase):
    """Gate 7 must not modify any artifact, production file, or version."""

    @staticmethod
    def _tree():
        return ast.parse(Path(__file__).read_text(encoding="utf-8"))

    def _imported(self):
        names = set()
        for n in ast.walk(self._tree()):
            if isinstance(n, ast.Import):
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                names.add(n.module or "")
                names.update(a.name for a in n.names)
        return names

    def test_this_module_writes_nothing(self):
        writes = [
            n for n in ast.walk(self._tree())
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in {
                "write_text", "write_bytes", "to_csv", "to_json", "dump", "save", "mkdir", "unlink"
            }
        ]
        self.assertEqual(writes, [], "Gate 7 must not write or delete anything")

    def test_no_serialisation_library_imported(self):
        imported = self._imported()
        for forbidden in ("joblib", "pickle", "cloudpickle"):
            self.assertNotIn(forbidden, imported)

    def test_no_tuning_or_calibration_imported(self):
        imported = self._imported()
        for forbidden in ("GridSearchCV", "RandomizedSearchCV", "CalibratedClassifierCV",
                          "models.calibration"):
            self.assertNotIn(forbidden, imported)

    def test_tolerance_constant_is_exactly_1e_minus_9(self):
        self.assertEqual(REGRESSION_TOLERANCE, 1e-9)

    def test_reference_constant_is_not_rounded(self):
        self.assertEqual(repr(REFERENCE_LOG_LOSS), "0.9960487649063601")

    def test_model_version_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_no_v2_module_or_artifact(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_no_gate7_artifact_created(self):
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate7", p.name.lower())

    def test_phase4c_and_production_artifacts_unchanged(self):
        expected = {
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
            "data/audit/phase4c_prerun_manifest.json": "4f166d1826edbf7bfe1e022639a8b1f4",
            "data/audit/phase4c_predictions/model_b_logistic_regression_final_test.csv":
                "a8bb4493f1e9ff4fd1ae1b504d2319cf",
            "data/audit/phase4c_predictions/v1_logistic_regression_final_test.csv":
                "e32c5928a2593a5c5cd804c81134f17d",
            "data/audit/phase5a_dropped_feature_comparison.json": "3ceb7090f9e19c106d88eaa7a8848818",
            "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
            "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
            "src/models/ablation.py": "9bf8bf4a1f332f8bb47d640a72384ed7",
            # NOTE: `candidate_contract.py` is deliberately NOT pinned here.
            # Gate 7's invariant is "Phase 4C and V1 production artifacts
            # unchanged"; the candidate contract module is neither -- it is
            # the V2 candidate scaffolding that successive gates extend
            # under explicit authorization (Gate 3 added the availability
            # check, Gate 8 added the inference boundary). Pinning it here
            # was a design error: it would fail on every authorized gate
            # addition. Its integrity is instead enforced behaviourally by
            # the gate suites -- contract identity/order (Gate 1), no fit,
            # no top-level ML import, no final-split reachability and no
            # artifact writes (Gate 8).
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
        }
        for rel, exp in expected.items():
            actual = hashlib.md5(Path(rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"artifact modified during Gate 7: {rel}")


if __name__ == "__main__":
    unittest.main()
