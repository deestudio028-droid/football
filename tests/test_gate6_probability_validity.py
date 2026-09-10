"""Gate 6 — Probability validity.

Gate 6 requirement (docs/PHASE5_V2_MODEL_SPEC_DRAFT.md):
    "All outputs valid 3-class probability vectors."
    Verification: `validate_probabilities` over a full prediction batch.
    Pass: 100% pass. Stop: any NaN/inf/negative/non-summing row.

The authoritative validator is `evaluate.validate_probabilities`. It is
tested and integrated here, NOT rewritten -- it is V1 production code and
a second probability definition is explicitly forbidden.

This is a VALIDITY gate, not a calibration gate. Draw-class predictive
weakness is out of scope (governed by Condition 2 / Phase 5B) and is not
a Gate 6 concern.

OPEN SPEC/IMPLEMENTATION DISCREPANCY (reported, not silently resolved):
spec §8 states rows must sum to 1 "within atol=1e-6", but the
implementation calls `np.allclose(row_sums, 1.0, atol=1e-6)`, and
numpy's allclose applies |a-b| <= atol + rtol*|b| with a default
rtol=1e-05. The effective acceptance boundary is therefore ~1.1e-05,
about 11x looser than the documented atol. The boundary tests below are
written as CHARACTERIZATION tests of the implementation's actual
behaviour and are named to make that explicit. They deliberately do not
assert the spec's 1e-6 figure (which would fail) nor silently endorse
1.1e-5 as the intended rule. See the Gate 6 report: this requires a
human decision, and resolving it would mean editing V1 production code
(forbidden here) plus revalidating Phase 3/4C artifacts.

Practical note: every real prediction row observed in this repository
sums to 1 within <= 4.5e-16, so no real output is anywhere near either
candidate boundary. The discrepancy concerns the strictness of the guard,
not the validity of current predictions.
"""
import _pathfix  # noqa: F401
import ast
import hashlib
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models.baselines import CLASS_ORDER
from models.evaluate import MetricInputError, validate_probabilities

# Empirically measured acceptance boundary of the CURRENT implementation
# (binary-searched): rows deviating from 1.0 by up to ~1.1e-5 are
# accepted. Documented as observed behaviour, not as the spec's rule.
OBSERVED_ROW_SUM_BOUNDARY = 1.1e-5
SPEC_STATED_ATOL = 1e-6

_PRED_DIRS = [
    Path("data/audit/phase4c_predictions"),
    Path("data/audit/phase4a_predictions"),
    Path("data/audit/phase5a_predictions"),
]
_PROB_COLUMNS = ["p_home", "p_draw", "p_away"]


def _row(*vals):
    return np.array([list(vals)], dtype=float)


# =====================================================================
# 1. Validator audit — is this really the authoritative one?
# =====================================================================
class TestGate6ValidatorAudit(unittest.TestCase):
    def test_validator_is_the_evaluate_module_function(self):
        self.assertEqual(validate_probabilities.__module__, "models.evaluate")

    def test_validator_defaults_to_frozen_class_order(self):
        sig = inspect.signature(validate_probabilities)
        self.assertEqual(sig.parameters["class_order"].default, CLASS_ORDER)

    def test_class_order_is_exactly_h_d_a(self):
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])
        self.assertEqual(len(CLASS_ORDER), 3)

    def test_class_order_derives_from_config_allowed_labels(self):
        from models.config import ALLOWED_LABELS
        self.assertEqual(tuple(CLASS_ORDER), ALLOWED_LABELS)

    def test_validator_raises_rather_than_returning_a_flag(self):
        # Spec §8: "A failure is a hard stop, not a warning."
        self.assertIsNone(validate_probabilities(_row(0.5, 0.3, 0.2)))
        with self.assertRaises(MetricInputError):
            validate_probabilities(_row(0.5, 0.3, 0.9))

    def test_validator_performs_no_mutation_or_normalization(self):
        # It must not clip, renormalize, or zero-fill to force validity.
        src = inspect.getsource(validate_probabilities)
        tree = ast.parse(src)
        mutating = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                name = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
                if name in {"clip", "nan_to_num", "fillna", "normalize", "round"}:
                    mutating.append(name)
        self.assertEqual(mutating, [], f"validator must not mutate input: {mutating}")

    def test_validator_does_not_modify_caller_array(self):
        P = _row(0.5, 0.3, 0.2)
        before = P.copy()
        validate_probabilities(P)
        np.testing.assert_array_equal(P, before)


# =====================================================================
# 2. Positive controls
# =====================================================================
class TestGate6PositiveControls(unittest.TestCase):
    def test_single_valid_row(self):
        validate_probabilities(_row(0.5, 0.3, 0.2))

    def test_multiple_valid_rows(self):
        P = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [1 / 3, 1 / 3, 1 / 3]])
        validate_probabilities(P)

    def test_row_with_zero_probability_for_one_class(self):
        validate_probabilities(_row(1.0, 0.0, 0.0))
        validate_probabilities(_row(0.0, 1.0, 0.0))
        validate_probabilities(_row(0.0, 0.0, 1.0))

    def test_row_with_two_zero_probabilities(self):
        validate_probabilities(_row(1.0, 0.0, 0.0))

    def test_row_with_genuine_float64_residue_is_accepted(self):
        # (0.7, 0.2, 0.1) sums to 0.9999999999999999 in float64 -- a real
        # representation residue of ~1.1e-16, which must be accepted.
        # (Note: 1/3+1/3+1/3 does sum to exactly 1.0 in float64, so it is
        # not a usable example of residue.)
        P = _row(0.7, 0.2, 0.1)
        self.assertNotEqual(P.sum(), 1.0)
        self.assertLess(abs(P.sum() - 1.0), 1e-15)
        validate_probabilities(P)

    def test_thirds_sum_exactly_and_are_accepted(self):
        P = _row(1 / 3, 1 / 3, 1 / 3)
        self.assertEqual(P.sum(), 1.0)
        validate_probabilities(P)

    def test_large_valid_batch(self):
        rng = np.random.default_rng(0)
        P = rng.dirichlet([2, 1, 2], size=5000)
        validate_probabilities(P)
        self.assertEqual(P.shape, (5000, 3))

    def test_empty_but_well_shaped_matrix_is_accepted(self):
        # Defined behaviour: (0, 3) has no offending row.
        validate_probabilities(np.empty((0, 3)))


# =====================================================================
# 3. Negative controls
# =====================================================================
class TestGate6NegativeControls(unittest.TestCase):
    def _rejects(self, P, expect_fragment=None):
        with self.assertRaises(MetricInputError) as ctx:
            validate_probabilities(P)
        if expect_fragment:
            self.assertIn(expect_fragment, str(ctx.exception))

    def test_two_class_output_rejected(self):
        self._rejects(np.array([[0.5, 0.5]]), "shape")

    def test_four_class_output_rejected(self):
        self._rejects(np.array([[0.25, 0.25, 0.25, 0.25]]), "shape")

    def test_nan_rejected(self):
        self._rejects(_row(np.nan, 0.5, 0.5), "NaN or infinite")

    def test_positive_infinity_rejected(self):
        self._rejects(_row(np.inf, 0.0, 0.0), "NaN or infinite")

    def test_negative_infinity_rejected(self):
        self._rejects(_row(-np.inf, 1.0, 1.0), "NaN or infinite")

    def test_negative_probability_rejected(self):
        self._rejects(_row(1.2, -0.1, -0.1), "negative")

    def test_sum_greater_than_one_rejected(self):
        self._rejects(_row(0.6, 0.3, 0.2), "do not sum")

    def test_sum_less_than_one_rejected(self):
        self._rejects(_row(0.2, 0.3, 0.2), "do not sum")

    def test_one_dimensional_input_rejected(self):
        self._rejects(np.array([0.5, 0.3, 0.2]), "shape")

    def test_three_dimensional_input_rejected(self):
        self._rejects(np.zeros((2, 3, 1)), "shape")

    def test_flat_empty_list_rejected(self):
        self._rejects(np.array([]), "shape")

    def test_single_bad_row_in_an_otherwise_valid_batch_is_rejected(self):
        P = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.9], [0.2, 0.3, 0.5]])
        self._rejects(P, "do not sum")

    def test_error_message_identifies_an_offending_row(self):
        P = np.array([[0.5, 0.3, 0.2], [0.9, 0.9, 0.9]])
        with self.assertRaises(MetricInputError) as ctx:
            validate_probabilities(P)
        self.assertIn("row 1", str(ctx.exception))


# =====================================================================
# 4. Boundary / tolerance — CHARACTERIZATION of current behaviour
# =====================================================================
class TestGate6ToleranceCharacterization(unittest.TestCase):
    """These record the implementation's ACTUAL acceptance boundary.

    They are named "observed_" deliberately: spec §8 documents
    atol=1e-6, while the implementation's effective boundary is ~1.1e-5
    because np.allclose adds rtol*|b|. See the module docstring and the
    Gate 6 report -- this discrepancy is reported for human decision, not
    resolved here.
    """

    def _accepts(self, delta):
        try:
            validate_probabilities(_row(0.5 + delta, 0.3, 0.2))
            return True
        except MetricInputError:
            return False

    def test_exact_sum_accepted(self):
        self.assertTrue(self._accepts(0.0))

    def test_deviation_at_spec_stated_atol_is_accepted(self):
        # 1e-6 is inside both readings, so this holds either way.
        self.assertTrue(self._accepts(SPEC_STATED_ATOL))

    def test_deviation_just_inside_observed_boundary_accepted(self):
        self.assertTrue(self._accepts(OBSERVED_ROW_SUM_BOUNDARY * 0.99))

    def test_deviation_just_beyond_observed_boundary_rejected(self):
        self.assertFalse(self._accepts(OBSERVED_ROW_SUM_BOUNDARY * 1.01))

    def test_deviation_far_beyond_any_tolerance_rejected(self):
        for delta in (1e-4, 1e-3, 1e-2):
            self.assertFalse(self._accepts(delta), f"delta={delta} must be rejected")

    def test_observed_boundary_is_looser_than_spec_stated_atol(self):
        # Documents the open discrepancy explicitly so it cannot be
        # forgotten. If the validator is later tightened to a strict
        # atol=1e-6, THIS TEST WILL FAIL -- which is the intended signal
        # to revisit the governance decision, not a regression.
        self.assertTrue(self._accepts(5e-6), "5e-6 currently accepted (beyond spec's atol)")
        self.assertGreater(OBSERVED_ROW_SUM_BOUNDARY, SPEC_STATED_ATOL)

    def test_tolerance_is_not_weakened_by_these_tests(self):
        # No approximate assertion is used to decide acceptance anywhere
        # in this class: acceptance is decided solely by the validator.
        src = inspect.getsource(TestGate6ToleranceCharacterization)
        tree = ast.parse(src)
        loose = [
            (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
            for n in ast.walk(tree) if isinstance(n, ast.Call)
        ]
        for forbidden in ("assertAlmostEqual", "allclose", "assert_allclose"):
            self.assertNotIn(forbidden, loose)


# =====================================================================
# 5. Real candidate prediction validation (no sklearn needed)
# =====================================================================
def _existing_prediction_files():
    out = []
    for d in _PRED_DIRS:
        if d.exists():
            out.extend(sorted(d.glob("*.csv")))
    return out


@unittest.skipUnless(_existing_prediction_files(), "no existing prediction artifacts present")
class TestGate6RealCandidatePredictions(unittest.TestCase):
    """Validates real, already-produced candidate output. Validity only:
    no metric is computed and no performance comparison is made."""

    @classmethod
    def setUpClass(cls):
        cls.files = _existing_prediction_files()

    def test_every_prediction_artifact_passes_validation(self):
        for path in self.files:
            df = pd.read_csv(path)
            if not all(c in df.columns for c in _PROB_COLUMNS):
                continue
            P = df[_PROB_COLUMNS].to_numpy()
            try:
                validate_probabilities(P)
            except MetricInputError as exc:
                self.fail(f"{path.name} failed probability validation: {exc}")

    def test_candidate_model_b_final_test_batch_is_valid(self):
        p = Path("data/audit/phase4c_predictions/model_b_logistic_regression_final_test.csv")
        if not p.exists():
            self.skipTest("candidate final-test prediction artifact absent")
        df = pd.read_csv(p)
        P = df[_PROB_COLUMNS].to_numpy()
        validate_probabilities(P)
        self.assertEqual(P.shape, (1751, 3))

    def test_probability_columns_are_in_frozen_class_order(self):
        # p_home/p_draw/p_away must map to H/D/A positionally.
        self.assertEqual(_PROB_COLUMNS[CLASS_ORDER.index("H")], "p_home")
        self.assertEqual(_PROB_COLUMNS[CLASS_ORDER.index("D")], "p_draw")
        self.assertEqual(_PROB_COLUMNS[CLASS_ORDER.index("A")], "p_away")

    def test_real_rows_are_far_inside_any_candidate_tolerance(self):
        worst = 0.0
        for path in self.files:
            df = pd.read_csv(path)
            if not all(c in df.columns for c in _PROB_COLUMNS):
                continue
            P = df[_PROB_COLUMNS].to_numpy()
            worst = max(worst, float(np.abs(P.sum(axis=1) - 1.0).max()))
        # Real output is ~1e-16, orders of magnitude inside 1e-6.
        self.assertLess(worst, SPEC_STATED_ATOL)

    def test_no_real_row_is_negative_or_nonfinite(self):
        for path in self.files:
            df = pd.read_csv(path)
            if not all(c in df.columns for c in _PROB_COLUMNS):
                continue
            P = df[_PROB_COLUMNS].to_numpy()
            self.assertTrue(np.isfinite(P).all(), path.name)
            self.assertTrue((P >= 0).all(), path.name)


# =====================================================================
# 6. Governance / side-effect verification (AST-based, not self-grepping)
# =====================================================================
class TestGate6GovernanceBoundary(unittest.TestCase):
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

    def test_no_training_or_tuning_machinery_imported(self):
        imported = self._imported()
        for forbidden in ("sklearn", "models.train", "LogisticRegression",
                          "GridSearchCV", "RandomizedSearchCV"):
            self.assertNotIn(forbidden, imported)

    def test_no_calibration_imported_or_called(self):
        imported = self._imported()
        for forbidden in ("models.calibration", "CalibratedClassifierCV"):
            self.assertNotIn(forbidden, imported)
        calls = [
            (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
            for n in self._tree().body and ast.walk(self._tree()) if isinstance(n, ast.Call)
        ]
        for forbidden in ("fit_calibration", "calibrate"):
            self.assertNotIn(forbidden, calls)

    def test_final_test_seasons_not_used_for_selection(self):
        imported = self._imported()
        for forbidden in ("final_split", "FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, imported)

    def test_no_metric_computed_in_this_gate(self):
        imported = self._imported()
        for forbidden in ("evaluate", "log_loss", "brier_score", "macro_f1"):
            self.assertNotIn(forbidden, imported)

    def test_no_artifact_written(self):
        writes = [
            n for n in ast.walk(self._tree())
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in {"write_text", "write_bytes", "to_csv", "to_json", "dump", "save"}
        ]
        self.assertEqual(writes, [])

    def test_model_version_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_class_order_unchanged(self):
        self.assertEqual(CLASS_ORDER, ["H", "D", "A"])

    def test_candidate_contract_unchanged(self):
        from models.ablation import MODEL_B_COLUMNS
        from models.candidate_contract import CANDIDATE_FEATURE_COLUMNS
        self.assertEqual(len(CANDIDATE_FEATURE_COLUMNS), 80)
        self.assertEqual(tuple(CANDIDATE_FEATURE_COLUMNS), tuple(MODEL_B_COLUMNS))
        self.assertNotIn("league_home_advantage_season", CANDIDATE_FEATURE_COLUMNS)

    def test_no_v2_module_or_artifact(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_no_gate6_artifact_created(self):
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate6", p.name.lower())

    def test_v1_source_and_locked_artifacts_unchanged(self):
        expected = {
            "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/baselines.py": "42e64e3a0ba8c4bf8cf264f80cdcd208",
            "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
            "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
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
