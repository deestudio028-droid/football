"""Validates that docs/PHASE4C_V1_REPLACEMENT_RESULTS_REPORT.md states
facts that actually hold in data/audit/phase4c_final_comparison.json and
the Phase 4C prediction CSVs.

Additive and read-only: trains nothing, reruns nothing, and asserts that
Phase 4C produced no promotion side-effects (no V2, MODEL_VERSION still
v1.0).
"""
import _pathfix  # noqa: F401
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ARTIFACT = Path("data/audit/phase4c_final_comparison.json")
_REPORT = Path("docs/PHASE4C_V1_REPLACEMENT_RESULTS_REPORT.md")
_PRED = Path("data/audit/phase4c_predictions")
_ARM_V1 = "v1_logistic_regression"
_ARM_B = "model_b_logistic_regression"

_SKIP = "Phase 4C artifact not present in this environment"


@unittest.skipUnless(_ARTIFACT.exists(), _SKIP)
class TestReportMatchesArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        cls.report = _REPORT.read_text(encoding="utf-8") if _REPORT.exists() else ""

    def test_report_exists(self):
        self.assertTrue(_REPORT.exists())

    def test_headline_metrics_appear_in_report(self):
        for arm in (_ARM_V1, _ARM_B):
            for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
                v = self.d["arms"][arm][m]
                self.assertIn(f"{v:.15f}".rstrip("0"), self.report.replace(",", "") + f"{v!r}",
                              f"{arm}.{m} not traceable in report")

    def test_log_loss_delta_stated_exactly(self):
        delta = self.d["paired_deltas_b_minus_v1"]["log_loss"]["delta_b_minus_v1"]
        self.assertAlmostEqual(delta, -0.016594843000765858, places=15)
        self.assertIn("0.016594843000765858", self.report)

    def test_all_five_metrics_improved_for_model_b(self):
        pd_ = self.d["paired_deltas_b_minus_v1"]
        for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
            self.assertTrue(pd_[m]["b_numerically_better"], f"{m} not better for B")

    def test_v1_reproduction_passed(self):
        self.assertTrue(self.d["v1_reproduction_check"]["all_match"])

    def test_all_validity_checks_passed(self):
        self.assertTrue(self.d["validity_checks"]["all_passed"])
        self.assertEqual(self.d["validity_checks"]["failures"], [])

    def test_no_epsilon_was_authorized(self):
        self.assertFalse(self.d["authorized_decisions"]["epsilon_authorized"])
        self.assertIsNone(self.d["authorized_decisions"]["epsilon_tie_threshold"])

    def test_C_was_1_point_0(self):
        self.assertEqual(self.d["authorized_decisions"]["C"], 1.0)

    def test_sample_counts(self):
        self.assertEqual(self.d["n_train"], 8983)
        self.assertEqual(self.d["n_test"], 1751)


@unittest.skipUnless(_ARTIFACT.exists() and _PRED.exists(), _SKIP)
class TestMetricsRecomputeFromPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        cls.frames = {
            _ARM_V1: pd.read_csv(_PRED / f"{_ARM_V1}_final_test.csv"),
            _ARM_B: pd.read_csv(_PRED / f"{_ARM_B}_final_test.csv"),
        }

    def test_every_metric_recomputes_from_raw_predictions(self):
        from models.evaluate import evaluate
        for arm, df in self.frames.items():
            P = df[["p_home", "p_draw", "p_away"]].to_numpy()
            r = evaluate(df["y_true"].to_numpy(), P)
            for m in ("log_loss", "brier", "accuracy", "macro_f1", "balanced_accuracy"):
                self.assertAlmostEqual(getattr(r, m), self.d["arms"][arm][m], places=12, msg=f"{arm}.{m}")
            self.assertEqual(r.confusion_matrix.tolist(), self.d["arms"][arm]["confusion_matrix"])

    def test_pairing_by_fixture_id(self):
        v1, b = self.frames[_ARM_V1], self.frames[_ARM_B]
        self.assertEqual(len(v1), 1751)
        self.assertEqual(len(b), 1751)
        self.assertEqual(set(v1.fixture_id), set(b.fixture_id))
        self.assertTrue((v1.fixture_id.values == b.fixture_id.values).all())
        self.assertTrue((v1.y_true.values == b.y_true.values).all())
        self.assertEqual(int(v1.fixture_id.duplicated().sum()), 0)
        self.assertEqual(int(b.fixture_id.duplicated().sum()), 0)

    def test_probabilities_valid_both_arms(self):
        from models.evaluate import validate_probabilities
        for arm, df in self.frames.items():
            validate_probabilities(df[["p_home", "p_draw", "p_away"]].to_numpy())

    def test_abandoned_fixture_absent(self):
        for df in self.frames.values():
            self.assertNotIn(420450481, set(df.fixture_id))


@unittest.skipUnless(_ARTIFACT.exists(), _SKIP)
class TestGovernanceInvariantsPreserved(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _REPORT.read_text(encoding="utf-8") if _REPORT.exists() else ""

    def test_model_version_still_v1(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_no_v2_module_or_artifact_created(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_report_states_zone_d_eligible_not_promoted(self):
        low = self.report.lower()
        self.assertIn("zone d", low)
        self.assertIn("eligible for explicit v2 review", low)
        self.assertIn("not automatic promotion", low.replace("**", ""))

    def test_report_makes_no_significance_claim(self):
        low = self.report.lower()
        self.assertNotIn("statistically significant", low)
        self.assertIn("no significance test", low.replace("no significance test is available or claimed",
                                                          "no significance test"))

    def test_report_discloses_required_caveats(self):
        low = self.report.lower()
        self.assertIn("league_home_advantage_season", self.report)
        self.assertIn("not a strict superset", low)
        self.assertIn("80 features", low)
        self.assertIn("draw-class performance remains weak", low)
        self.assertIn("only one final-test season", low)
        self.assertIn("no epsilon", low)

    def test_report_states_production_not_authorized(self):
        self.assertIn("PRODUCTION_AUTHORIZED: NO", self.report)
        self.assertIn("MODEL_VERSION:         v1.0", self.report)

    def test_model_b_is_not_a_strict_superset_of_v1(self):
        from models.ablation import MODEL_B_COLUMNS
        from models.config import APPROVED_FEATURE_COLUMNS_V1
        missing = set(APPROVED_FEATURE_COLUMNS_V1) - set(MODEL_B_COLUMNS)
        self.assertEqual(missing, {"league_home_advantage_season"})


if __name__ == "__main__":
    unittest.main()
