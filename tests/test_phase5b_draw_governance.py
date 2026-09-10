"""Phase 5B — draw-limitation governance invariants.

Read-only. Verifies that the Condition 2 review states an outcome that is
one of the permitted values, that its factual claims match the frozen
Phase 4C artifacts, that the draw limitation is never described as
solved, and that no production/V2 side-effect occurred.
"""
import _pathfix  # noqa: F401
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REVIEW5 = Path("docs/PHASE5_V2_CANDIDATE_REVIEW.md")
_REVIEW5B = Path("docs/PHASE5B_DRAW_LIMITATION_REVIEW.md")
_P4C = Path("data/audit/phase4c_final_comparison.json")
_P5A = Path("data/audit/phase5a_dropped_feature_comparison.json")
_RESULTS5A = Path("docs/PHASE5A_DROPPED_FEATURE_RESULTS.md")
_SPEC_DRAFT = Path("docs/PHASE5_V2_MODEL_SPEC_DRAFT.md")
_PRED = Path("data/audit/phase4c_predictions")

PERMITTED_CONDITION2 = {
    "DRAW_LIMITATION_ACCEPTED",
    "DRAW_LIMITATION_NOT_ACCEPTED",
    "DRAW_REQUIREMENT_UNSPECIFIED",
}
PERMITTED_CONDITION1 = {
    "RETAIN_OMISSION_SUPPORTED",
    "RETAIN_OMISSION_NOT_SUPPORTED",
    "FEATURE_EFFECT_INCONCLUSIVE",
}
PERMITTED_V2_STATUS = {
    "REJECT_V2_CANDIDATE",
    "CONDITIONAL_V2_CANDIDATE",
    "APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION",
}


@unittest.skipUnless(_REVIEW5B.exists(), "Phase 5B review absent")
class TestCondition2Outcome(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = _REVIEW5B.read_text(encoding="utf-8")

    def test_declares_exactly_one_permitted_condition2_value(self):
        present = {s for s in PERMITTED_CONDITION2 if s in self.text}
        self.assertEqual(
            len(present), 1,
            f"Phase 5B must declare exactly one Condition 2 outcome; found {present}",
        )

    def test_no_invented_status_used(self):
        for bogus in ("DRAW_OK", "DRAW_SOLVED", "DRAW_ACCEPTABLE", "DRAW_LIMITATION_WAIVED"):
            self.assertNotIn(bogus, self.text)

    def test_draw_never_described_as_solved_or_reliable(self):
        low = self.text.lower()
        for forbidden in (
            "draw prediction is good",
            "draw prediction is solved",
            "predicts draws reliably",
            "the draw problem doesn't matter",
            "draw problem is solved",
        ):
            self.assertNotIn(forbidden, low)
        self.assertIn("does **not** solve draw classification", low.replace("*", "*"))

    def test_no_statistical_significance_claim(self):
        low = self.text.lower()
        self.assertNotIn("statistically significant", low)
        self.assertIn("no significance test", low)

    def test_no_epsilon_invented(self):
        low = self.text.lower()
        self.assertIn("no epsilon threshold exists", low)

    def test_states_no_causal_claim(self):
        self.assertIn("no causal claim", self.text.lower())


@unittest.skipUnless(_REVIEW5B.exists() and _P4C.exists(), "artifacts absent")
class TestDrawFiguresMatchPhase4CArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = _REVIEW5B.read_text(encoding="utf-8")
        cls.d = json.loads(_P4C.read_text(encoding="utf-8"))

    def test_draw_metrics_match_artifact(self):
        v1 = self.d["arms"]["v1_logistic_regression"]["per_class"]["D"]
        b = self.d["arms"]["model_b_logistic_regression"]["per_class"]["D"]
        self.assertEqual(v1["tp"], 0)
        self.assertEqual(b["tp"], 8)
        self.assertEqual(v1["support"], 445)
        self.assertEqual(b["support"], 445)
        self.assertAlmostEqual(b["recall"], 0.017977528089887642, places=15)
        self.assertAlmostEqual(b["f1"], 0.03440860215053764, places=15)
        self.assertAlmostEqual(b["precision"], 0.4, places=15)

    def test_review_quotes_the_exact_draw_values(self):
        for v in ("0.017977528089887642", "0.03440860215053764", "445"):
            self.assertIn(v, self.text)

    @unittest.skipUnless(_PRED.exists(), "prediction CSVs absent")
    def test_probability_mass_claim_is_reproducible(self):
        b = pd.read_csv(_PRED / "model_b_logistic_regression_final_test.csv")
        # Σ p_draw close to actual draws is the load-bearing claim.
        self.assertAlmostEqual(float(b.p_draw.sum()), 446.5199, places=2)
        self.assertEqual(int((b.y_true == "D").sum()), 445)
        # Hard-class weakness must also hold.
        self.assertEqual(int((b.y_pred == "D").sum()), 20)
        # p_draw ceiling below 0.40 is the mechanism cited.
        self.assertLess(float(b.p_draw.max()), 0.40)


@unittest.skipUnless(_P5A.exists(), "Phase 5A artifact absent")
class TestCondition1RemainsResolved(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads(_P5A.read_text(encoding="utf-8"))

    def test_conclusion_is_retain_omission_supported(self):
        self.assertEqual(self.d["classification"]["conclusion"], "RETAIN_OMISSION_SUPPORTED")
        self.assertIn(self.d["classification"]["conclusion"], PERMITTED_CONDITION1)

    def test_case_is_worsens(self):
        self.assertEqual(self.d["classification"]["case"], "CASE_B_worsens_mean_log_loss")

    def test_delta_is_positive_meaning_feature_worsened(self):
        self.assertGreater(self.d["classification"]["mean_log_loss_delta_plus_minus_b"], 0)

    def test_phase5a_did_not_use_final_test(self):
        self.assertFalse(self.d["scope"]["final_test_used"])

    def test_phase5_review_records_condition1_resolved(self):
        text = _REVIEW5.read_text(encoding="utf-8")
        self.assertIn("RETAIN_OMISSION_SUPPORTED", text)
        self.assertIn("0.9996821579734537", text)

    def test_phase5a_results_document_exists(self):
        # Previously a dangling cross-reference: the Phase 5 review cited
        # this document before it had been written. It must exist so
        # Condition 1's closure is recorded in writing at the cited path.
        self.assertTrue(_RESULTS5A.exists(), "docs/PHASE5A_DROPPED_FEATURE_RESULTS.md missing")

    @unittest.skipUnless(_RESULTS5A.exists(), "Phase 5A results doc absent")
    def test_phase5a_document_matches_authoritative_json(self):
        doc = _RESULTS5A.read_text(encoding="utf-8")
        s = self.d["summary_by_arm"]
        c = self.d["classification"]
        # Headline values must be transcribed exactly from the artifact.
        for value in (
            repr(s["model_b"]["mean_log_loss"]).strip("'"),
            repr(s["model_b_plus_home_advantage"]["mean_log_loss"]).strip("'"),
            str(c["mean_log_loss_delta_plus_minus_b"]),
            c["case"],
            c["conclusion"],
        ):
            self.assertIn(value, doc, f"Phase 5A doc missing authoritative value: {value}")

    @unittest.skipUnless(_RESULTS5A.exists(), "Phase 5A results doc absent")
    def test_phase5a_document_records_validation_only_scope(self):
        doc = _RESULTS5A.read_text(encoding="utf-8")
        self.assertIn("final_test_used", doc)
        self.assertIn("`false`", doc.lower())
        self.assertIn("v1.0", doc)

    @unittest.skipUnless(_RESULTS5A.exists(), "Phase 5A results doc absent")
    def test_phase5a_document_makes_no_significance_or_causal_claim(self):
        low = _RESULTS5A.read_text(encoding="utf-8").lower()
        self.assertNotIn("statistically significant", low)
        self.assertIn("no statistical significance claim is made", low)
        self.assertIn("no causal claim is made", low)

    @unittest.skipUnless(_RESULTS5A.exists(), "Phase 5A results doc absent")
    def test_phase5a_document_records_mixed_secondary_evidence(self):
        # The conclusion rests on the primary metric while secondary
        # metrics leaned the other way; that nuance must not be smoothed
        # over in the written record.
        low = _RESULTS5A.read_text(encoding="utf-8").lower()
        self.assertIn("mixed", low)
        self.assertIn("1 of 3 folds", low)


@unittest.skipUnless(_REVIEW5.exists(), "Phase 5 review absent")
class TestPhase5StatusUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = _REVIEW5.read_text(encoding="utf-8")

    def test_declared_status_is_a_permitted_value(self):
        import re
        # Character class must include digits: status names contain "V2".
        m = re.search(r"## STATUS: [`]([A-Z0-9_]+)[`]", self.text)
        self.assertIsNotNone(m, "no declared status heading found")
        self.assertIn(m.group(1), PERMITTED_V2_STATUS)

    def test_status_history_preserved_not_overwritten(self):
        # The original conditional status must remain visible as history.
        self.assertIn("CONDITIONAL_V2_CANDIDATE", self.text)
        self.assertIn("Status history", self.text)

    def test_approval_does_not_authorize_production(self):
        low = self.text.lower()
        self.assertIn("does not authorize production deployment", low)
        self.assertIn("does not create v2", low)

    def test_condition2_acceptance_is_scoped(self):
        low = self.text.lower()
        self.assertIn("scoped", low)
        self.assertIn("must be reopened", low)


@unittest.skipUnless(_SPEC_DRAFT.exists(), "spec draft absent")
class TestSpecDraftStatusCurrent(unittest.TestCase):
    """The spec draft's governance language must reflect the closed
    conditions while remaining explicitly non-production."""

    @classmethod
    def setUpClass(cls):
        cls.text = _SPEC_DRAFT.read_text(encoding="utf-8")

    def test_still_marked_draft_not_production_authorized(self):
        self.assertIn("DRAFT — NOT PRODUCTION AUTHORIZED", self.text)

    def test_declares_approved_candidate_status(self):
        self.assertIn("APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION", self.text)

    def test_both_conditions_recorded_closed(self):
        self.assertIn("Condition 1 — dropped V1 feature: CLOSED", self.text)
        self.assertIn("RETAIN_OMISSION_SUPPORTED", self.text)
        self.assertIn("Condition 2 — draw-class limitation: CLOSED", self.text)
        self.assertIn("DRAW_LIMITATION_ACCEPTED", self.text)

    def test_no_stale_unresolved_condition_language(self):
        low = self.text.lower()
        self.assertNotIn("two unresolved conditions", low)
        self.assertNotIn("until both conditions are resolved", low)
        self.assertNotIn("feature contract (provisional", low)

    def test_history_preserved_not_rewritten(self):
        # The prior conditional status must remain visible as history.
        self.assertIn("CONDITIONAL_V2_CANDIDATE", self.text)

    def test_approval_scope_is_limited(self):
        low = self.text.lower()
        self.assertIn("does not authorize production deployment", low)
        self.assertIn("does not create v2", low)

    def test_feature_contract_and_estimator_unchanged(self):
        from models.ablation import MODEL_B_COLUMNS
        self.assertEqual(len(MODEL_B_COLUMNS), 80)
        self.assertIn("| **Total** | **80** |", self.text)
        self.assertIn("| `C` | **1.0** |", self.text)
        self.assertIn("| `max_iter` | **2000** |", self.text)
        self.assertIn("| `random_state` | **0** |", self.text)

    def test_omitted_feature_still_recorded_as_omitted(self):
        self.assertIn("league_home_advantage_season", self.text)
        self.assertIn("must not be added back", self.text)


class TestNoProductionOrV2SideEffects(unittest.TestCase):
    def test_model_version_still_v1(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_no_v2_module_or_artifact(self):
        for p in Path("src/models").glob("*.py"):
            self.assertNotIn("v2", p.stem.lower())
        for p in Path("data/audit").glob("*"):
            self.assertNotIn("v2", p.name.lower())

    def test_no_new_final_test_artifact_created(self):
        top = sorted(p.name for p in Path("data/audit").glob("*phase4c*"))
        self.assertEqual(
            top,
            ["phase4c_final_comparison.json", "phase4c_predictions", "phase4c_prerun_manifest.json"],
        )

    def test_no_phase5b_result_artifact_created(self):
        # Phase 5B is a document-only governance review.
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("phase5b", p.name.lower())

    def test_locked_artifacts_unchanged(self):
        expected = {
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
            "data/audit/phase3_model_comparison.json": "616279914b2730749d52eae15b5f96b9",
            "data/audit/phase4a_ablation_comparison.json": "075b0686bce20bfce9f7289fa37076c0",
            "data/audit/phase4b_robustness_comparison.json": "97d0f8b8674c9bf8598b6c3d6b7c825c",
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
            "data/audit/phase4c_prerun_manifest.json": "4f166d1826edbf7bfe1e022639a8b1f4",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/ablation.py": "9bf8bf4a1f332f8bb47d640a72384ed7",
            "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
        }
        for rel, exp in expected.items():
            actual = hashlib.md5(Path(rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"locked artifact modified: {rel}")


if __name__ == "__main__":
    unittest.main()
