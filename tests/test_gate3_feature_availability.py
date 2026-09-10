"""Gate 3 — Feature availability tests.

Gate 3 requirement (docs/PHASE5_V2_MODEL_SPEC_DRAFT.md):
    "All 80 columns present with null rates in documented bounds."
    Verification: pre-inference coverage check against §10.
    Pass: all present; no all-null column; null rates within bounds.
    Stop: missing column, all-null column, or out-of-bounds null rate.

Covers the check's logic (synthetic, always runs) and its behaviour on
the real authoritative dataset across every documented partition.

Trains nothing, tunes nothing, calibrates nothing. 2025/26 is inspected
for schema/coverage only, exactly as §10 requires — never as a model
evaluation.
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
    KNOWN_COMPETITION_IDS,
    MAX_FINAL_SEASON_NULL_RATE,
    MAX_HISTORICAL_NULL_RATE,
    check_feature_availability,
)

_REAL_DB = Path("data/processed/features.db")
_DATA_SKIP = "data/processed/features.db not present in this environment"


def _synthetic_batch(n=5):
    data = {c: [float(i + 1) for i in range(n)] for c in CANDIDATE_FEATURE_COLUMNS}
    data["competition_id"] = [sorted(KNOWN_COMPETITION_IDS)[i % 5] for i in range(n)]
    return pd.DataFrame(data)


class TestGate3DocumentedBounds(unittest.TestCase):
    def test_bounds_are_traceable_to_spec_section_10(self):
        # §10: "<= ~7.7% observed historically; ~5.5% on 2025/26".
        # Encoded at a precision that admits the 0.077218 observation the
        # bound was written from (see module PRECISION NOTE).
        self.assertEqual(MAX_HISTORICAL_NULL_RATE, 0.078)
        self.assertEqual(MAX_FINAL_SEASON_NULL_RATE, 0.056)

    def test_known_competition_ids_match_spec(self):
        self.assertEqual(set(KNOWN_COMPETITION_IDS), {200, 419, 423, 477, 499})

    def test_precision_note_is_documented_in_source(self):
        src = Path("src/models/candidate_contract.py").read_text(encoding="utf-8")
        self.assertIn("PRECISION NOTE", src)
        self.assertIn("0.077218", src)


class TestGate3CheckLogic(unittest.TestCase):
    """Synthetic tests: the check must actually detect each stop condition."""

    def test_clean_batch_passes(self):
        r = check_feature_availability(_synthetic_batch())
        self.assertTrue(r["passed"], r["failures"])
        self.assertEqual(r["missing_columns"], [])
        self.assertEqual(r["all_null_columns"], [])
        self.assertEqual(r["constant_columns"], [])

    def test_missing_column_fails(self):
        batch = _synthetic_batch().drop(columns=["strength_diff"])
        r = check_feature_availability(batch)
        self.assertFalse(r["passed"])
        self.assertIn("strength_diff", r["missing_columns"])

    def test_all_null_column_fails(self):
        batch = _synthetic_batch()
        batch["strength_diff"] = np.nan
        r = check_feature_availability(batch)
        self.assertFalse(r["passed"])
        self.assertEqual(r["all_null_columns"], ["strength_diff"])

    def test_out_of_bounds_null_rate_fails(self):
        batch = _synthetic_batch(n=10)
        batch.loc[batch.index[:5], "home_points_last5"] = np.nan  # 50% null
        r = check_feature_availability(batch)
        self.assertFalse(r["passed"])
        self.assertIn("home_points_last5", r["out_of_bounds_columns"])

    def test_constant_column_is_flagged_as_failure(self):
        batch = _synthetic_batch()
        batch["home_points_last5"] = 1.0
        r = check_feature_availability(batch)
        self.assertFalse(r["passed"])
        self.assertIn("home_points_last5", r["constant_columns"])

    def test_null_competition_id_fails(self):
        batch = _synthetic_batch()
        batch.loc[batch.index[0], "competition_id"] = np.nan
        r = check_feature_availability(batch)
        self.assertFalse(r["passed"])

    def test_unseen_league_is_flagged_but_not_a_failure(self):
        # §10: an unseen league "must be flagged (behaviour is
        # degraded-not-failed)". So the batch still passes, but the
        # unseen ID is surfaced for the caller to act on.
        batch = _synthetic_batch()
        batch.loc[batch.index[0], "competition_id"] = 999999
        r = check_feature_availability(batch)
        self.assertTrue(r["passed"], r["failures"])
        self.assertTrue(r["competition_id"]["unseen_flagged"])
        self.assertEqual(r["competition_id"]["unseen_ids"], [999999])

    def test_known_leagues_are_not_flagged_unseen(self):
        r = check_feature_availability(_synthetic_batch())
        self.assertFalse(r["competition_id"]["unseen_flagged"])
        self.assertEqual(r["competition_id"]["unseen_ids"], [])

    def test_check_reports_all_findings_not_just_the_first(self):
        batch = _synthetic_batch(n=10)
        batch["strength_diff"] = np.nan                       # all-null
        batch["home_points_last5"] = 1.0                      # constant
        r = check_feature_availability(batch)
        self.assertFalse(r["passed"])
        self.assertGreaterEqual(len(r["failures"]), 2)

    def test_check_never_mutates_input(self):
        batch = _synthetic_batch()
        before = batch.copy()
        check_feature_availability(batch)
        pd.testing.assert_frame_equal(batch, before)

    def test_check_reports_worst_column_and_rate(self):
        batch = _synthetic_batch(n=10)
        batch.loc[batch.index[:3], "away_points_last10"] = np.nan
        r = check_feature_availability(batch, max_null_rate=0.5)
        self.assertEqual(r["max_null_rate_column"], "away_points_last10")
        self.assertAlmostEqual(r["max_null_rate_observed"], 0.3)


@unittest.skipUnless(_REAL_DB.exists(), _DATA_SKIP)
class TestGate3AgainstRealPartitions(unittest.TestCase):
    """Empirical Gate 3 verification on the authoritative dataset."""

    @classmethod
    def setUpClass(cls):
        from models.data import load_supervised_dataset
        from models.splits import final_split, iter_walk_forward_folds
        cls.dataset = load_supervised_dataset(_REAL_DB)
        cls.folds = list(iter_walk_forward_folds(cls.dataset))
        cls.final_train, cls.final_test = final_split(cls.dataset)

    def test_all_eighty_columns_present_in_features_db(self):
        missing = [c for c in CANDIDATE_FEATURE_COLUMNS if c not in self.dataset.X.columns]
        self.assertEqual(missing, [])

    def test_every_walk_forward_partition_passes(self):
        for fold, train_ds, val_ds in self.folds:
            for label, part in ((f"{fold.name}_train", train_ds), (f"{fold.name}_val", val_ds)):
                r = check_feature_availability(part.X, MAX_HISTORICAL_NULL_RATE, label)
                self.assertTrue(r["passed"], f"{label}: {r['failures']}")

    def test_final_train_partition_passes(self):
        r = check_feature_availability(self.final_train.X, MAX_HISTORICAL_NULL_RATE, "final_train")
        self.assertTrue(r["passed"], r["failures"])

    def test_final_season_coverage_within_tighter_bound(self):
        # Schema/coverage inspection only, per §10 -- NOT a model evaluation.
        r = check_feature_availability(self.final_test.X, MAX_FINAL_SEASON_NULL_RATE, "final_test_2025_26")
        self.assertTrue(r["passed"], r["failures"])
        self.assertLessEqual(r["max_null_rate_observed"], MAX_FINAL_SEASON_NULL_RATE)

    def test_no_all_null_contract_column_in_any_partition(self):
        parts = [("full", self.dataset), ("final_train", self.final_train), ("final_test", self.final_test)]
        for fold, tr, va in self.folds:
            parts += [(f"{fold.name}_train", tr), (f"{fold.name}_val", va)]
        for label, part in parts:
            all_null = [c for c in CANDIDATE_FEATURE_COLUMNS if part.X[c].isna().all()]
            self.assertEqual(all_null, [], f"{label} has all-null contract column(s): {all_null}")

    def test_no_constant_contract_column_in_any_partition(self):
        parts = [("full", self.dataset), ("final_train", self.final_train), ("final_test", self.final_test)]
        for fold, tr, va in self.folds:
            parts += [(f"{fold.name}_train", tr), (f"{fold.name}_val", va)]
        for label, part in parts:
            const = [
                c for c in CANDIDATE_FEATURE_COLUMNS
                if not part.X[c].isna().all() and part.X[c].dropna().nunique() <= 1
            ]
            self.assertEqual(const, [], f"{label} has constant contract column(s): {const}")

    def test_worst_historical_null_rate_is_the_documented_observation(self):
        worst = 0.0
        for fold, tr, va in self.folds:
            for part in (tr, va):
                worst = max(worst, max(float(part.X[c].isna().mean()) for c in CANDIDATE_FEATURE_COLUMNS))
        worst = max(worst, max(float(self.final_train.X[c].isna().mean()) for c in CANDIDATE_FEATURE_COLUMNS))
        # This is the 0.0772 figure the spec's "~7.7%" was written from.
        self.assertAlmostEqual(worst, 0.077218, places=5)
        self.assertLessEqual(worst, MAX_HISTORICAL_NULL_RATE)

    def test_competition_id_is_exactly_the_known_five_everywhere(self):
        parts = [("full", self.dataset), ("final_train", self.final_train), ("final_test", self.final_test)]
        for fold, tr, va in self.folds:
            parts += [(f"{fold.name}_train", tr), (f"{fold.name}_val", va)]
        for label, part in parts:
            r = check_feature_availability(part.X, MAX_HISTORICAL_NULL_RATE, label)
            ci = r["competition_id"]
            self.assertEqual(set(ci["observed_ids"]), set(KNOWN_COMPETITION_IDS), label)
            self.assertEqual(ci["unseen_ids"], [], label)
            self.assertEqual(ci["null_count"], 0, label)


class TestGate3NoSideEffects(unittest.TestCase):
    def test_check_does_not_train_or_predict(self):
        src = inspect.getsource(check_feature_availability)
        for forbidden in (".fit(", ".predict", "LogisticRegression", "sklearn"):
            self.assertNotIn(forbidden, src)

    def test_contract_module_still_imports_no_estimator(self):
        tree = ast.parse(Path("src/models/candidate_contract.py").read_text(encoding="utf-8"))
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        for forbidden in ("sklearn", "LogisticRegression"):
            self.assertNotIn(forbidden, imported)
        self.assertFalse(any("sklearn" in m for m in modules))

    def test_contract_module_cannot_reach_final_split(self):
        tree = ast.parse(Path("src/models/candidate_contract.py").read_text(encoding="utf-8"))
        imported = {a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for forbidden in ("final_split", "FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS"):
            self.assertNotIn(forbidden, imported)

    def test_model_version_unchanged(self):
        from models.config import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_no_gate3_artifact_created(self):
        for p in Path("data/audit").rglob("*"):
            self.assertNotIn("gate3", p.name.lower())

    def test_contract_itself_unchanged_by_gate3(self):
        from models.ablation import MODEL_B_COLUMNS
        self.assertEqual(tuple(CANDIDATE_FEATURE_COLUMNS), tuple(MODEL_B_COLUMNS))
        self.assertEqual(len(CANDIDATE_FEATURE_COLUMNS), 80)
        self.assertNotIn("league_home_advantage_season", CANDIDATE_FEATURE_COLUMNS)

    def test_v1_and_phase_artifacts_unchanged(self):
        expected = {
            "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
            "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
            "src/models/config.py": "c2ed32cb53ec34199fd245624afea4dd",
            "src/models/train.py": "21425459195311492f49e73f5ae38fe0",
            "src/models/ablation.py": "9bf8bf4a1f332f8bb47d640a72384ed7",
            "src/models/data.py": "b78e30eb45dbc46621c0160a188ce981",
            "src/models/splits.py": "8b7991ab3739c7d4daa2bf1998163da4",
            "src/models/evaluate.py": "4e9d9313867d47a19001a383a301c2fe",
            "data/audit/phase4c_final_comparison.json": "effd9e54130b2bc5aaf51cd643962396",
            "data/audit/phase5a_dropped_feature_comparison.json": "3ceb7090f9e19c106d88eaa7a8848818",
        }
        for rel, exp in expected.items():
            actual = hashlib.md5(Path(rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"locked artifact modified: {rel}")


if __name__ == "__main__":
    unittest.main()
