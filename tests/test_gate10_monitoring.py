"""Gate 10 -- monitoring acceptance tests.

Covers the 24 required areas. Design rules followed throughout:

  - No self-referential source-text greps. Where a test must prove that
    the monitoring code does NOT do something (train, touch 2025/26, use
    the rejected +/-0.03 heuristic, merge ceiling with drift), it inspects
    the parsed AST of the monitoring modules or the imported objects
    themselves -- never this test file's own text, and never a substring
    search that its own literals could satisfy.
  - Positive AND negative controls for every threshold: each boundary is
    asserted from both sides, so a test cannot pass merely because the
    code always returns PASS (or always ALERT).
  - Frozen-state assertions read the pinned checksums from
    data/audit/phase4c_prerun_manifest.json rather than restating hashes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import _pathfix  # noqa: F401  -- import-path bootstrap

import numpy as np
import pandas as pd

import monitoring as mon
from models.ablation import (
    COMPETITION_ID_COLUMN,
    FORM_COLUMNS,
    GOALS_CORE_COLUMNS,
    MODEL_B_COLUMNS,
    SHOTS_CORE_COLUMNS,
    SHOTS_ON_CORE_COLUMNS,
    STRENGTH_COLUMNS,
    XG_CORE_COLUMNS,
)
from models.candidate_contract import (
    KNOWN_COMPETITION_IDS,
    MAX_FINAL_SEASON_NULL_RATE,
    MAX_HISTORICAL_NULL_RATE,
)
from models.config import MODEL_VERSION
from monitoring.config import (
    NON_TEST_PARTITION_NULL_RATES,
    REQUIRED_SIGNAL_COVERAGE,
    SIGNAL_S1_CEILING,
    SIGNAL_S1_DRIFT,
    SIGNAL_S2_UNSEEN_COMPETITION,
    SIGNAL_S3_CLASS_DISTRIBUTION,
    SIGNAL_S4_CALIBRATION,
    SIGNAL_S5_PROBABILITY_VALIDITY,
    SIGNAL_S6_DRAW_BEHAVIOUR,
    Severity,
    Status,
)
from monitoring.events import REQUIRED_EVENT_FIELDS, MonitoringEvent, SchemaViolation, validate_event
from monitoring.signals import (
    MonitoringInputError,
    build_null_rate_baseline,
    evaluate_calibration_post_outcome,
    evaluate_class_distribution,
    evaluate_draw_behaviour,
    evaluate_null_rate_ceiling,
    evaluate_null_rate_drift,
    evaluate_unseen_competition_ids,
    load_stored_base_rate_reference,
    observed_family_null_rates,
    select_eligible_observations,
)
from monitoring.store import JsonlEventStore
from monitoring.validity import (
    count_invalid_rows,
    monitored_predict_candidate,
    record_validity_failure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MONITORING_DIR = REPO_ROOT / "src" / "monitoring"
MANIFEST = REPO_ROOT / "data" / "audit" / "phase4c_prerun_manifest.json"
CANDIDATE_REVIEW = REPO_ROOT / "docs" / "PHASE5_V2_CANDIDATE_REVIEW.md"

BASE_RATES = {"H": 0.4, "D": 0.25, "A": 0.35}


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def full_contract_frame(n: int = 100, null_spec: dict[str, float] | None = None) -> pd.DataFrame:
    """A batch containing every contract column, optionally with nulls.

    `null_spec` maps a column name to the fraction of rows to null out.
    """
    frame = pd.DataFrame({c: np.ones(n, dtype=float) for c in MODEL_B_COLUMNS})
    frame["competition_id"] = 200
    for column, fraction in (null_spec or {}).items():
        k = int(round(fraction * n))
        frame.loc[frame.index[:k], column] = np.nan
    return frame


def probability_matrix(mass: dict[str, float], n: int = 50) -> np.ndarray:
    return np.tile(np.array([[mass["H"], mass["D"], mass["A"]]], dtype=float), (n, 1))


def settled_records(
    n: int,
    mass: dict[str, float],
    outcomes: list[str],
    settled_at: datetime,
    maturity_days: float = 10.0,
) -> list[dict]:
    predicted_at = settled_at - timedelta(days=maturity_days)
    return [
        {
            "predicted_at": predicted_at,
            "settled_at": settled_at,
            "probabilities": dict(mass),
            "outcome": outcomes[i % len(outcomes)],
            "fixture_id": 1000 + i,
        }
        for i in range(n)
    ]


def monitoring_modules() -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for path in sorted(MONITORING_DIR.glob("*.py")):
        trees[path.name] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return trees


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def names_used(node: ast.AST) -> set[str]:
    """Identifier and attribute names appearing in executable positions.

    Docstrings and other string literals are excluded by construction:
    only Name.id and Attribute.attr are collected.
    """
    used: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            used.add(child.id)
        elif isinstance(child, ast.Attribute):
            used.add(child.attr)
    return used


def numeric_constants(node: ast.AST) -> set[float]:
    out: set[float] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)) and not isinstance(child.value, bool):
            out.add(float(child.value))
    return out


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# =====================================================================
# 1. Family mapping exactness
# =====================================================================
class TestGate10FamilyMapping(unittest.TestCase):
    def test_six_families_named_exactly(self):
        self.assertEqual(
            tuple(mon.FAMILY_COLUMNS),
            ("goals_core", "form", "strength", "competition_id", "shots_core", "shots_on_core"),
        )

    def test_family_counts_match_spec(self):
        expected = {
            "goals_core": 18, "form": 20, "strength": 5,
            "competition_id": 1, "shots_core": 18, "shots_on_core": 18,
        }
        self.assertEqual({k: len(v) for k, v in mon.FAMILY_COLUMNS.items()}, expected)
        self.assertEqual(sum(expected.values()), 80)

    def test_families_are_the_frozen_ablation_tuples(self):
        """Derived from the frozen contract, not restated as literals."""
        self.assertEqual(mon.FAMILY_COLUMNS["goals_core"], tuple(GOALS_CORE_COLUMNS))
        self.assertEqual(mon.FAMILY_COLUMNS["form"], tuple(FORM_COLUMNS))
        self.assertEqual(mon.FAMILY_COLUMNS["strength"], tuple(STRENGTH_COLUMNS))
        self.assertEqual(mon.FAMILY_COLUMNS["competition_id"], tuple(COMPETITION_ID_COLUMN))
        self.assertEqual(mon.FAMILY_COLUMNS["shots_core"], tuple(SHOTS_CORE_COLUMNS))
        self.assertEqual(mon.FAMILY_COLUMNS["shots_on_core"], tuple(SHOTS_ON_CORE_COLUMNS))

    def test_families_partition_the_contract_exactly(self):
        flat = [c for cols in mon.FAMILY_COLUMNS.values() for c in cols]
        self.assertEqual(len(flat), len(set(flat)), "families must be pairwise disjoint")
        self.assertEqual(set(flat), set(MODEL_B_COLUMNS))
        self.assertEqual(tuple(flat), tuple(MODEL_B_COLUMNS), "order must match contract order")

    def test_no_unrelated_feature_group_included(self):
        flat = {c for cols in mon.FAMILY_COLUMNS.values() for c in cols}
        for column in XG_CORE_COLUMNS:
            self.assertNotIn(column, flat)
        for column in ("league_home_advantage_season", "league_mean_goals_per_team_match_season",
                       "label_result", "label_home_goals", "label_away_goals", "fixture_id"):
            self.assertNotIn(column, flat)

    def test_import_time_partition_guard_exists_and_rejects_drift(self):
        from monitoring import config as mon_config

        original = dict(mon_config.FAMILY_COLUMNS)
        try:
            mon_config.FAMILY_COLUMNS["goals_core"] = original["goals_core"][:-1]
            with self.assertRaises(RuntimeError):
                mon_config._verify_family_partition()
        finally:
            mon_config.FAMILY_COLUMNS.clear()
            mon_config.FAMILY_COLUMNS.update(original)
        mon_config._verify_family_partition()  # restored state is valid again


# =====================================================================
# 2. Null-rate baseline calculation
# =====================================================================
class TestGate10NullRateBaseline(unittest.TestCase):
    def test_baseline_evidence_matches_recorded_document(self):
        """Transcription is verified against the recorded Part 4 table."""
        text = CANDIDATE_REVIEW.read_text(encoding="utf-8")
        row_labels = {
            "fold_1_train": "fold_1 train",
            "fold_2_train": "fold_2 train",
            "fold_3_train": "fold_3 train",
            "final_train": "final train",
        }
        families = ["goals_core", "form", "strength", "competition_id", "shots_core", "shots_on_core"]
        for key, label in row_labels.items():
            pattern = re.compile(r"^\|\s*(?:\*\*)?" + re.escape(label) + r".*$", re.MULTILINE)
            match = pattern.search(text)
            self.assertIsNotNone(match, f"row {label!r} not found in the recorded table")
            cells = [c.strip().strip("*") for c in match.group(0).strip().strip("|").split("|")]
            values = [float(c) for c in cells[1:7]]
            for family, recorded in zip(families, values):
                self.assertAlmostEqual(
                    NON_TEST_PARTITION_NULL_RATES[key][family], recorded, places=6,
                    msg=f"{key}/{family} transcription differs from the recorded document",
                )

    def test_no_test_season_partition_in_baseline_evidence(self):
        for name in NON_TEST_PARTITION_NULL_RATES:
            self.assertNotIn("test", name.lower())
            self.assertNotIn("2025", name)
        self.assertEqual(len(NON_TEST_PARTITION_NULL_RATES), 4)

    def test_aggregation_is_required_with_no_default(self):
        with self.assertRaises(TypeError):
            build_null_rate_baseline()  # type: ignore[call-arg]

    def test_max_and_mean_aggregations(self):
        max_bl = build_null_rate_baseline("max")
        mean_bl = build_null_rate_baseline("mean")
        self.assertAlmostEqual(max_bl["form"], 0.0764)
        self.assertAlmostEqual(
            mean_bl["form"], (0.0764 + 0.0555 + 0.0444 + 0.0374) / 4, places=10
        )
        for family in mon.FAMILY_COLUMNS:
            self.assertGreaterEqual(max_bl[family] + 1e-12, mean_bl[family])

    def test_single_aggregation_requires_exactly_one_partition(self):
        self.assertAlmostEqual(build_null_rate_baseline("single", ["final_train"])["goals_core"], 0.0543)
        with self.assertRaises(MonitoringInputError):
            build_null_rate_baseline("single", ["final_train", "fold_1_train"])

    def test_unknown_aggregation_and_partition_rejected(self):
        with self.assertRaises(MonitoringInputError):
            build_null_rate_baseline("median")
        for forbidden in ("final_test", "2025/2026", "test", "final_test_2025_26"):
            with self.assertRaises(MonitoringInputError):
                build_null_rate_baseline("single", [forbidden])

    def test_observed_family_null_rates_uses_family_maximum(self):
        frame = full_contract_frame(100, {"home_points_last5": 0.10, "home_win_rate_last5": 0.02})
        observed = observed_family_null_rates(frame)
        self.assertAlmostEqual(observed["form"], 0.10, places=10)
        self.assertAlmostEqual(observed["goals_core"], 0.0, places=10)

    def test_missing_contract_column_raises(self):
        frame = full_contract_frame(10).drop(columns=["home_points_last5"])
        with self.assertRaises(MonitoringInputError):
            observed_family_null_rates(frame)


# =====================================================================
# 3. +2pp drift boundary (positive and negative controls)
# =====================================================================
class TestGate10DriftBoundary(unittest.TestCase):
    def setUp(self):
        self.baseline = {f: 0.0 for f in mon.FAMILY_COLUMNS}

    def test_threshold_value_is_two_percentage_points(self):
        self.assertEqual(mon.DRIFT_ALERT_PP, 2.0)

    def test_just_below_threshold_passes(self):
        frame = full_contract_frame(1000, {"home_points_last5": 0.019})
        event = evaluate_null_rate_drift(frame, self.baseline)
        self.assertEqual(event.status, Status.PASS)

    def test_exactly_at_threshold_passes(self):
        """Boundary rule: exactly +2.00 pp is PASS, not ALERT."""
        frame = full_contract_frame(1000, {"home_points_last5": 0.020})
        event = evaluate_null_rate_drift(frame, self.baseline)
        self.assertAlmostEqual(event.metrics["deviation_pp_by_family"]["form"], 2.0, places=9)
        self.assertEqual(event.status, Status.PASS)

    def test_just_above_threshold_alerts(self):
        frame = full_contract_frame(1000, {"home_points_last5": 0.021})
        event = evaluate_null_rate_drift(frame, self.baseline)
        self.assertEqual(event.status, Status.ALERT)
        self.assertIn("form", event.metrics["breaches_pp"])

    def test_improvement_below_baseline_never_alerts(self):
        """Direction control: only above-baseline deviation can alert."""
        baseline = {f: 0.50 for f in mon.FAMILY_COLUMNS}
        frame = full_contract_frame(100)
        event = evaluate_null_rate_drift(frame, baseline)
        self.assertEqual(event.status, Status.PASS)
        self.assertTrue(all(v < 0 for v in event.metrics["deviation_pp_by_family"].values()))

    def test_incomplete_baseline_rejected(self):
        with self.assertRaises(MonitoringInputError):
            evaluate_null_rate_drift(full_contract_frame(10), {"form": 0.0})


# =====================================================================
# 4. Absolute availability ceiling stays separate from drift
# =====================================================================
class TestGate10CeilingSeparateFromDrift(unittest.TestCase):
    def test_ceilings_are_imported_not_redefined(self):
        self.assertEqual(mon.config.CEILING_HISTORICAL, MAX_HISTORICAL_NULL_RATE)
        self.assertEqual(mon.config.CEILING_FINAL_SEASON, MAX_FINAL_SEASON_NULL_RATE)
        self.assertEqual(MAX_HISTORICAL_NULL_RATE, 0.078)
        self.assertEqual(MAX_FINAL_SEASON_NULL_RATE, 0.056)

    def test_two_distinct_signal_ids(self):
        self.assertNotEqual(SIGNAL_S1_CEILING, SIGNAL_S1_DRIFT)
        frame = full_contract_frame(100)
        self.assertEqual(evaluate_null_rate_ceiling(frame).signal_id, SIGNAL_S1_CEILING)
        baseline = build_null_rate_baseline("max")
        self.assertEqual(evaluate_null_rate_drift(frame, baseline).signal_id, SIGNAL_S1_DRIFT)

    def test_ceiling_function_never_references_the_drift_threshold(self):
        """AST proof, not a text search."""
        tree = monitoring_modules()["signals.py"]
        used = names_used(function_node(tree, "evaluate_null_rate_ceiling"))
        self.assertNotIn("DRIFT_ALERT_PP", used)

    def test_drift_function_never_references_the_ceilings(self):
        tree = monitoring_modules()["signals.py"]
        used = names_used(function_node(tree, "evaluate_null_rate_drift"))
        self.assertNotIn("CEILING_HISTORICAL", used)
        self.assertNotIn("CEILING_FINAL_SEASON", used)

    def test_ceiling_violation_without_drift_alert(self):
        """A batch at baseline level but above the ceiling: ceiling alerts, drift does not."""
        baseline = {f: 0.30 for f in mon.FAMILY_COLUMNS}
        frame = full_contract_frame(100, {"home_points_last5": 0.30})
        self.assertEqual(evaluate_null_rate_ceiling(frame).status, Status.ALERT)
        self.assertEqual(evaluate_null_rate_drift(frame, baseline).status, Status.PASS)

    def test_drift_alert_while_under_the_ceiling(self):
        """The recorded `form` case: 0.0074 -> 0.070 drifts but stays under 0.078."""
        baseline = {f: 0.0074 for f in mon.FAMILY_COLUMNS}
        frame = full_contract_frame(1000, {"home_points_last5": 0.070})
        self.assertEqual(evaluate_null_rate_ceiling(frame).status, Status.PASS)
        self.assertEqual(evaluate_null_rate_drift(frame, baseline).status, Status.ALERT)

    def test_ceiling_metrics_declare_their_comparison_kind(self):
        frame = full_contract_frame(50)
        self.assertEqual(evaluate_null_rate_ceiling(frame).metrics["comparison"], "absolute_ceiling")
        self.assertEqual(
            evaluate_null_rate_drift(frame, build_null_rate_baseline("max")).metrics["comparison"],
            "drift_from_baseline",
        )


# =====================================================================
# 5. Unseen competition_id -> DEGRADED, never a hard failure
# =====================================================================
class TestGate10UnseenCompetition(unittest.TestCase):
    def test_known_set_is_imported_unchanged(self):
        self.assertEqual(set(KNOWN_COMPETITION_IDS), {200, 419, 423, 477, 499})

    def test_all_known_ids_pass(self):
        event = evaluate_unseen_competition_ids([200, 419, 423, 477, 499])
        self.assertEqual(event.status, Status.PASS)
        self.assertEqual(event.metrics["unseen_ids"], [])

    def test_unseen_id_emits_degraded_and_is_flagged(self):
        event = evaluate_unseen_competition_ids([200, 419, 777])
        self.assertEqual(event.status, Status.DEGRADED)
        self.assertEqual(event.severity, Severity.WARNING)
        self.assertEqual(event.metrics["unseen_ids"], [777])
        self.assertEqual(event.metrics["unseen_row_count"], 1)

    def test_unseen_id_never_hard_fails_and_never_raises(self):
        for ids in ([777], [777, 888, 200], list(range(900, 950))):
            event = evaluate_unseen_competition_ids(ids)
            self.assertNotEqual(event.status, Status.HARD_FAIL)
            self.assertEqual(event.status, Status.DEGRADED)

    def test_null_competition_id_counted_separately_and_does_not_hard_fail(self):
        event = evaluate_unseen_competition_ids([200, None, float("nan")])
        self.assertEqual(event.metrics["null_competition_id_count"], 2)
        self.assertNotEqual(event.status, Status.HARD_FAIL)

    def test_no_numeric_threshold_is_applied(self):
        """One occurrence is as flagged as fifty: the rule is categorical."""
        one = evaluate_unseen_competition_ids([200] * 99 + [777])
        many = evaluate_unseen_competition_ids([777] * 100)
        self.assertEqual(one.status, many.status)


# =====================================================================
# 6 & 7. Signal 3: probability mass, +/-5pp per class
# =====================================================================
class TestGate10ClassDistribution(unittest.TestCase):
    def test_mean_probability_mass_is_computed_correctly(self):
        P = np.array([[0.5, 0.2, 0.3], [0.3, 0.3, 0.4]])
        event = evaluate_class_distribution(P, {"H": 0.4, "D": 0.25, "A": 0.35})
        mass = event.metrics["mean_predicted_probability_mass"]
        self.assertAlmostEqual(mass["H"], 0.4, places=10)
        self.assertAlmostEqual(mass["D"], 0.25, places=10)
        self.assertAlmostEqual(mass["A"], 0.35, places=10)
        self.assertEqual(event.status, Status.PASS)

    def test_threshold_value_is_five_percentage_points(self):
        self.assertEqual(mon.CLASS_DISTRIBUTION_TOLERANCE_PP, 5.0)

    def test_just_inside_threshold_passes(self):
        event = evaluate_class_distribution(
            probability_matrix({"H": 0.4 + 0.0499, "D": 0.25, "A": 0.35 - 0.0499}), BASE_RATES
        )
        self.assertEqual(event.status, Status.PASS)

    def test_exactly_at_threshold_passes(self):
        event = evaluate_class_distribution(
            probability_matrix({"H": 0.45, "D": 0.25, "A": 0.30}), BASE_RATES
        )
        self.assertAlmostEqual(event.metrics["deviation_pp"]["H"], 5.0, places=9)
        self.assertEqual(event.status, Status.PASS)

    def test_just_above_threshold_alerts(self):
        event = evaluate_class_distribution(
            probability_matrix({"H": 0.4 + 0.0501, "D": 0.25, "A": 0.35 - 0.0501}), BASE_RATES
        )
        self.assertEqual(event.status, Status.ALERT)
        self.assertIn("H", event.metrics["breaches_pp"])

    def test_negative_deviation_also_alerts(self):
        event = evaluate_class_distribution(
            probability_matrix({"H": 0.30, "D": 0.25, "A": 0.45}), BASE_RATES
        )
        self.assertEqual(event.status, Status.ALERT)
        self.assertIn("H", event.metrics["breaches_pp"])
        self.assertIn("A", event.metrics["breaches_pp"])

    def test_classes_evaluated_independently(self):
        """A single class breaching is enough; the others stay clean."""
        event = evaluate_class_distribution(
            probability_matrix({"H": 0.4, "D": 0.31, "A": 0.29}), BASE_RATES
        )
        self.assertEqual(event.status, Status.ALERT)
        self.assertEqual(sorted(event.metrics["breaches_pp"]), ["A", "D"])
        self.assertNotIn("H", event.metrics["breaches_pp"])

    def test_argmax_is_diagnostic_only_and_cannot_change_status(self):
        """Two batches with identical mass but opposite argmax shares."""
        balanced = np.array([[0.4, 0.25, 0.35]] * 100)
        polarised = np.array([[0.8, 0.05, 0.15], [0.0, 0.45, 0.55]] * 50)
        a = evaluate_class_distribution(balanced, BASE_RATES)
        b = evaluate_class_distribution(polarised, BASE_RATES)
        for cls in ("H", "D", "A"):
            self.assertAlmostEqual(
                a.metrics["mean_predicted_probability_mass"][cls],
                b.metrics["mean_predicted_probability_mass"][cls], places=10,
            )
        self.assertNotEqual(
            a.metrics["diagnostic_only_argmax_share"], b.metrics["diagnostic_only_argmax_share"]
        )
        self.assertEqual(a.status, b.status)

    def test_primary_quantity_is_declared(self):
        event = evaluate_class_distribution(probability_matrix(BASE_RATES), BASE_RATES)
        self.assertEqual(event.metrics["primary_quantity"], "mean_predicted_probability_mass")

    def test_status_function_does_not_reference_argmax(self):
        """AST proof that the breach computation precedes and excludes argmax."""
        tree = monitoring_modules()["signals.py"]
        node = function_node(tree, "evaluate_class_distribution")
        assign_lines = [
            stmt.lineno for stmt in ast.walk(node)
            if isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "breaches" for t in stmt.targets)
        ]
        argmax_lines = [
            child.lineno for child in ast.walk(node)
            if isinstance(child, ast.Attribute) and child.attr == "argmax"
        ]
        self.assertTrue(assign_lines and argmax_lines)
        self.assertLess(max(assign_lines), min(argmax_lines),
                        "status inputs must be computed before argmax is even calculated")

    def test_stored_reference_loads_from_frozen_artifact(self):
        reference = load_stored_base_rate_reference()
        self.assertAlmostEqual(reference["H"], 0.43631588820108796, places=12)
        self.assertAlmostEqual(reference["D"], 0.2519227161883324, places=12)
        self.assertAlmostEqual(reference["A"], 0.31176139561057964, places=12)

    def test_prediction_time_only_no_outcomes_required(self):
        """Signal 3 must be computable with no realized outcome whatsoever."""
        event = evaluate_class_distribution(probability_matrix(BASE_RATES), BASE_RATES)
        self.assertEqual(event.status, Status.PASS)
        tree = monitoring_modules()["signals.py"]
        used = names_used(function_node(tree, "evaluate_class_distribution"))
        for forbidden in ("outcome", "settled_at", "select_eligible_observations", "realized"):
            self.assertNotIn(forbidden, used)


# =====================================================================
# 8-11. Signal 4: minimum n, threshold, maturity, window
# =====================================================================
class TestGate10CalibrationPolicy(unittest.TestCase):
    AS_OF = datetime(2026, 8, 17, 12, 0, 0)

    def test_locked_policy_values(self):
        self.assertEqual(mon.CALIBRATION_TOLERANCE_PP, 5.0)
        self.assertEqual(mon.MIN_SETTLED_SAMPLE, 200)
        self.assertEqual(mon.OBSERVATION_WINDOW_DAYS, 30)
        self.assertEqual(mon.SETTLEMENT_MATURITY_DAYS, 7)

    # -- 8. minimum sample size ----------------------------------------
    def test_one_below_minimum_is_insufficient_sample(self):
        records = settled_records(199, BASE_RATES, ["H"], self.AS_OF - timedelta(days=1))
        event = evaluate_calibration_post_outcome(records, self.AS_OF)
        self.assertEqual(event.status, Status.INSUFFICIENT_SAMPLE)
        self.assertEqual(event.sample_count, 199)
        self.assertNotIn("deviation_pp", event.metrics)

    def test_exactly_minimum_yields_a_verdict(self):
        records = settled_records(200, BASE_RATES, ["H"], self.AS_OF - timedelta(days=1))
        event = evaluate_calibration_post_outcome(records, self.AS_OF)
        self.assertIn(event.status, (Status.PASS, Status.ALERT))
        self.assertEqual(event.sample_count, 200)

    def test_insufficient_sample_is_never_pass_or_alert(self):
        for n in (0, 1, 50, 199):
            records = settled_records(n, BASE_RATES, ["H"], self.AS_OF - timedelta(days=1))
            event = evaluate_calibration_post_outcome(records, self.AS_OF)
            self.assertEqual(event.status, Status.INSUFFICIENT_SAMPLE)
            self.assertEqual(event.severity, Severity.INFO)

    # -- 9. +/-5pp threshold -------------------------------------------
    def test_perfectly_calibrated_batch_passes(self):
        outcomes = ["H"] * 40 + ["D"] * 25 + ["A"] * 35
        records = settled_records(200, {"H": 0.40, "D": 0.25, "A": 0.35},
                                  outcomes, self.AS_OF - timedelta(days=1))
        event = evaluate_calibration_post_outcome(records, self.AS_OF)
        self.assertEqual(event.status, Status.PASS)

    def test_exactly_five_pp_deviation_passes(self):
        outcomes = ["H"] * 45 + ["D"] * 25 + ["A"] * 30
        records = settled_records(200, {"H": 0.40, "D": 0.25, "A": 0.35},
                                  outcomes, self.AS_OF - timedelta(days=1))
        event = evaluate_calibration_post_outcome(records, self.AS_OF)
        self.assertAlmostEqual(event.metrics["deviation_pp"]["H"], -5.0, places=9)
        self.assertEqual(event.status, Status.PASS)

    def test_above_five_pp_deviation_alerts(self):
        outcomes = ["H"] * 50 + ["D"] * 25 + ["A"] * 25
        records = settled_records(200, {"H": 0.40, "D": 0.25, "A": 0.35},
                                  outcomes, self.AS_OF - timedelta(days=1))
        event = evaluate_calibration_post_outcome(records, self.AS_OF)
        self.assertEqual(event.status, Status.ALERT)
        self.assertIn("H", event.metrics["breaches_pp"])

    # -- 10. 7-day settlement maturity ---------------------------------
    def test_maturity_below_seven_days_is_excluded(self):
        settled = self.AS_OF - timedelta(days=1)
        records = settled_records(300, BASE_RATES, ["H"], settled, maturity_days=6.99)
        event = evaluate_calibration_post_outcome(records, self.AS_OF)
        self.assertEqual(event.status, Status.INSUFFICIENT_SAMPLE)
        self.assertEqual(event.metrics["exclusion_breakdown"]["immature"], 300)

    def test_maturity_exactly_seven_days_is_included(self):
        settled = self.AS_OF - timedelta(days=1)
        records = settled_records(200, BASE_RATES, ["H"], settled, maturity_days=7.0)
        eligible, reasons = select_eligible_observations(records, self.AS_OF)
        self.assertEqual(len(eligible), 200)
        self.assertEqual(reasons["immature"], 0)

    def test_unsettled_records_are_excluded(self):
        records = settled_records(10, BASE_RATES, ["H"], self.AS_OF - timedelta(days=1))
        for record in records:
            record["settled_at"] = None
        eligible, reasons = select_eligible_observations(records, self.AS_OF)
        self.assertEqual(eligible, [])
        self.assertEqual(reasons["unsettled"], 10)

    # -- 11. 30-day rolling window -------------------------------------
    def test_inside_window_included(self):
        records = settled_records(200, BASE_RATES, ["H"], self.AS_OF - timedelta(days=29))
        eligible, reasons = select_eligible_observations(records, self.AS_OF)
        self.assertEqual(len(eligible), 200)
        self.assertEqual(reasons["outside_window"], 0)

    def test_outside_window_excluded(self):
        records = settled_records(200, BASE_RATES, ["H"], self.AS_OF - timedelta(days=31))
        eligible, reasons = select_eligible_observations(records, self.AS_OF)
        self.assertEqual(eligible, [])
        self.assertEqual(reasons["outside_window"], 200)

    def test_window_boundary_exactly_thirty_days_included(self):
        records = settled_records(200, BASE_RATES, ["H"], self.AS_OF - timedelta(days=30))
        eligible, _ = select_eligible_observations(records, self.AS_OF)
        self.assertEqual(len(eligible), 200)

    def test_future_settlement_excluded(self):
        records = settled_records(5, BASE_RATES, ["H"], self.AS_OF + timedelta(days=1))
        eligible, reasons = select_eligible_observations(records, self.AS_OF)
        self.assertEqual(eligible, [])
        self.assertEqual(reasons["outside_window"], 5)

    def test_mixed_population_partitions_correctly(self):
        records = (
            settled_records(200, BASE_RATES, ["H"], self.AS_OF - timedelta(days=2))
            + settled_records(7, BASE_RATES, ["H"], self.AS_OF - timedelta(days=40))
            + settled_records(3, BASE_RATES, ["H"], self.AS_OF - timedelta(days=2), maturity_days=1)
        )
        eligible, reasons = select_eligible_observations(records, self.AS_OF)
        self.assertEqual(len(eligible), 200)
        self.assertEqual(reasons["outside_window"], 7)
        self.assertEqual(reasons["immature"], 3)

    def test_post_outcome_requires_outcomes(self):
        """Signal 4 is not computable without realized outcomes."""
        records = settled_records(200, BASE_RATES, ["H"], self.AS_OF - timedelta(days=1))
        for record in records:
            del record["outcome"]
        with self.assertRaises(KeyError):
            evaluate_calibration_post_outcome(records, self.AS_OF)


# =====================================================================
# 12 & 13. Signal 5: records, re-raises, zero tolerance
# =====================================================================
class _Boom(RuntimeError):
    pass


class TestGate10ProbabilityValidity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = JsonlEventStore(Path(self.tmp.name) / "events.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_wrapper_re_raises_the_original_exception_object_unchanged(self):
        sentinel = _Boom("row 3 does not sum to ~1")

        def failing(*_args, **_kwargs):
            raise sentinel

        with self.assertRaises(_Boom) as caught:
            monitored_predict_candidate(
                None, None, pd.DataFrame(), [], store=self.store, predict_fn=failing
            )
        self.assertIs(caught.exception, sentinel, "must re-raise the identical exception object")
        self.assertEqual(str(caught.exception), "row 3 does not sum to ~1")

    def test_failure_is_recorded_as_hard_fail(self):
        def failing(*_args, **_kwargs):
            raise _Boom("invalid probabilities")

        with self.assertRaises(_Boom):
            monitored_predict_candidate(
                None, None, pd.DataFrame(), [], store=self.store, predict_fn=failing
            )
        records = self.store.read_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["signal_id"], SIGNAL_S5_PROBABILITY_VALIDITY)
        self.assertEqual(records[0]["status"], "HARD_FAIL")
        self.assertEqual(records[0]["severity"], "CRITICAL")
        self.assertEqual(records[0]["metrics"]["invalid_batches"], 1)

    def test_success_path_returns_prediction_unchanged(self):
        class FakePrediction:
            probabilities = np.array([[0.4, 0.25, 0.35]])

        expected = FakePrediction()

        def ok(*_args, **_kwargs):
            return expected

        result = monitored_predict_candidate(
            None, None, pd.DataFrame(), [], store=self.store, predict_fn=ok
        )
        self.assertIs(result, expected)
        self.assertEqual(self.store.read_all()[0]["status"], "PASS")

    def test_wrapper_never_repairs_probabilities(self):
        """AST proof: no clipping/normalising/filling call exists in the module."""
        tree = monitoring_modules()["validity.py"]
        used = names_used(tree)
        for forbidden in ("clip", "normalize", "fillna", "nan_to_num", "renormalize", "repair"):
            self.assertNotIn(forbidden, used)

    def test_wrapper_uses_a_bare_raise(self):
        """A bare `raise` preserves type, message, instance and traceback."""
        tree = monitoring_modules()["validity.py"]
        node = function_node(tree, "monitored_predict_candidate")
        raises = [child for child in ast.walk(node) if isinstance(child, ast.Raise)]
        self.assertTrue(raises, "wrapper must re-raise")
        self.assertTrue(all(r.exc is None for r in raises),
                        "must be a bare `raise`, never `raise SomethingElse(...)`")

    def test_authority_remains_validate_probabilities(self):
        event = record_validity_failure(_Boom("x"), np.array([[0.5, 0.5, 0.5]]))
        self.assertEqual(event.metrics["authority"], "models.evaluate.validate_probabilities")

    def test_monitoring_does_not_reimplement_the_validator(self):
        """The package must not define its own validate_probabilities."""
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.assertNotEqual(node.name, "validate_probabilities",
                                        f"{name} must not redefine the authoritative validator")

    def test_zero_tolerance_single_invalid_row(self):
        matrix = np.array([[0.4, 0.25, 0.35]] * 99 + [[0.4, 0.25, 0.90]])
        breakdown = count_invalid_rows(matrix)
        self.assertEqual(breakdown["invalid_rows"], 1)
        self.assertEqual(breakdown["bad_sum_rows"], 1)
        event = record_validity_failure(_Boom("bad"), matrix)
        self.assertEqual(event.status, Status.HARD_FAIL)

    def test_row_breakdown_attributes_each_failure_mode(self):
        matrix = np.array([
            [0.4, 0.25, 0.35],          # valid
            [np.nan, 0.25, 0.35],       # non-finite
            [-0.1, 0.6, 0.5],           # negative
            [0.2, 0.2, 0.2],            # bad sum
        ])
        breakdown = count_invalid_rows(matrix)
        self.assertEqual(breakdown["n_rows"], 4)
        self.assertEqual(breakdown["invalid_rows"], 3)
        self.assertEqual(breakdown["non_finite_rows"], 1)
        self.assertEqual(breakdown["negative_rows"], 1)
        self.assertGreaterEqual(breakdown["bad_sum_rows"], 1)

    def test_valid_matrix_reports_no_invalid_rows(self):
        breakdown = count_invalid_rows(np.array([[0.4, 0.25, 0.35]] * 10))
        self.assertEqual(breakdown["invalid_rows"], 0)

    def test_default_predict_fn_resolves_to_candidate_contract(self):
        """Proves the production default without importing scikit-learn."""
        tree = monitoring_modules()["validity.py"]
        node = function_node(tree, "_default_predict_candidate")
        imported: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                imported.append(f"{child.module}:{','.join(a.name for a in child.names)}")
        self.assertIn("models.candidate_contract:predict_candidate", imported)


# =====================================================================
# 14-16. Signal 6: probability mass, no +/-0.03, shares Signal 4 policy
# =====================================================================
class TestGate10DrawBehaviour(unittest.TestCase):
    AS_OF = datetime(2026, 8, 17, 12, 0, 0)

    def _records(self, n, p_draw, draw_fraction):
        outcomes = ["D"] * int(round(draw_fraction * 100)) + ["H"] * (100 - int(round(draw_fraction * 100)))
        mass = {"H": (1.0 - p_draw) / 2, "D": p_draw, "A": (1.0 - p_draw) / 2}
        return settled_records(n, mass, outcomes, self.AS_OF - timedelta(days=1))

    def test_uses_probability_mass_not_argmax_share(self):
        records = self._records(200, 0.25, 0.25)
        event = evaluate_draw_behaviour(records, self.AS_OF)
        self.assertAlmostEqual(event.metrics["mean_predicted_p_draw"], 0.25, places=10)
        self.assertAlmostEqual(event.metrics["realized_draw_frequency"], 0.25, places=10)
        self.assertEqual(event.status, Status.PASS)
        # Draw never wins the argmax here (0.375 vs 0.25) yet the signal passes,
        # which is only possible if argmax plays no part in the verdict.
        self.assertEqual(event.metrics["diagnostic_only_argmax_draw_share"], 0.0)

    def test_argmax_share_cannot_trigger_an_alert(self):
        """Draw argmax share 0.0 -- the accepted Condition 2 limitation -- must not alert."""
        for p_draw in (0.24, 0.25, 0.26):
            event = evaluate_draw_behaviour(self._records(200, p_draw, 0.25), self.AS_OF)
            self.assertEqual(event.metrics["diagnostic_only_argmax_draw_share"], 0.0)
            self.assertEqual(event.status, Status.PASS)

    def test_no_separate_conflicting_argmax_alert_exists(self):
        tree = monitoring_modules()["signals.py"]
        node = function_node(tree, "evaluate_draw_behaviour")
        # The status must be derived from `breach`, which comes from the
        # probability-mass deviation only.
        source_names = names_used(node)
        self.assertIn("deviation_pp", source_names)
        assign_targets = {
            t.id for stmt in ast.walk(node) if isinstance(stmt, ast.Assign)
            for t in stmt.targets if isinstance(t, ast.Name)
        }
        self.assertIn("breach", assign_targets)
        self.assertNotIn("argmax_breach", assign_targets)

    def test_rejected_heuristic_constant_absent_from_whole_package(self):
        """AST constant scan: the 0.03 band must appear nowhere in monitoring."""
        for name, tree in monitoring_modules().items():
            self.assertNotIn(0.03, numeric_constants(tree),
                             f"{name} must not contain the rejected +/-0.03 heuristic")

    def test_phase3_draw_heuristic_is_not_imported(self):
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        self.assertNotEqual(alias.name, "investigate_draw_probability",
                                            f"{name} must not import the descriptive heuristic")
                    module = getattr(node, "module", None) or ""
                    self.assertNotIn("calibration", module,
                                     f"{name} must not depend on the Phase 3 calibration module")

    def test_shares_signal_four_threshold_and_policy(self):
        draw = evaluate_draw_behaviour(self._records(200, 0.25, 0.25), self.AS_OF)
        calib = evaluate_calibration_post_outcome(self._records(200, 0.25, 0.25), self.AS_OF)
        for key in ("tolerance_pp", "minimum_sample", "observation_window_days",
                    "settlement_maturity_days"):
            self.assertEqual(draw.threshold[key], calib.threshold[key])
        self.assertEqual(draw.threshold["tolerance_pp"], 5.0)
        self.assertEqual(draw.threshold["minimum_sample"], 200)

    def test_shares_signal_four_insufficient_sample_policy(self):
        event = evaluate_draw_behaviour(self._records(199, 0.25, 0.25), self.AS_OF)
        self.assertEqual(event.status, Status.INSUFFICIENT_SAMPLE)

    def test_shares_signal_four_maturity_and_window(self):
        immature = settled_records(250, {"H": 0.375, "D": 0.25, "A": 0.375}, ["D"],
                                   self.AS_OF - timedelta(days=1), maturity_days=6.99)
        self.assertEqual(evaluate_draw_behaviour(immature, self.AS_OF).status,
                         Status.INSUFFICIENT_SAMPLE)
        stale = settled_records(250, {"H": 0.375, "D": 0.25, "A": 0.375}, ["D"],
                                self.AS_OF - timedelta(days=31))
        self.assertEqual(evaluate_draw_behaviour(stale, self.AS_OF).status,
                         Status.INSUFFICIENT_SAMPLE)

    def test_five_pp_boundary_positive_and_negative_controls(self):
        at_boundary = evaluate_draw_behaviour(self._records(200, 0.30, 0.25), self.AS_OF)
        self.assertAlmostEqual(at_boundary.metrics["deviation_pp"], 5.0, places=9)
        self.assertEqual(at_boundary.status, Status.PASS)
        beyond = evaluate_draw_behaviour(self._records(200, 0.31, 0.25), self.AS_OF)
        self.assertEqual(beyond.status, Status.ALERT)
        under = evaluate_draw_behaviour(self._records(200, 0.19, 0.25), self.AS_OF)
        self.assertEqual(under.status, Status.ALERT)

    def test_accepted_limitation_is_documented_in_the_event(self):
        event = evaluate_draw_behaviour(self._records(200, 0.25, 0.25), self.AS_OF)
        self.assertIn("Condition 2", event.metrics["accepted_limitation"])
        self.assertIn("accepted limitation", event.metrics["accepted_limitation"])

    def test_no_reliable_draw_classifier_claim(self):
        """No production-facing output may present the candidate as a reliable Draw classifier.

        The invariant is about ASSERTIONS, not about the words appearing
        anywhere in a file: a disclaimer ("... is NOT a reliable Draw
        classifier") upholds the invariant, whereas a bare claim violates
        it. A substring search cannot tell those apart, so this test is
        negation-aware and applies to (a) every string actually emitted in
        a monitoring event, which is what a consumer sees, and (b) every
        non-docstring string constant in the package, which is where an
        emitted string could originate.
        """
        for text, origin in self._production_facing_strings():
            for match in re.finditer(r"reliable draw|good draw classifier|solves draw", text.lower()):
                prefix = text.lower()[max(0, match.start() - 60):match.start()]
                self.assertTrue(
                    re.search(r"\b(not|never|no|cannot|must not|non-)\b[^.]*$", prefix),
                    f"{origin} makes an unqualified Draw-reliability claim: "
                    f"...{text[max(0, match.start() - 60):match.end() + 20]}...",
                )

    def _production_facing_strings(self):
        """(string, origin) pairs a consumer could see, plus their sources."""
        pairs: list[tuple[str, str]] = []

        as_of = self.AS_OF
        records = self._records(200, 0.25, 0.25)
        frame = full_contract_frame(50)
        events = [
            evaluate_null_rate_ceiling(frame),
            evaluate_null_rate_drift(frame, build_null_rate_baseline("max")),
            evaluate_unseen_competition_ids([200, 999]),
            evaluate_class_distribution(probability_matrix(BASE_RATES), BASE_RATES),
            evaluate_calibration_post_outcome(records, as_of),
            evaluate_draw_behaviour(records, as_of),
            record_validity_failure(_Boom("x"), np.array([[0.5, 0.5, 0.5]])),
        ]
        for event in events:
            payload = event.to_dict()
            pairs.append((payload["message"], f"event {payload['signal_id']}.message"))
            for key, value in payload["metrics"].items():
                if isinstance(value, str):
                    pairs.append((value, f"event {payload['signal_id']}.metrics.{key}"))

        # Non-docstring string constants: where an emitted string is authored.
        for name, tree in monitoring_modules().items():
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                ):
                    pairs.append((node.value, f"{name}:{node.lineno}"))
        return pairs

    def test_claim_detector_catches_a_real_violation(self):
        """Negative control: the detector must still reject an actual claim."""
        offending = "this model is a reliable draw classifier"
        matched = False
        for match in re.finditer(r"reliable draw", offending):
            prefix = offending[max(0, match.start() - 60):match.start()]
            if not re.search(r"\b(not|never|no|cannot|must not|non-)\b[^.]*$", prefix):
                matched = True
        self.assertTrue(matched, "detector failed to flag an unqualified claim")

    def test_claim_detector_accepts_a_disclaimer(self):
        """Positive control: a disclaimer must not be flagged."""
        disclaimer = "the candidate is not a reliable draw classifier"
        flagged = False
        for match in re.finditer(r"reliable draw", disclaimer):
            prefix = disclaimer[max(0, match.start() - 60):match.start()]
            if not re.search(r"\b(not|never|no|cannot|must not|non-)\b[^.]*$", prefix):
                flagged = True
        self.assertFalse(flagged, "detector wrongly flagged a disclaimer")


# =====================================================================
# 17. No 2025/26 access
# =====================================================================
class TestGate10NoTestSeasonAccess(unittest.TestCase):
    def test_no_season_constant_or_split_machinery_imported(self):
        forbidden_names = {
            "FINAL_TEST_SEASONS", "FINAL_TRAIN_SEASONS", "WALK_FORWARD_FOLDS",
            "final_split", "split_by_seasons", "iter_walk_forward_folds",
            "load_supervised_dataset",
        }
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        self.assertNotIn(alias.name, forbidden_names,
                                         f"{name} imports split machinery: {alias.name}")
                    module = getattr(node, "module", None) or ""
                    self.assertNotIn("splits", module.split("."),
                                     f"{name} imports the splits module")

    def test_no_test_season_literal_in_executable_constants(self):
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn("2025/2026", node.value,
                                     f"{name} contains a test-season literal")

    def test_baseline_api_structurally_refuses_a_test_partition(self):
        for attempt in ("final_test", "2025_26", "test_2025_2026"):
            with self.assertRaises(MonitoringInputError):
                build_null_rate_baseline("single", [attempt])

    def test_base_rate_artifact_is_walk_forward_only(self):
        payload = json.loads((REPO_ROOT / mon.config.BASE_RATE_ARTIFACT).read_text(encoding="utf-8"))
        self.assertEqual(payload["n_folds"], 3)
        self.assertEqual(payload["diagnostics"]["n"], 5331)

    def test_monitoring_reads_no_database(self):
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, "module", None) or ""
                    names = [a.name for a in node.names]
                    self.assertNotIn("sqlite3", names + [module],
                                     f"{name} must not open a database")


# =====================================================================
# 18. No frozen V1 file modified
# =====================================================================
class TestGate10FrozenV1Untouched(unittest.TestCase):
    def setUp(self):
        self.pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]

    def test_all_pinned_baselines_byte_identical(self):
        mismatches = []
        for rel, record in self.pins.items():
            path = REPO_ROOT / rel
            self.assertTrue(path.exists(), f"pinned baseline missing: {rel}")
            if md5(path) != record["expected"]:
                mismatches.append(rel)
        self.assertEqual(mismatches, [], f"frozen baseline drift: {mismatches}")

    def test_seven_frozen_v1_sources_pinned_and_unchanged(self):
        sources = sorted(r for r in self.pins if r.startswith("src/models/"))
        self.assertEqual(len(sources), 7)
        for rel in sources:
            self.assertEqual(md5(REPO_ROOT / rel), self.pins[rel]["expected"], rel)

    def test_evaluate_py_specifically_unchanged(self):
        rel = "src/models/evaluate.py"
        self.assertIn(rel, self.pins)
        self.assertEqual(md5(REPO_ROOT / rel), self.pins[rel]["expected"])

    def test_model_version_still_v1(self):
        self.assertEqual(MODEL_VERSION, "v1.0")

    def test_monitoring_never_writes_model_version(self):
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.assertNotEqual(target.id, "MODEL_VERSION", name)
                        if isinstance(target, ast.Attribute):
                            self.assertNotEqual(target.attr, "MODEL_VERSION", name)

    def test_monitoring_lives_outside_the_models_package(self):
        self.assertTrue(MONITORING_DIR.is_dir())
        self.assertFalse((REPO_ROOT / "src" / "models" / "monitoring.py").exists())


# =====================================================================
# 19. No model training
# =====================================================================
class TestGate10NoTraining(unittest.TestCase):
    def test_no_fit_call_anywhere(self):
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, {"fit", "fit_transform", "partial_fit"},
                                     f"{name} calls {node.func.attr}()")

    def test_no_estimator_or_trainer_import(self):
        forbidden_modules = {"sklearn", "models.train", "models.baselines", "models.run_experiments"}
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, "module", None) or ""
                    for forbidden in forbidden_modules:
                        self.assertFalse(
                            module == forbidden or module.startswith(forbidden + "."),
                            f"{name} imports {module}",
                        )

    def test_no_estimator_is_constructed(self):
        """AST: no estimator/transformer/calibrator object is instantiated.

        Semantic, not token-based: a monitoring evaluator that merely
        computes post-outcome diagnostics (e.g. `evaluate_calibration_post_outcome`)
        is permitted, because it fits nothing. What is forbidden is
        CONSTRUCTING a fittable object.
        """
        forbidden_constructors = {
            "LogisticRegression", "HistGradientBoostingClassifier", "SimpleImputer",
            "StandardScaler", "OneHotEncoder", "Pipeline", "ColumnTransformer",
            "MulticlassPlattCalibrator", "MulticlassIsotonicCalibrator", "IdentityCalibrator",
            "LogisticRegressionPreprocessor", "CompetitionCodeEncoder",
            "FrequencyBaseline", "MajorityClassBaseline",
        }
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    called = node.func.id if isinstance(node.func, ast.Name) else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else None
                    )
                    self.assertNotIn(called, forbidden_constructors,
                                     f"{name} constructs an estimator: {called}")

    def test_no_training_tuning_or_calibration_fitting_call(self):
        """AST: no fitting, training, tuning, or calibration-fitting call.

        The invariant is behavioural. `evaluate_*` diagnostics are allowed;
        anything that fits parameters is not.
        """
        forbidden_calls = {
            "fit", "fit_transform", "partial_fit",
            "train", "train_model", "train_logistic_regression",
            "train_hist_gradient_boosting", "tune", "grid_search", "search",
            "_fit_platt_1d", "fit_calibrator", "select_calibration_method",
            "evaluate_calibration_methods", "run_final_frozen_evaluation",
        }
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    called = node.func.id if isinstance(node.func, ast.Name) else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else None
                    )
                    self.assertNotIn(called, forbidden_calls,
                                     f"{name} performs a fitting/training call: {called}")

    def test_monitoring_evaluators_are_permitted(self):
        """Guard against the check becoming over-broad again.

        `evaluate_calibration_post_outcome` is a monitoring evaluator: it
        computes a post-outcome diagnostic and fits nothing. It must remain
        present and callable.
        """
        from monitoring import signals as signals_module

        self.assertTrue(callable(signals_module.evaluate_calibration_post_outcome))
        tree = monitoring_modules()["signals.py"]
        node = function_node(tree, "evaluate_calibration_post_outcome")
        used = names_used(node)
        for fitting in ("fit", "fit_transform", "partial_fit"):
            self.assertNotIn(fitting, used)


# =====================================================================
# 20. No artifact overwrite
# =====================================================================
class TestGate10NoArtifactOverwrite(unittest.TestCase):
    def test_running_every_signal_leaves_pinned_artifacts_unchanged(self):
        pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
        before = {rel: md5(REPO_ROOT / rel) for rel in pins}
        artifact = REPO_ROOT / mon.config.BASE_RATE_ARTIFACT
        artifact_before = md5(artifact)

        as_of = datetime(2026, 8, 17, 12, 0, 0)
        frame = full_contract_frame(100)
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlEventStore(Path(tmp) / "events.jsonl")
            store.append(evaluate_null_rate_ceiling(frame))
            store.append(evaluate_null_rate_drift(frame, build_null_rate_baseline("max")))
            store.append(evaluate_unseen_competition_ids([200, 999]))
            store.append(evaluate_class_distribution(probability_matrix(BASE_RATES)))
            store.append(evaluate_calibration_post_outcome(
                settled_records(200, BASE_RATES, ["H", "D", "A"], as_of - timedelta(days=1)), as_of))
            store.append(evaluate_draw_behaviour(
                settled_records(200, BASE_RATES, ["H", "D", "A"], as_of - timedelta(days=1)), as_of))
            store.append(record_validity_failure(_Boom("x"), np.array([[0.5, 0.5, 0.5]])))
            self.assertEqual(store.count(), 7)

        after = {rel: md5(REPO_ROOT / rel) for rel in pins}
        self.assertEqual(before, after, "monitoring must not modify any pinned artifact")
        self.assertEqual(artifact_before, md5(artifact), "base-rate artifact must be read-only")

    def test_store_writes_only_to_the_supplied_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "events.jsonl"
            store = JsonlEventStore(target)
            store.append(evaluate_unseen_competition_ids([200]))
            self.assertTrue(target.exists())
            created = {p.name for p in Path(tmp).rglob("*") if p.is_file()}
            self.assertEqual(created, {"events.jsonl"})

    def test_store_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlEventStore(Path(tmp) / "events.jsonl")
            store.append(evaluate_unseen_competition_ids([200]))
            store.append(evaluate_unseen_competition_ids([999]))
            self.assertEqual(store.count(), 2)
            for method in ("update", "delete", "overwrite", "truncate"):
                self.assertFalse(hasattr(store, method), f"store must not expose {method}()")

    def test_no_default_write_path_inside_the_repository(self):
        """The store must never silently default to a repo location."""
        tree = monitoring_modules()["store.py"]
        node = function_node(tree, "__init__")
        defaults = [d for d in node.args.defaults]
        self.assertEqual(defaults, [], "store path must be caller-supplied with no default")


# =====================================================================
# 21. Determinism
# =====================================================================
class TestGate10Determinism(unittest.TestCase):
    AS_OF = datetime(2026, 8, 17, 12, 0, 0)

    @staticmethod
    def _stable(event: MonitoringEvent) -> str:
        payload = event.to_dict()
        for volatile in ("event_id", "timestamp"):
            payload.pop(volatile)
        return json.dumps(payload, sort_keys=True, default=str)

    def test_all_signals_are_deterministic_across_repeated_calls(self):
        frame = full_contract_frame(200, {"home_points_last5": 0.05})
        baseline = build_null_rate_baseline("max")
        records = settled_records(200, BASE_RATES, ["H", "D", "A"], self.AS_OF - timedelta(days=1))
        matrix = probability_matrix(BASE_RATES)

        cases = (
            lambda: evaluate_null_rate_ceiling(frame),
            lambda: evaluate_null_rate_drift(frame, baseline),
            lambda: evaluate_unseen_competition_ids([200, 419, 999]),
            lambda: evaluate_class_distribution(matrix, BASE_RATES),
            lambda: evaluate_calibration_post_outcome(records, self.AS_OF),
            lambda: evaluate_draw_behaviour(records, self.AS_OF),
        )
        for index, case in enumerate(cases):
            first, second, third = case(), case(), case()
            self.assertEqual(self._stable(first), self._stable(second), f"case {index}")
            self.assertEqual(self._stable(second), self._stable(third), f"case {index}")

    def test_baseline_construction_is_deterministic(self):
        for aggregation in ("max", "mean"):
            self.assertEqual(
                build_null_rate_baseline(aggregation), build_null_rate_baseline(aggregation)
            )

    def test_event_ids_are_unique_per_event(self):
        a = evaluate_unseen_competition_ids([200])
        b = evaluate_unseen_competition_ids([200])
        self.assertNotEqual(a.event_id, b.event_id)

    def test_row_order_changes_aggregates_only_at_machine_epsilon(self):
        """Row-order invariance is NOT a Gate 10 requirement and is not claimed.

        Floating-point summation is not associative, so reordering rows can
        legitimately change an aggregate in the last bit or two. Gate 10
        requires DETERMINISM (identical input -> identical output), which is
        asserted bit-for-bit by the tests above. Here the weaker, numerically
        correct property is asserted instead: reordering perturbs aggregates
        only at machine-epsilon scale -- far below any monitoring threshold,
        the tightest of which is 2.0 pp (i.e. 0.02), some fourteen orders of
        magnitude larger. No acceptance threshold is relaxed by this test.
        """
        matrix = np.array([[0.5, 0.2, 0.3], [0.3, 0.3, 0.4], [0.4, 0.25, 0.35]])
        forward = evaluate_class_distribution(matrix, BASE_RATES)
        reversed_ = evaluate_class_distribution(matrix[::-1], BASE_RATES)

        for cls in ("H", "D", "A"):
            a = forward.metrics["mean_predicted_probability_mass"][cls]
            b = reversed_.metrics["mean_predicted_probability_mass"][cls]
            self.assertLessEqual(
                abs(a - b), 1e-12,
                f"class {cls} differs by more than machine-epsilon scale: {abs(a - b)}",
            )
        # The verdict itself must be unaffected by row order.
        self.assertEqual(forward.status, reversed_.status)

    def test_reordering_can_never_flip_a_threshold_verdict(self):
        """The epsilon-scale perturbation is irrelevant next to the thresholds."""
        rng = np.random.default_rng(0)
        matrix = rng.dirichlet((2.0, 1.5, 2.0), size=500)
        permuted = matrix[rng.permutation(len(matrix))]
        a = evaluate_class_distribution(matrix, BASE_RATES)
        b = evaluate_class_distribution(permuted, BASE_RATES)
        self.assertEqual(a.status, b.status)
        for cls in ("H", "D", "A"):
            self.assertLessEqual(
                abs(a.metrics["deviation_pp"][cls] - b.metrics["deviation_pp"][cls]), 1e-9
            )


# =====================================================================
# 22. Event schema validation
# =====================================================================
class TestGate10EventSchema(unittest.TestCase):
    def _event(self) -> MonitoringEvent:
        return evaluate_unseen_competition_ids([200])

    def test_all_required_fields_present(self):
        payload = self._event().to_dict()
        for field in REQUIRED_EVENT_FIELDS:
            self.assertIn(field, payload)

    def test_required_field_set_matches_the_emission_contract(self):
        self.assertEqual(
            set(REQUIRED_EVENT_FIELDS),
            {
                "schema_version", "event_id", "timestamp", "model_version", "signal_id",
                "signal_name", "status", "severity", "metrics", "baseline", "threshold",
                "sample_count", "context", "message",
            },
        )

    def test_model_version_is_v1_and_validated(self):
        self.assertEqual(self._event().to_dict()["model_version"], "v1.0")
        payload = self._event().to_dict()
        payload["model_version"] = "v2.0"
        with self.assertRaises(SchemaViolation):
            validate_event(payload)

    def test_statuses_are_exactly_the_five_authorized(self):
        self.assertEqual(
            {s.value for s in Status},
            {"PASS", "ALERT", "HARD_FAIL", "INSUFFICIENT_SAMPLE", "DEGRADED"},
        )

    def test_severity_is_derived_from_status(self):
        self.assertEqual(mon.config.severity_for(Status.PASS), Severity.INFO)
        self.assertEqual(mon.config.severity_for(Status.INSUFFICIENT_SAMPLE), Severity.INFO)
        self.assertEqual(mon.config.severity_for(Status.DEGRADED), Severity.WARNING)
        self.assertEqual(mon.config.severity_for(Status.ALERT), Severity.WARNING)
        self.assertEqual(mon.config.severity_for(Status.HARD_FAIL), Severity.CRITICAL)

    def test_missing_field_rejected(self):
        for field in REQUIRED_EVENT_FIELDS:
            payload = self._event().to_dict()
            payload.pop(field)
            with self.assertRaises(SchemaViolation, msg=f"missing {field} must be rejected"):
                validate_event(payload)

    def test_unknown_status_and_severity_rejected(self):
        for key, bad in (("status", "MAYBE"), ("severity", "URGENT")):
            payload = self._event().to_dict()
            payload[key] = bad
            with self.assertRaises(SchemaViolation):
                validate_event(payload)

    def test_bad_types_rejected(self):
        for key, bad in (
            ("metrics", "not-a-mapping"),
            ("context", 3),
            ("sample_count", "many"),
            ("sample_count", -1),
            ("signal_id", ""),
            ("message", ""),
            ("timestamp", "not-a-date"),
            ("schema_version", "something-else"),
        ):
            payload = self._event().to_dict()
            payload[key] = bad
            with self.assertRaises(SchemaViolation, msg=f"{key}={bad!r} must be rejected"):
                validate_event(payload)

    def test_events_round_trip_through_the_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlEventStore(Path(tmp) / "events.jsonl")
            event = self._event()
            store.append(event)
            record = store.read_all()[0]
            self.assertEqual(record["event_id"], event.event_id)
            validate_event(record)

    def test_corrupt_store_line_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"not":"closed"\n', encoding="utf-8")
            with self.assertRaises(SchemaViolation):
                JsonlEventStore(path).read_all()

    def test_event_is_immutable(self):
        event = self._event()
        with self.assertRaises(Exception):
            event.status = Status.PASS  # type: ignore[misc]


# =====================================================================
# 23. Empty / insufficient data behaviour
# =====================================================================
class TestGate10EmptyAndInsufficientData(unittest.TestCase):
    AS_OF = datetime(2026, 8, 17, 12, 0, 0)

    def test_empty_batch_raises_rather_than_reporting_a_false_pass(self):
        empty = full_contract_frame(0)
        with self.assertRaises(MonitoringInputError):
            observed_family_null_rates(empty)
        with self.assertRaises(MonitoringInputError):
            evaluate_null_rate_ceiling(empty)
        with self.assertRaises(MonitoringInputError):
            evaluate_null_rate_drift(empty, build_null_rate_baseline("max"))

    def test_empty_probability_matrix_raises(self):
        with self.assertRaises(MonitoringInputError):
            evaluate_class_distribution(np.empty((0, 3)), BASE_RATES)

    def test_malformed_probability_shape_raises(self):
        with self.assertRaises(MonitoringInputError):
            evaluate_class_distribution(np.array([[0.5, 0.5]]), BASE_RATES)

    def test_no_records_yields_insufficient_sample_not_pass(self):
        for evaluate in (evaluate_calibration_post_outcome, evaluate_draw_behaviour):
            event = evaluate([], self.AS_OF)
            self.assertEqual(event.status, Status.INSUFFICIENT_SAMPLE)
            self.assertEqual(event.sample_count, 0)

    def test_empty_competition_id_list_does_not_fabricate_a_finding(self):
        event = evaluate_unseen_competition_ids([])
        self.assertEqual(event.status, Status.PASS)
        self.assertEqual(event.metrics["unseen_ids"], [])
        self.assertEqual(event.sample_count, 0)

    def test_empty_store_reads_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlEventStore(Path(tmp) / "absent.jsonl")
            self.assertEqual(store.read_all(), [])
            self.assertEqual(store.count(), 0)

    def test_missing_base_rate_artifact_raises(self):
        with self.assertRaises(MonitoringInputError):
            load_stored_base_rate_reference("does/not/exist.json")

    def test_incomplete_base_rate_reference_raises(self):
        with self.assertRaises(MonitoringInputError):
            evaluate_class_distribution(probability_matrix(BASE_RATES), {"H": 0.4, "D": 0.25})


# =====================================================================
# 24. Coverage: all six required signals actually emitted
# =====================================================================
class TestGate10SignalCoverage(unittest.TestCase):
    AS_OF = datetime(2026, 8, 17, 12, 0, 0)

    def test_six_requirements_are_mapped(self):
        self.assertEqual(len(REQUIRED_SIGNAL_COVERAGE), 6)

    def test_every_required_signal_is_emitted_end_to_end(self):
        frame = full_contract_frame(100)
        records = settled_records(200, BASE_RATES, ["H", "D", "A"], self.AS_OF - timedelta(days=1))
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlEventStore(Path(tmp) / "events.jsonl")
            store.append(evaluate_null_rate_ceiling(frame))
            store.append(evaluate_null_rate_drift(frame, build_null_rate_baseline("max")))
            store.append(evaluate_unseen_competition_ids([200, 999]))
            store.append(evaluate_class_distribution(probability_matrix(BASE_RATES)))
            store.append(evaluate_calibration_post_outcome(records, self.AS_OF))
            store.append(evaluate_draw_behaviour(records, self.AS_OF))
            store.append(record_validity_failure(_Boom("x"), np.array([[0.5, 0.5, 0.5]])))

            emitted = store.signal_ids()
            for requirement, ids in REQUIRED_SIGNAL_COVERAGE.items():
                self.assertTrue(
                    any(sid in emitted for sid in ids),
                    f"§12 requirement not emitted: {requirement}",
                )
            for record in store.read_all():
                validate_event(record)

    def test_expected_signal_ids_exist(self):
        self.assertEqual(
            {
                SIGNAL_S1_CEILING, SIGNAL_S1_DRIFT, SIGNAL_S2_UNSEEN_COMPETITION,
                SIGNAL_S3_CLASS_DISTRIBUTION, SIGNAL_S4_CALIBRATION,
                SIGNAL_S5_PROBABILITY_VALIDITY, SIGNAL_S6_DRAW_BEHAVIOUR,
            },
            set(mon.SIGNAL_NAMES),
        )

    def test_no_dashboard_api_scheduler_or_alert_integration(self):
        forbidden = {
            "flask", "fastapi", "django", "requests", "http", "http.client", "smtplib",
            "urllib", "urllib.request", "socket", "schedule", "apscheduler", "celery",
            "boto3", "google.cloud",
        }
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, "module", None) or ""
                    candidates = {module} | {a.name for a in node.names}
                    overlap = candidates & forbidden
                    self.assertEqual(overlap, set(), f"{name} imports {overlap}")

    def test_package_declares_no_third_party_dependency_beyond_numpy_pandas(self):
        allowed_roots = {
            "numpy", "pandas", "models", "monitoring",
            "json", "uuid", "dataclasses", "datetime", "typing", "pathlib", "enum",
            "__future__", "hashlib", "collections",
        }
        for name, tree in monitoring_modules().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], allowed_roots, f"{name}: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        continue  # relative import inside the package
                    root = (node.module or "").split(".")[0]
                    self.assertIn(root, allowed_roots, f"{name}: {node.module}")


if __name__ == "__main__":
    unittest.main()
