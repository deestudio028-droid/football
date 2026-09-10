"""Comprehensive Test Suite for the Prospective Validation Pipeline.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python tests/test_prospective_validation_pipeline.py

Covers 30+ test cases across:
1. Contract & Frozen Hash Verification
2. Probability Validity & Boundary Enforcement
3. Causal Timing & Prediction Lock
4. Immutability & Tamper Detection
5. Store Operations, Deduplication & Replay Protection
6. Two-Stage Outcome Separation
7. Historical 300 OOS Exclusion
8. Continuous Monitoring & Bucket Splits
9. Statistical Decision Gating
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from monitoring.prospective_pipeline import (
    CohortStatus,
    DuplicatePredictionError,
    LockedPredictionRecord,
    LockedRecordMutationError,
    MethodologyMismatchError,
    OutcomeJoinError,
    OutcomeRecord,
    PreKickoffViolationError,
    ProbabilityValidationError,
    ProspectiveContract,
    ProspectiveMonitor,
    ProspectiveValidationStore,
    ReusedFixtureRejectionError,
    canonical_json_hash,
    run_final_statistical_validation,
)


class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.passes = 0
        self.failures = 0
        self.messages: list[str] = []

    def check(self, condition: bool, description: str, detail: str = ""):
        if condition:
            self.passes += 1
            self.messages.append(f"  PASS: {description}" + (f" — {detail}" if detail else ""))
        else:
            self.failures += 1
            self.messages.append(f"  FAIL: {description}" + (f" — {detail}" if detail else ""))

    def report(self):
        print(f"\n{'=' * 78}\nSuite: {self.name}\n{'=' * 78}")
        for msg in self.messages:
            print(msg)
        print(f"\n  Pass: {self.passes}  Fail: {self.failures}")


def test_contract_and_hashes() -> TestSuite:
    s = TestSuite("1. Contract & Frozen Hash Verification")
    contract = ProspectiveContract.load_and_verify()

    s.check(contract.model_id == "v4_draw_champion", "Model ID == v4_draw_champion")
    s.check(contract.model_version == "v4.0-champion-dc-elo-stacking", "Model Version == v4.0-champion-dc-elo-stacking")
    s.check(contract.methodology_hash == "9c396e7e5364f93f079313726c1ba499", "Methodology Hash matches frozen champion (9c396e...)")
    s.check(contract.protocol_hash == "1311eb7fa75f51c77a1fc09c0cf4df68", "Protocol Hash matches frozen protocol (1311eb...)")
    s.check(contract.min_sample_size == 1050, "Minimum prospective cohort == 1,050")
    s.check(contract.preferred_sample_size == 1500, "Preferred prospective cohort == 1,500")
    s.check(len(contract.reused_300_fixture_ids) == 300, "Reused 300 OOS fixtures loaded for exclusion", f"count={len(contract.reused_300_fixture_ids)}")
    return s


def test_prediction_lock_and_timing() -> TestSuite:
    s = TestSuite("2. Prediction Lock & Causal Timing Enforcement")
    contract = ProspectiveContract.load_and_verify()

    # Valid pre-match timing
    t_pred = "2026-09-01T13:00:00+00:00"
    t_kick = "2026-09-01T15:00:00+00:00"

    rec = LockedPredictionRecord.create_and_lock(
        fixture_id=900001,
        prediction_timestamp=t_pred,
        kickoff_timestamp=t_kick,
        p_v4={"H": 0.45, "D": 0.25, "A": 0.30},
        p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
        p_draw_dc=0.27,
        p_draw_elo=0.28,
        expected_goals_home=1.5,
        expected_goals_away=1.2,
        modal_scoreline="1-1",
        league="Premier League",
        home_team="Team A",
        away_team="Team B",
        home_elo=1600.0,
        away_elo=1550.0,
        lambda_home=1.5,
        lambda_away=1.2,
        contract=contract,
    )

    s.check(rec.status == "LOCKED", "Record status is LOCKED")
    s.check(rec.verify_hash(), "SHA-256 prediction digest is valid")
    s.check(rec.prediction_id.startswith("pred_"), "Prediction ID has standard prefix")

    # Timing Violation: prediction after kickoff
    late_caught = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=900002,
            prediction_timestamp="2026-09-01T15:05:00+00:00",
            kickoff_timestamp="2026-09-01T15:00:00+00:00",
            p_v4={"H": 0.45, "D": 0.25, "A": 0.30},
            p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27,
            p_draw_elo=0.28,
            expected_goals_home=1.5,
            expected_goals_away=1.2,
            modal_scoreline="1-1",
            league="Premier League",
            home_team="Team A",
            away_team="Team B",
            home_elo=1600.0,
            away_elo=1550.0,
            lambda_home=1.5,
            lambda_away=1.2,
            contract=contract,
        )
    except PreKickoffViolationError:
        late_caught = True
    s.check(late_caught, "Post-kickoff prediction attempt strictly rejected")
    return s


def test_probability_validation() -> TestSuite:
    s = TestSuite("3. Probability Validation & Invariant Gates")
    contract = ProspectiveContract.load_and_verify()
    t_pred, t_kick = "2026-09-01T13:00:00+00:00", "2026-09-01T15:00:00+00:00"

    # 1. Non-simplex (sum != 1)
    bad_sum = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=900003, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": 0.50, "D": 0.50, "A": 0.50},
            p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Serie A", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract,
        )
    except ProbabilityValidationError:
        bad_sum = True
    s.check(bad_sum, "Non-simplex probabilities rejected")

    # 2. Negative probability
    bad_neg = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=900004, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": -0.10, "D": 0.60, "A": 0.50},
            p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Serie A", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract,
        )
    except ProbabilityValidationError:
        bad_neg = True
    s.check(bad_neg, "Negative probabilities rejected")

    # 3. NaN probability
    bad_nan = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=900005, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": float("nan"), "D": 0.50, "A": 0.50},
            p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Serie A", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract,
        )
    except ProbabilityValidationError:
        bad_nan = True
    s.check(bad_nan, "NaN probabilities rejected")

    # 4. Infinity probability
    bad_inf = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=900006, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": float("inf"), "D": 0.50, "A": 0.50},
            p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Serie A", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract,
        )
    except ProbabilityValidationError:
        bad_inf = True
    s.check(bad_inf, "Infinity probabilities rejected")
    return s


def test_reused_300_exclusion() -> TestSuite:
    s = TestSuite("4. Reused Historical 300 OOS Exclusion")
    contract = ProspectiveContract.load_and_verify()
    t_pred, t_kick = "2026-09-01T13:00:00+00:00", "2026-09-01T15:00:00+00:00"

    reused_sample_id = next(iter(contract.reused_300_fixture_ids))
    caught = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=reused_sample_id, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": 0.45, "D": 0.25, "A": 0.30},
            p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="La Liga", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract,
        )
    except ReusedFixtureRejectionError:
        caught = True
    s.check(caught, f"Historical 300 OOS fixture {reused_sample_id} rejected from prospective cohort")
    return s


def test_store_and_tamper_detection() -> TestSuite:
    s = TestSuite("5. Store Operations, Tamper Detection & Replay Protection")
    contract = ProspectiveContract.load_and_verify()
    store = ProspectiveValidationStore(":memory:")
    t_pred, t_kick = "2026-09-01T13:00:00+00:00", "2026-09-01T15:00:00+00:00"

    rec = LockedPredictionRecord.create_and_lock(
        fixture_id=900010, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
        p_v4={"H": 0.45, "D": 0.25, "A": 0.30},
        p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
        p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
        modal_scoreline="1-1", league="Bundesliga", home_team="A", away_team="B",
        home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract,
    )

    store.lock_prediction(rec)
    retrieved = store.get_prediction(900010)
    s.check(retrieved is not None, "Record locked and retrieved from store")
    s.check(retrieved.prediction_hash == rec.prediction_hash, "Retrieved hash matches locked hash")

    # Duplicate prediction lock attempt
    dup_caught = False
    try:
        store.lock_prediction(rec)
    except DuplicatePredictionError:
        dup_caught = True
    s.check(dup_caught, "Duplicate fixture lock attempt rejected")

    # Tampering test (simulate byte tampering)
    tampered_rec = LockedPredictionRecord(
        fixture_id=900011, prediction_id="pred_tampered", prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
        model_id=contract.model_id, model_version=contract.model_version, methodology_hash=contract.methodology_hash,
        p_home_v4=0.45, p_draw_v4=0.25, p_away_v4=0.30,
        p_home_champion=0.99, p_draw_champion=0.01, p_away_champion=0.00,  # tampered probs
        p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
        modal_scoreline="1-1", league="Bundesliga", home_team="A", away_team="B",
        home_elo=1500.0, away_elo=1500.0, elo_diff=100.0, abs_elo_diff=100.0,
        lambda_home=1.5, lambda_away=1.2, prediction_hash=rec.prediction_hash, status="LOCKED",
    )
    tamper_caught = False
    try:
        store.lock_prediction(tampered_rec)
    except LockedRecordMutationError:
        tamper_caught = True
    s.check(tamper_caught, "Tampered prediction rejected on lock")
    return s


def test_two_stage_outcome_join() -> TestSuite:
    s = TestSuite("6. Two-Stage Outcome Join & Separation")
    contract = ProspectiveContract.load_and_verify()
    store = ProspectiveValidationStore(":memory:")
    t_pred, t_kick = "2026-09-01T13:00:00+00:00", "2026-09-01T15:00:00+00:00"
    t_out = "2026-09-01T17:00:00+00:00"

    rec = LockedPredictionRecord.create_and_lock(
        fixture_id=900020, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
        p_v4={"H": 0.45, "D": 0.25, "A": 0.30},
        p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
        p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
        modal_scoreline="1-1", league="Ligue 1", home_team="A", away_team="B",
        home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract,
    )
    store.lock_prediction(rec)

    outcome = OutcomeRecord.create(
        fixture_id=900020, outcome_timestamp=t_out, home_goals=2, away_goals=1, kickoff_timestamp=t_kick
    )
    store.join_outcome(outcome)

    pairs = store.get_joined_records()
    s.check(len(pairs) == 1, "Joined pair retrieved successfully")
    pred_res, out_res = pairs[0]
    s.check(pred_res.prediction_hash == rec.prediction_hash, "Original prediction completely unchanged after outcome join")
    s.check(out_res.actual_class == "H", "Actual outcome derived deterministically (2-1 -> H)")

    # Outcome before kickoff violation
    early_out_caught = False
    try:
        OutcomeRecord.create(fixture_id=900021, outcome_timestamp="2026-09-01T14:50:00+00:00", home_goals=1, away_goals=0, kickoff_timestamp=t_kick)
    except OutcomeJoinError:
        early_out_caught = True
    s.check(early_out_caught, "Outcome before kickoff strictly rejected")

    # Duplicate outcome join violation
    dup_out_caught = False
    try:
        dup_out = OutcomeRecord.create(fixture_id=900020, outcome_timestamp=t_out, home_goals=2, away_goals=1)
        store.join_outcome(dup_out)
    except OutcomeJoinError:
        dup_out_caught = True
    s.check(dup_out_caught, "Duplicate outcome join strictly rejected")

    # Negative goals violation
    neg_goals_caught = False
    try:
        OutcomeRecord.create(fixture_id=900022, outcome_timestamp=t_out, home_goals=-1, away_goals=0)
    except OutcomeJoinError:
        neg_goals_caught = True
    s.check(neg_goals_caught, "Negative goals in outcome strictly rejected")
    return s


def test_monitoring_and_statistical_gating() -> TestSuite:
    s = TestSuite("7. Continuous Monitoring & Statistical Decision Gating")
    contract = ProspectiveContract.load_and_verify()

    # Generate synthetic pairs
    pairs = []
    for i in range(100):
        t_pred, t_kick = f"2026-09-01T13:{i%60:02d}:00+00:00", "2026-09-01T15:00:00+00:00"
        t_out = "2026-09-01T17:00:00+00:00"
        p = LockedPredictionRecord.create_and_lock(
            fixture_id=910000 + i, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": 0.45, "D": 0.23, "A": 0.32},
            p_champion={"H": 0.43, "D": 0.26, "A": 0.31},
            p_draw_dc=0.25, p_draw_elo=0.26, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Premier League", home_team=f"H_{i}", away_team=f"A_{i}",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract,
        )
        o = OutcomeRecord.create(fixture_id=910000 + i, outcome_timestamp=t_out, home_goals=1 if i % 3 == 0 else 2, away_goals=1 if i % 3 == 0 else 0)
        pairs.append((p, o))

    metrics = ProspectiveMonitor.compute_metrics(pairs)
    s.check(metrics["n"] == 100, "Monitor computes metrics on 100 pairs")
    s.check(len(metrics["buckets_50"]) == 2, "Rolling 50-match buckets computed correctly (2 buckets)")

    # Premature statistical validation attempt (N=100 < 1050)
    premature_eval = run_final_statistical_validation(pairs)
    s.check(premature_eval["status"] == "VALIDATION_BLOCKED", "Premature validation (< 1,050) rejected as VALIDATION_BLOCKED")
    s.check(premature_eval["n_required"] == 1050, "Required threshold 1,050 explicitly reported")
    return s


def main() -> int:
    print("=" * 78)
    print("PROSPECTIVE VALIDATION PIPELINE — TEST SUITE")
    print("=" * 78)

    suites = [
        test_contract_and_hashes(),
        test_prediction_lock_and_timing(),
        test_probability_validation(),
        test_reused_300_exclusion(),
        test_store_and_tamper_detection(),
        test_two_stage_outcome_join(),
        test_monitoring_and_statistical_gating(),
    ]

    for st in suites:
        st.report()

    total_pass = sum(st.passes for st in suites)
    total_fail = sum(st.failures for st in suites)

    print("\n" + "=" * 78 + "\nOVERALL TEST SUMMARY\n" + "=" * 78)
    print(f"  Suites: {len(suites)}")
    print(f"  Pass:   {total_pass}")
    print(f"  Fail:   {total_fail}")
    print(f"  Total:  {total_pass + total_fail}")
    print(f"\n  VERDICT: {'ALL PROSPECTIVE PIPELINE TESTS PASSED' if total_fail == 0 else 'TESTS FAILED'}")

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
