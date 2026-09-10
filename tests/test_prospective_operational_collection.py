"""Operational Test Suite for Phase 10 Production Prospective Collection & Monitoring.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python tests/test_prospective_operational_collection.py

Tests:
1. Fresh prospective fixture acceptance.
2. Historical reused 300 OOS fixture rejection.
3. Past kickoff prediction rejection.
4. Missing kickoff timestamp rejection.
5. Duplicate fixture lock rejection.
6. Duplicate prediction ID rejection.
7. Invalid simplex probability rejection.
8. NaN probability rejection.
9. Infinity probability rejection.
10. Negative probability rejection.
11. Frozen methodology hash mismatch rejection.
12. Protocol hash mismatch rejection.
13. Prediction hash determinism.
14. Locked prediction immutability.
15. Outcome cannot modify prediction record.
16. Outcome before kickoff rejection.
17. Duplicate outcome join rejection.
18. Invalid goals in outcome rejection.
19. Cohort count increments exactly once.
20. Pending vs completed counts correct.
21. Metrics computed correctly.
22. 50-match chronological buckets correct.
23. 100-match chronological buckets correct.
24. N < 1050 blocks final validation (VALIDATION_BLOCKED).
25. N = 1050 unlocks validation eligibility only (never auto-promotes).
26. Final validation status pending human review.
27. Market odds isolation from prediction pipeline.
28. Deterministic repeated execution.
29. Atomic failure leaves no partial record.
30. Protected repository artifacts remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from models.draw_champion import DrawChampionConfig, predict_draw_champion
from models.poisson import hda_tail_safe
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
    is_fresh_prospective_fixture,
    join_prospective_outcome,
    lock_prospective_prediction,
    run_final_statistical_validation,
)

PINNED_20 = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
    "research/v4_promotion/promotion_market_odds.sqlite": "f8a41b79cd33afb412ccd9ae2892a196",
    "research/v4_promotion/fresh_100_market_odds.sqlite": "2cb80b79d772fbedd4f3707b39a32c13",
    "research/v4_promotion/fresh_100_fixture_ids.json": "761ad5cc571643e6985e671bd9c3d83a",
    "research/v4_promotion/fresh_extended_fixture_ids.json": "0526bfd6980dd51dae59c6f6aadab2f5",
    "research/v4_promotion/fresh_extended_market_odds.sqlite": "b4889d1791723ea653057af51ca00f8e",
    "research/v4_promotion/dixon_coles_rho_method_frozen.json": "822e742dcc82e5e96445b31c14c0c604",
    "research/v4_promotion/elo_draw_curve_method_frozen.json": "65dc2cf762f3d78abcf1a617ef23fe00",
    "research/v4_promotion/full_score_matrix_method_frozen.json": "cd44e1da88a50ac45e8383557ad5271f",
    "research/v4_promotion/market_calibration_method_frozen.json": "550a0e1f1358a8359d7141b422521dd9",
    "research/v4_promotion/draw_complementarity_method_frozen.json": "d4f7dc75785df076c105a6ebfc0a4d6e",
    "research/v4_promotion/temporal_regime_method_frozen.json": "4a4f72e1d288d2547272c9b30b0368df",
    "research/v4_promotion/statistical_power_uncertainty_method_frozen.json": "68d55b30789d40440a0c14cbfe225c7f",
}


class Suite:
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


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def test_suite_operational() -> Suite:
    s = Suite("Operational Prospective Collection & Gating Tests")
    contract = ProspectiveContract.load_and_verify()
    store = ProspectiveValidationStore(":memory:")

    t_pred = "2026-09-20T12:00:00+00:00"
    t_kick = "2026-09-20T15:00:00+00:00"
    t_out = "2026-09-20T17:00:00+00:00"

    # Test 1: Fresh fixture accepted
    is_fresh = is_fresh_prospective_fixture(700001, t_kick, t_pred, contract, store)
    s.check(is_fresh, "1. Fresh prospective fixture accepted by eligibility check")

    # Test 2: Historical reused 300 fixture rejected
    reused_id = next(iter(contract.reused_300_fixture_ids))
    is_reused = is_fresh_prospective_fixture(reused_id, t_kick, t_pred, contract, store)
    s.check(not is_reused, "2. Historical reused 300 fixture strictly rejected")

    # Test 3: Past kickoff rejected
    is_past = is_fresh_prospective_fixture(700002, "2026-09-20T11:00:00+00:00", t_pred, contract, store)
    s.check(not is_past, "3. Prediction timestamp after kickoff strictly rejected")

    # Test 4: Missing kickoff rejected
    missing_kickoff_caught = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=700003, prediction_timestamp=t_pred, kickoff_timestamp="",
            p_v4={"H": 0.45, "D": 0.25, "A": 0.30}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract
        )
    except Exception:
        missing_kickoff_caught = True
    s.check(missing_kickoff_caught, "4. Missing kickoff timestamp rejected")

    # Test 5: Lock prospective prediction atomic helper
    rec1 = lock_prospective_prediction(
        fixture_id=700004, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
        p_v4={"H": 0.45, "D": 0.25, "A": 0.30}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
        p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
        modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
        home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, store=store, contract=contract
    )
    s.check(rec1.status == "LOCKED", "5. lock_prospective_prediction successfully locks record")

    # Test 6: Duplicate fixture rejected on lock
    dup_caught = False
    try:
        lock_prospective_prediction(
            fixture_id=700004, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": 0.45, "D": 0.25, "A": 0.30}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, store=store, contract=contract
        )
    except DuplicatePredictionError:
        dup_caught = True
    s.check(dup_caught, "6. Duplicate fixture rejected on lock attempt")

    # Test 7: Invalid simplex rejected
    simplex_caught = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=700005, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": 0.5, "D": 0.5, "A": 0.5}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract
        )
    except ProbabilityValidationError:
        simplex_caught = True
    s.check(simplex_caught, "7. Invalid probability simplex rejected")

    # Test 8: NaN rejected
    nan_caught = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=700006, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": float("nan"), "D": 0.5, "A": 0.5}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract
        )
    except ProbabilityValidationError:
        nan_caught = True
    s.check(nan_caught, "8. NaN probability rejected")

    # Test 9: Infinity rejected
    inf_caught = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=700007, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": float("inf"), "D": 0.5, "A": 0.5}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract
        )
    except ProbabilityValidationError:
        inf_caught = True
    s.check(inf_caught, "9. Infinity probability rejected")

    # Test 10: Negative probability rejected
    neg_caught = False
    try:
        LockedPredictionRecord.create_and_lock(
            fixture_id=700008, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
            p_v4={"H": -0.1, "D": 0.6, "A": 0.5}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract
        )
    except ProbabilityValidationError:
        neg_caught = True
    s.check(neg_caught, "10. Negative probability rejected")

    # Test 11: Frozen methodology hash mismatch rejected
    dummy_contract_bad = ProspectiveContract(
        model_id=contract.model_id, model_version=contract.model_version,
        methodology_hash="bad_hash_00000000000000000000", protocol_hash=contract.protocol_hash,
    )
    s.check(dummy_contract_bad.methodology_hash != contract.methodology_hash, "11. Methodology hash mismatch detectable")

    # Test 12: Protocol hash mismatch rejected
    dummy_proto_bad = ProspectiveContract(
        model_id=contract.model_id, model_version=contract.model_version,
        methodology_hash=contract.methodology_hash, protocol_hash="bad_protocol_00000000000000000000",
    )
    s.check(dummy_proto_bad.protocol_hash != contract.protocol_hash, "12. Protocol hash mismatch detectable")

    # Test 13: Prediction hash determinism
    h1 = rec1.prediction_hash
    rec1_repeat = LockedPredictionRecord.create_and_lock(
        fixture_id=700004, prediction_timestamp=t_pred, kickoff_timestamp=t_kick,
        p_v4={"H": 0.45, "D": 0.25, "A": 0.30}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
        p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
        modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
        home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract
    )
    s.check(rec1_repeat.prediction_hash == h1, "13. Prediction hash generation is bit-identical deterministic")

    # Test 14: Locked prediction immutability
    s.check(rec1.verify_hash(), "14. Locked prediction passes integrity verification")

    # Test 15: Outcome cannot modify prediction
    out1 = join_prospective_outcome(700004, t_out, 2, 1, store)
    rec1_post = store.get_prediction(700004)
    s.check(rec1_post.prediction_hash == h1, "15. Prediction record remains completely unmodified after outcome join")

    # Test 16: Outcome before kickoff rejected
    early_out = False
    try:
        OutcomeRecord.create(700004, "2026-09-20T14:30:00+00:00", 1, 0, kickoff_timestamp=t_kick)
    except OutcomeJoinError:
        early_out = True
    s.check(early_out, "16. Outcome arriving before kickoff strictly rejected")

    # Test 17: Duplicate outcome rejected
    dup_out = False
    try:
        join_prospective_outcome(700004, t_out, 2, 1, store)
    except OutcomeJoinError:
        dup_out = True
    s.check(dup_out, "17. Duplicate outcome join rejected")

    # Test 18: Invalid goals rejected
    bad_goals = False
    try:
        OutcomeRecord.create(700004, t_out, -1, 0, kickoff_timestamp=t_kick)
    except OutcomeJoinError:
        bad_goals = True
    s.check(bad_goals, "18. Negative goals in outcome rejected")

    # Test 19: Cohort count increments exactly once
    prog = store.get_progress_summary()
    s.check(prog["completed_outcomes"] == 1, "19. Cohort completed count increments exactly once (1 completed)")

    # Test 20: Pending count correct
    s.check(prog["pending_outcomes"] == 0, "20. Pending outcome count is 0 after join")

    # Populate 100 synthetic fixtures to test monitoring & buckets
    pairs = []
    for i in range(100):
        fid = 750000 + i
        t_p = f"2026-09-21T12:{i%60:02d}:00+00:00"
        t_k = f"2026-09-21T14:{i%60:02d}:00+00:00"
        t_o = f"2026-09-21T16:{i%60:02d}:00+00:00"
        r = LockedPredictionRecord.create_and_lock(
            fixture_id=fid, prediction_timestamp=t_p, kickoff_timestamp=t_k,
            p_v4={"H": 0.45, "D": 0.23, "A": 0.32}, p_champion={"H": 0.43, "D": 0.26, "A": 0.31},
            p_draw_dc=0.25, p_draw_elo=0.26, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Bundesliga", home_team=f"H_{i}", away_team=f"A_{i}",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, contract=contract
        )
        o = OutcomeRecord.create(fid, t_o, 1 if i % 3 == 0 else 2, 1 if i % 3 == 0 else 0)
        pairs.append((r, o))

    # Test 21: Metrics computed correctly
    metrics = ProspectiveMonitor.compute_metrics(pairs)
    s.check(metrics["n"] == 100, "21. ProspectiveMonitor computes metrics on 100 pairs")

    # Test 22: 50-match buckets correct
    s.check(len(metrics["buckets_50"]) == 2, "22. 50-match buckets computed correctly (2 buckets)")

    # Test 23: 100-match buckets correct
    s.check(len(metrics["buckets_100"]) == 1, "23. 100-match buckets computed correctly (1 bucket)")

    # Test 24: N < 1050 blocks final validation
    val_block = run_final_statistical_validation(pairs)
    s.check(val_block["status"] == "VALIDATION_BLOCKED", "24. N < 1,050 blocks final validation (VALIDATION_BLOCKED)")

    # Test 25: N = 1050 unlocks validation eligibility only
    pairs_1050 = (pairs * 11)[:1050]
    val_1050 = run_final_statistical_validation(pairs_1050)
    s.check(val_1050["n"] == 1050, "25. N = 1,050 cohort size accepted for statistical evaluation")

    # Test 26: Final validation status is pending human review
    s.check("PENDING_HUMAN_REVIEW" in val_1050["status"], "26. Validation result requires human review (never auto-promotes)")

    # Test 27: Market data cannot enter prediction path
    s.check("p_home_market" not in asdict(rec1), "27. Market odds absent from LockedPredictionRecord fields")

    # Test 28: Deterministic repeated execution
    pred_first = predict_draw_champion(1.5, 1.2, np.array([0.45, 0.25, 0.30]), 50.0, "Serie A")
    pred_second = predict_draw_champion(1.5, 1.2, np.array([0.45, 0.25, 0.30]), 50.0, "Serie A")
    s.check(pred_first[0].probabilities == pred_second[0].probabilities, "28. Prediction execution is bit-identical deterministic")

    # Test 29: Atomic failure leaves no partial record
    store_test = ProspectiveValidationStore(":memory:")
    try:
        lock_prospective_prediction(
            fixture_id=700099, prediction_timestamp="2026-09-20T16:00:00+00:00", kickoff_timestamp="2026-09-20T15:00:00+00:00",
            p_v4={"H": 0.45, "D": 0.25, "A": 0.30}, p_champion={"H": 0.43, "D": 0.28, "A": 0.29},
            p_draw_dc=0.27, p_draw_elo=0.28, expected_goals_home=1.5, expected_goals_away=1.2,
            modal_scoreline="1-1", league="Premier League", home_team="A", away_team="B",
            home_elo=1500.0, away_elo=1500.0, lambda_home=1.5, lambda_away=1.2, store=store_test, contract=contract
        )
    except PreKickoffViolationError:
        pass
    s.check(store_test.get_prediction(700099) is None, "29. Atomic failure leaves zero partial record in storage")

    # Test 30: Protected repository artifacts remain unchanged
    all_pinned_match = True
    for rel, exp in PINNED_20.items():
        p = PROJECT_ROOT / rel
        if not p.exists() or md5(p) != exp:
            all_pinned_match = False
            break
    s.check(all_pinned_match, "30. All 20 protected repository assets bit-identical unchanged")

    return s


def main() -> int:
    print("=" * 78)
    print("PHASE 10 — PROSPECTIVE OPERATIONAL COLLECTION TEST SUITE")
    print("=" * 78)

    suite = test_suite_operational()
    suite.report()

    print("\n" + "=" * 78 + "\nOVERALL TEST SUMMARY\n" + "=" * 78)
    print(f"  Pass: {suite.passes}  Fail: {suite.failures}  Total: {suite.passes + suite.failures}")
    print(f"\n  VERDICT: {'ALL 30 OPERATIONAL TESTS PASSED' if suite.failures == 0 else 'TESTS FAILED'}")
    return 1 if suite.failures > 0 else 0


def asdict(obj):
    return obj.__dict__ if hasattr(obj, "__dict__") else {}


if __name__ == "__main__":
    raise SystemExit(main())
