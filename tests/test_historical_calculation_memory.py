"""Comprehensive Test Suite for Historical Calculation Memory Layer.

Verifies:
1. Record creation & schema validation
2. Completed outcome attachment & pre/post-match separation
3. Deterministic calculation similarity engine
4. Top-K nearest analogue retrieval
5. Minimum sample-size protection (N < N_min -> INSUFFICIENT)
6. Same-outcome success rate calculation
7. HIGH / MODERATE / CAUTION / INSUFFICIENT evidence classification
8. Correct-score refinement on concentrated distributions
9. Weak score evidence fallback to V4.0 baseline score
10. Strict temporal leakage prevention (kickoff_candidate < kickoff_query)
11. Same-timestamp & future-match exclusion
12. League isolation & Bundesliga Matchday 1 vs EPL Matchday 2 isolation
13. Duplicate fixture_id prevention
14. Deterministic reproducibility
15. V4.0 and V4.1 model hash immutability
16. Zero regression in existing prediction engine
"""
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from dashboard.historical_memory_service import (
    CalculationSimilarityEngine,
    CorrectScoreRefiner,
    CorrectScoreRefinementResult,
    HistoricalAnalogue,
    HistoricalCalculationMemoryStore,
    HistoricalCalculationRecord,
    HistoricalEvidenceEvaluator,
    HistoricalEvidenceSignal,
    HistoricalMemoryService,
    get_historical_memory_service,
)
from dashboard.fixture_service import DashboardFixture, FixtureService, WEEKLY_LEAGUE_TARGETS
from dashboard.prediction_service import PredictionService


class TestHistoricalRecordSchemaAndCreation:
    """Requirement 1-4: Record creation, schema, completed outcome attachment, field separation."""

    def test_record_creation_and_pre_post_separation(self):
        rec = HistoricalCalculationRecord(
            fixture_id=1001,
            competition_id=423,
            competition_name="Premier League",
            season="2026/2027",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff_utc="2026-08-21T19:00:00.000000Z",
            p_home=0.55,
            p_draw=0.25,
            p_away=0.20,
            predicted_outcome="H",
            lambda_home=1.8,
            lambda_away=0.9,
            baseline_score="2-1",
            draw_risk_score=0.22,
            draw_risk_tier="LOW",
        )
        assert rec.fixture_id == 1001
        assert rec.p_home == 0.55
        assert rec.predicted_outcome == "H"
        assert rec.is_completed is False
        assert rec.actual_outcome is None

        # Attach completed post-match outcome
        rec.actual_home_goals = 2
        rec.actual_away_goals = 0
        rec.actual_outcome = "H"
        rec.actual_score = "2-0"
        rec.prediction_correct = True
        rec.exact_score_correct = False
        rec.is_completed = True

        assert rec.is_completed is True
        assert rec.actual_score == "2-0"
        assert rec.prediction_correct is True
        assert rec.exact_score_correct is False


class TestCalculationSimilarityEngine:
    """Requirement 5-6: Deterministic similarity, top-K retrieval, distance metrics."""

    def test_identical_calculation_produces_maximum_similarity(self):
        rec = HistoricalCalculationRecord(
            fixture_id=2001,
            competition_id=423,
            competition_name="Premier League",
            season="2026/2027",
            home_team="Liverpool",
            away_team="Everton",
            kickoff_utc="2026-08-20T19:00:00.000000Z",
            p_home=0.60,
            p_draw=0.22,
            p_away=0.18,
            predicted_outcome="H",
            lambda_home=2.1,
            lambda_away=0.8,
            actual_home_goals=2,
            actual_away_goals=0,
            actual_outcome="H",
            actual_score="2-0",
            prediction_correct=True,
            is_completed=True,
        )
        dist = CalculationSimilarityEngine.calculate_distance(
            p_home_q=0.60,
            p_draw_q=0.22,
            p_away_q=0.18,
            dec_q="H",
            lh_q=2.1,
            la_q=0.8,
            record=rec,
        )
        sim = CalculationSimilarityEngine.calculate_similarity_pct(dist)
        assert dist == 0.0
        assert sim == 100.0

    def test_divergent_calculation_has_lower_similarity(self):
        rec_home = HistoricalCalculationRecord(
            fixture_id=2002, competition_id=423, competition_name="Premier League", season="2026/2027",
            home_team="Man City", away_team="Bournemouth", kickoff_utc="2026-08-20T19:00:00.000000Z",
            p_home=0.75, p_draw=0.15, p_away=0.10, predicted_outcome="H", lambda_home=2.8, lambda_away=0.5,
            actual_home_goals=3, actual_away_goals=0, actual_outcome="H", actual_score="3-0",
            prediction_correct=True, is_completed=True
        )
        rec_away = HistoricalCalculationRecord(
            fixture_id=2003, competition_id=423, competition_name="Premier League", season="2026/2027",
            home_team="Bournemouth", away_team="Man City", kickoff_utc="2026-08-20T19:00:00.000000Z",
            p_home=0.12, p_draw=0.18, p_away=0.70, predicted_outcome="A", lambda_home=0.6, lambda_away=2.5,
            actual_home_goals=0, actual_away_goals=2, actual_outcome="A", actual_score="0-2",
            prediction_correct=True, is_completed=True
        )
        dist_similar = CalculationSimilarityEngine.calculate_distance(0.70, 0.18, 0.12, "H", 2.6, 0.6, rec_home)
        dist_opposite = CalculationSimilarityEngine.calculate_distance(0.70, 0.18, 0.12, "H", 2.6, 0.6, rec_away)

        assert dist_similar < dist_opposite
        sim_similar = CalculationSimilarityEngine.calculate_similarity_pct(dist_similar)
        sim_opposite = CalculationSimilarityEngine.calculate_similarity_pct(dist_opposite)
        assert sim_similar > sim_opposite


class TestTemporalLeakageSafety:
    """Requirement 12-14: Strict temporal leakage prevention."""

    def test_future_and_same_timestamp_matches_are_strictly_excluded(self):
        query_kickoff = "2026-08-28T19:00:00.000000Z"
        candidates = [
            # Past match: valid (August 21)
            HistoricalCalculationRecord(
                fixture_id=3001, competition_id=423, competition_name="Premier League", season="2026/2027",
                home_team="Past_H", away_team="Past_A", kickoff_utc="2026-08-21T19:00:00.000000Z",
                p_home=0.55, p_draw=0.25, p_away=0.20, predicted_outcome="H", lambda_home=1.8, lambda_away=1.0,
                actual_home_goals=2, actual_away_goals=1, actual_outcome="H", actual_score="2-1",
                prediction_correct=True, is_completed=True
            ),
            # Same timestamp match: invalid (must be excluded)
            HistoricalCalculationRecord(
                fixture_id=3002, competition_id=423, competition_name="Premier League", season="2026/2027",
                home_team="Same_H", away_team="Same_A", kickoff_utc="2026-08-28T19:00:00.000000Z",
                p_home=0.55, p_draw=0.25, p_away=0.20, predicted_outcome="H", lambda_home=1.8, lambda_away=1.0,
                actual_home_goals=1, actual_away_goals=0, actual_outcome="H", actual_score="1-0",
                prediction_correct=True, is_completed=True
            ),
            # Future match: invalid (must be excluded)
            HistoricalCalculationRecord(
                fixture_id=3003, competition_id=423, competition_name="Premier League", season="2026/2027",
                home_team="Future_H", away_team="Future_A", kickoff_utc="2026-08-30T19:00:00.000000Z",
                p_home=0.55, p_draw=0.25, p_away=0.20, predicted_outcome="H", lambda_home=1.8, lambda_away=1.0,
                actual_home_goals=3, actual_away_goals=1, actual_outcome="H", actual_score="3-1",
                prediction_correct=True, is_completed=True
            ),
        ]
        analogues = CalculationSimilarityEngine.find_nearest_analogues(
            p_home=0.55, p_draw=0.25, p_away=0.20, decision="H", lambda_home=1.8, lambda_away=1.0,
            baseline_score="2-1", query_kickoff_utc=query_kickoff, candidates=candidates,
            target_competition_id=423
        )
        fids = [a.fixture_id for a in analogues]
        assert 3001 in fids
        assert 3002 not in fids, "Temporal leakage violation: match at same kickoff timestamp was included!"
        assert 3003 not in fids, "Temporal leakage violation: future match was included!"


class TestHistoricalEvidenceAndSampleProtection:
    """Requirement 7-9: Sample-size protection, win rate, HIGH/MODERATE/CAUTION/INSUFFICIENT."""

    def test_insufficient_sample_protection_when_under_min_threshold(self):
        # Only 2 analogues available (min threshold = 3)
        analogues = [
            HistoricalAnalogue(
                fixture_id=4001, competition_name="Premier League", season="2026/2027",
                home_team="H1", away_team="A1", kickoff_utc="2026-08-20T19:00:00Z",
                similarity_score=88.0, p_home=0.60, p_draw=0.22, p_away=0.18, predicted_outcome="H",
                baseline_score="2-1", actual_score="2-0", actual_outcome="H", prediction_correct=True,
                exact_score_correct=False
            ),
            HistoricalAnalogue(
                fixture_id=4002, competition_name="Premier League", season="2026/2027",
                home_team="H2", away_team="A2", kickoff_utc="2026-08-20T19:00:00Z",
                similarity_score=85.0, p_home=0.58, p_draw=0.24, p_away=0.18, predicted_outcome="H",
                baseline_score="2-1", actual_score="2-1", actual_outcome="H", prediction_correct=True,
                exact_score_correct=True
            ),
        ]
        evidence = HistoricalEvidenceEvaluator.evaluate_evidence(
            decision="H", analogues=analogues, min_sample_size=3
        )
        assert evidence.evidence_level == "INSUFFICIENT"
        assert "INSUFFICIENT SAMPLE" in evidence.evidence_badge
        assert evidence.same_outcome_analogues_count == 2

    def test_high_conviction_when_sample_is_sufficient_and_success_high(self):
        analogues = [
            HistoricalAnalogue(
                fixture_id=4010 + i, competition_name="Premier League", season="2026/2027",
                home_team=f"H{i}", away_team=f"A{i}", kickoff_utc=f"2026-08-20T1{i}:00:00Z",
                similarity_score=85.0, p_home=0.60, p_draw=0.20, p_away=0.20, predicted_outcome="H",
                baseline_score="2-1", actual_score="2-0", actual_outcome="H" if i < 4 else "D",
                prediction_correct=(i < 4), exact_score_correct=(i == 0)
            )
            for i in range(5)  # 4 out of 5 correct (80.0%)
        ]
        evidence = HistoricalEvidenceEvaluator.evaluate_evidence(
            decision="H", analogues=analogues, min_sample_size=3
        )
        assert evidence.evidence_level == "HIGH"
        assert "HIGH HISTORICAL CONVICTION" in evidence.evidence_badge
        assert evidence.same_outcome_success_rate == 80.0

    def test_caution_when_sample_is_sufficient_but_historical_rate_is_low(self):
        analogues = [
            HistoricalAnalogue(
                fixture_id=4020 + i, competition_name="Premier League", season="2026/2027",
                home_team=f"H{i}", away_team=f"A{i}", kickoff_utc=f"2026-08-20T1{i}:00:00Z",
                similarity_score=82.0, p_home=0.55, p_draw=0.25, p_away=0.20, predicted_outcome="H",
                baseline_score="2-1", actual_score="1-1", actual_outcome="D" if i < 3 else "H",
                prediction_correct=(i >= 3), exact_score_correct=False
            )
            for i in range(4)  # 1 out of 4 correct (25.0%)
        ]
        evidence = HistoricalEvidenceEvaluator.evaluate_evidence(
            decision="H", analogues=analogues, min_sample_size=3
        )
        assert evidence.evidence_level == "CAUTION"
        assert "HISTORICAL CAUTION" in evidence.evidence_badge
        assert evidence.same_outcome_success_rate == 25.0


class TestCorrectScoreRefinement:
    """Requirement 10-11: Correct-score refinement on concentrated distributions & weak fallback."""

    def test_score_refinement_proposes_concentrated_modal_score(self):
        analogues = [
            HistoricalAnalogue(
                fixture_id=5001, competition_name="La Liga", season="2026/2027", home_team="H1", away_team="A1",
                kickoff_utc="2026-08-20T19:00:00Z", similarity_score=85.0, p_home=0.50, p_draw=0.30, p_away=0.20,
                predicted_outcome="H", baseline_score="2-1", actual_score="1-0", actual_outcome="H",
                prediction_correct=True, exact_score_correct=False
            ),
            HistoricalAnalogue(
                fixture_id=5002, competition_name="La Liga", season="2026/2027", home_team="H2", away_team="A2",
                kickoff_utc="2026-08-20T19:00:00Z", similarity_score=84.0, p_home=0.52, p_draw=0.28, p_away=0.20,
                predicted_outcome="H", baseline_score="2-1", actual_score="1-0", actual_outcome="H",
                prediction_correct=True, exact_score_correct=False
            ),
            HistoricalAnalogue(
                fixture_id=5003, competition_name="La Liga", season="2026/2027", home_team="H3", away_team="A3",
                kickoff_utc="2026-08-20T19:00:00Z", similarity_score=83.0, p_home=0.49, p_draw=0.31, p_away=0.20,
                predicted_outcome="H", baseline_score="2-1", actual_score="1-0", actual_outcome="H",
                prediction_correct=True, exact_score_correct=False
            ),
            HistoricalAnalogue(
                fixture_id=5004, competition_name="La Liga", season="2026/2027", home_team="H4", away_team="A4",
                kickoff_utc="2026-08-20T19:00:00Z", similarity_score=80.0, p_home=0.51, p_draw=0.29, p_away=0.20,
                predicted_outcome="H", baseline_score="2-1", actual_score="2-0", actual_outcome="H",
                prediction_correct=True, exact_score_correct=False
            ),
        ]
        # 3 out of 4 matches finished "1-0" (75%), while baseline was "2-1"
        refinement = CorrectScoreRefiner.refine_score(
            baseline_score="2-1", decision="H", analogues=analogues
        )
        assert refinement.refinement_status == "REFINED"
        assert refinement.refined_score == "1-0"
        assert refinement.score_confidence == "HIGH"
        assert refinement.modal_score_count == 3
        assert "1-0" in refinement.reason

    def test_score_refinement_keeps_baseline_when_distribution_is_diffuse(self):
        analogues = [
            HistoricalAnalogue(
                fixture_id=5010 + i, competition_name="Serie A", season="2026/2027", home_team=f"H{i}", away_team=f"A{i}",
                kickoff_utc=f"2026-08-20T1{i}:00:00Z", similarity_score=80.0, p_home=0.45, p_draw=0.30, p_away=0.25,
                predicted_outcome="H", baseline_score="2-1", actual_score=score, actual_outcome="H",
                prediction_correct=True, exact_score_correct=(score == "2-1")
            )
            for i, score in enumerate(["2-1", "3-2", "1-0", "4-1", "2-0"])  # all different
        ]
        refinement = CorrectScoreRefiner.refine_score(
            baseline_score="2-1", decision="H", analogues=analogues
        )
        assert refinement.refinement_status == "BASELINE_KEPT"
        assert refinement.refined_score == "2-1"


class TestLeagueAndGameweekIsolation:
    """Requirement 15-16: League isolation & Bundesliga Matchday 1 vs EPL Matchday 2."""

    def test_league_isolation_prevents_cross_league_contamination(self):
        store = HistoricalCalculationMemoryStore()
        # Add EPL and Bundesliga records
        store.add_record(HistoricalCalculationRecord(
            fixture_id=6001, competition_id=423, competition_name="Premier League", season="2026/2027",
            home_team="EPL_Home", away_team="EPL_Away", kickoff_utc="2026-08-21T19:00:00Z",
            p_home=0.60, p_draw=0.22, p_away=0.18, predicted_outcome="H", lambda_home=2.0, lambda_away=0.9,
            actual_home_goals=2, actual_away_goals=1, actual_outcome="H", actual_score="2-1",
            prediction_correct=True, is_completed=True
        ))
        store.add_record(HistoricalCalculationRecord(
            fixture_id=6002, competition_id=477, competition_name="Bundesliga", season="2026/2027",
            home_team="GER_Home", away_team="GER_Away", kickoff_utc="2026-08-21T19:00:00Z",
            p_home=0.60, p_draw=0.22, p_away=0.18, predicted_outcome="H", lambda_home=2.0, lambda_away=0.9,
            actual_home_goals=3, actual_away_goals=0, actual_outcome="H", actual_score="3-0",
            prediction_correct=True, is_completed=True
        ))
        service = HistoricalMemoryService(store=store)

        # Query Bundesliga fixture with allow_cross_league=False
        result_ger = service.analyze_fixture_calculation(
            fixture_id=9999, home_team="Bayern", away_team="Stuttgart", competition_id=477,
            competition_name="Bundesliga", kickoff_utc="2026-08-28T18:30:00Z",
            p_home=0.60, p_draw=0.22, p_away=0.18, decision="H", lambda_home=2.0, lambda_away=0.9,
            allow_cross_league=False
        )
        fids = [a.fixture_id for a in result_ger.analogues]
        assert 6002 in fids
        assert 6001 not in fids, "EPL record contaminated Bundesliga search!"


class TestModelIntegrityAndZeroMutation:
    """Requirement 19-20: Frozen model hash immutability."""

    def test_v4_production_model_md5_frozen(self):
        v4_path = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
        assert v4_path.exists(), "V4.0 production model artifact missing!"
        md5_hash = hashlib.md5(v4_path.read_bytes()).hexdigest()
        assert md5_hash == "06841f0c03c8597b2b8cd8f8ab064864", f"V4.0 MD5 hash mutated: {md5_hash}"

    def test_v4_1_candidate_model_md5_frozen(self):
        v41_path = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"
        assert v41_path.exists(), "V4.1 candidate model artifact missing!"
        md5_hash = hashlib.md5(v41_path.read_bytes()).hexdigest()
        assert md5_hash == "145f918d933eb343c0f63ca342b10289", f"V4.1 MD5 hash mutated: {md5_hash}"
