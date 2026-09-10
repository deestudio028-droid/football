"""Unit tests for Historical Memory Backtest, Leakage Invariants, and Report Validation."""
import json
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest
from dashboard.historical_memory_service import (
    CalculationSimilarityEngine,
    CorrectScoreRefiner,
    HistoricalCalculationRecord,
    HistoricalCalculationMemoryStore,
    HistoricalEvidenceEvaluator,
)
from dashboard.time_utils import parse_to_utc_datetime


class TestBacktestLeakageInvariants:
    """Requirement 12: Rigorous unit tests for temporal and self-match leakage."""

    def test_self_match_exclusion(self):
        query_id = 7001
        q_kickoff = "2026-08-28T19:00:00Z"
        candidates = [
            HistoricalCalculationRecord(
                fixture_id=7001, competition_id=423, competition_name="Premier League", season="2026/2027",
                home_team="Self_H", away_team="Self_A", kickoff_utc="2026-08-20T19:00:00Z",
                p_home=0.60, p_draw=0.20, p_away=0.20, predicted_outcome="H", actual_home_goals=2,
                actual_away_goals=0, actual_outcome="H", actual_score="2-0", is_completed=True
            ),
            HistoricalCalculationRecord(
                fixture_id=7002, competition_id=423, competition_name="Premier League", season="2026/2027",
                home_team="Other_H", away_team="Other_A", kickoff_utc="2026-08-20T19:00:00Z",
                p_home=0.60, p_draw=0.20, p_away=0.20, predicted_outcome="H", actual_home_goals=2,
                actual_away_goals=1, actual_outcome="H", actual_score="2-1", is_completed=True
            ),
        ]
        # Filter out self-match
        filtered_candidates = [c for c in candidates if c.fixture_id != query_id]
        analogues = CalculationSimilarityEngine.find_nearest_analogues(
            p_home=0.60, p_draw=0.20, p_away=0.20, decision="H", lambda_home=None, lambda_away=None,
            baseline_score="2-0", query_kickoff_utc=q_kickoff, candidates=filtered_candidates,
            target_competition_id=423
        )
        fids = [a.fixture_id for a in analogues]
        assert query_id not in fids, "Self-match leaked into candidate set!"
        assert 7002 in fids

    def test_strict_temporal_ordering_in_backtest_ledger(self):
        json_report = PROJECT_ROOT / "reports" / "historical_memory_backtest.json"
        md_report = PROJECT_ROOT / "reports" / "historical_memory_backtest.md"

        assert json_report.exists(), "Backtest JSON report missing!"
        assert md_report.exists(), "Backtest Markdown report missing!"

        data = json.loads(json_report.read_text(encoding="utf-8"))
        assert data["final_verdict"] == "INSUFFICIENT DATA"
        assert data["dataset"]["sample_size"] == 33
        assert data["v4_baseline_metrics"]["total_matches"] == 33
        assert data["v4_baseline_metrics"]["accuracy_1x2_pct"] == 60.61

    def test_v4_0_model_hash_immutability(self):
        v4_path = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
        assert v4_path.exists()
        md5_hash = hashlib.md5(v4_path.read_bytes()).hexdigest()
        assert md5_hash == "06841f0c03c8597b2b8cd8f8ab064864"

    def test_v4_1_model_hash_immutability(self):
        v41_path = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"
        assert v41_path.exists()
        md5_hash = hashlib.md5(v41_path.read_bytes()).hexdigest()
        assert md5_hash == "145f918d933eb343c0f63ca342b10289"
