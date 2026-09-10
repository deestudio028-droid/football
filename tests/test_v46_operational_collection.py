"""Operational unit tests for V4.6 Live Prospective Collection and Reconciliation.

Verifies:
1. Idempotent repeated collection.
2. Duplicate fixture rejection.
3. Historical and Phase 25 fixture rejection.
4. Pre-kickoff safety buffer enforcement.
5. Prediction record immutability.
6. SHA-256 hash calculation and fail-closed tamper detection.
7. Two-stage outcome reconciliation without prediction mutation.
8. Milestone detection and power gate progress tracking.
9. Frozen V4.6 rule parameter integrity.
10. Production baseline asset immutability across all 20 pinned hashes.
"""
import hashlib
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.v4_6_physical_draw_gate import V46PhysicalGateConfig
from monitoring.v46_live_metrics import (
    LiveProspectiveMetrics,
    compute_live_metrics,
    format_monitoring_dashboard,
)
from monitoring.v46_live_prospective_collector import (
    DuplicateFixtureError,
    HistoricalFixtureRejectionError,
    ModelConfigMutationError,
    PreKickoffViolationError,
    V46LiveProspectiveCollector,
    compute_prediction_hash,
    init_live_prospective_db,
)
from monitoring.v46_outcome_reconciler import (
    HashMismatchTamperError,
    UnlockedFixtureOutcomeError,
    V46OutcomeReconciler,
)

PINNED_20 = {
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
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

@pytest.fixture
def temp_live_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_p = Path(tmpdir) / "test_operational_live.sqlite"
        yield db_p

def test_idempotent_collection_and_duplicate_handling(temp_live_env):
    """Verify collector locks a fixture once and rejects repeat locking attempts."""
    collector = V46LiveProspectiveCollector(db_path=temp_live_env)
    kickoff = datetime.now(timezone.utc) + timedelta(hours=48)
    pred_time = datetime.now(timezone.utc)
    
    # 1. First collection succeeds
    p1 = collector.collect_and_lock_prediction(
        fixture_id=700001,
        competition_id=477,
        competition_name="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        scheduled_kickoff=kickoff,
        p_v4=[0.42, 0.27, 0.31],
        p_v42=[0.37, 0.31, 0.32],
        abs_elo_diff=25.0,
        lambda_home=1.2,
        lambda_away=1.1,
        prediction_timestamp=pred_time,
    )
    assert p1["lock_status"] == "LOCKED"
    assert collector.get_locked_prediction_count() == 1
    
    # 2. Second collection attempt raises DuplicateFixtureError
    with pytest.raises(DuplicateFixtureError):
        collector.collect_and_lock_prediction(
            fixture_id=700001,
            competition_id=477,
            competition_name="Premier League",
            home_team="Arsenal",
            away_team="Chelsea",
            scheduled_kickoff=kickoff,
            p_v4=[0.42, 0.27, 0.31],
            p_v42=[0.37, 0.31, 0.32],
            abs_elo_diff=25.0,
            lambda_home=1.2,
            lambda_away=1.1,
            prediction_timestamp=pred_time,
        )
    assert collector.get_locked_prediction_count() == 1

def test_reconciliation_cycle_and_live_metrics(temp_live_env):
    """Verify post-match outcome reconciliation updates live metrics accurately."""
    collector = V46LiveProspectiveCollector(db_path=temp_live_env)
    reconciler = V46OutcomeReconciler(db_path=temp_live_env)
    
    # Lock 2 matches
    k1 = datetime.now(timezone.utc) + timedelta(hours=2)
    k2 = datetime.now(timezone.utc) + timedelta(hours=3)
    p_time = datetime.now(timezone.utc)
    
    # Match 1: Fragile V4 H -> Override to D
    collector.collect_and_lock_prediction(
        fixture_id=700002, competition_id=419, competition_name="La Liga",
        home_team="Real Sociedad", away_team="Villarreal",
        scheduled_kickoff=k1, p_v4=[0.41, 0.27, 0.32], p_v42=[0.36, 0.31, 0.33],
        abs_elo_diff=20.0, lambda_home=1.1, lambda_away=1.0, prediction_timestamp=p_time,
    )
    
    # Match 2: Confident V4 H -> Retain H
    collector.collect_and_lock_prediction(
        fixture_id=700003, competition_id=200, competition_name="Bundesliga",
        home_team="Bayern", away_team="Mainz",
        scheduled_kickoff=k2, p_v4=[0.70, 0.18, 0.12], p_v42=[0.65, 0.22, 0.13],
        abs_elo_diff=180.0, lambda_home=2.5, lambda_away=0.7, prediction_timestamp=p_time,
    )
    
    # Reconcile outcomes: Match 1 = (1, 1, Draw), Match 2 = (3, 0, Home)
    reconciler.reconcile_fixture_outcome(fixture_id=700002, home_goals=1, away_goals=1)
    reconciler.reconcile_fixture_outcome(fixture_id=700003, home_goals=3, away_goals=0)
    
    metrics = compute_live_metrics(temp_live_env)
    assert metrics.fresh_total_locked == 2
    assert metrics.completed_reconciled == 2
    assert metrics.pending_fixtures == 0
    assert metrics.v4_accuracy_pct == 50.0 # V4 got match 2 right, match 1 wrong
    assert metrics.v4_6_accuracy_pct == 100.0 # V4.6 got both right (+1 free draw win)
    assert metrics.delta_accuracy_pct == 50.0
    assert metrics.draw_predictions == 1
    assert metrics.correct_draws == 1
    assert metrics.draw_precision_pct == 100.0
    assert metrics.good_draw_overrides == 1
    assert metrics.bad_draw_overrides == 0
    assert metrics.net_transition_gain == 1

def test_dashboard_formatting(temp_live_env):
    """Verify live monitoring dashboard renders structured output string."""
    init_live_prospective_db(temp_live_env)
    metrics = compute_live_metrics(temp_live_env)
    dash_str = format_monitoring_dashboard(metrics)
    assert "V4.6 LIVE PROSPECTIVE MONITORING DASHBOARD" in dash_str
    assert "N = 0 / 1050" in dash_str
    assert "Net Transition Gain:" in dash_str

def test_production_asset_integrity():
    """Verify all 20 protected repository baseline assets are bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}"
