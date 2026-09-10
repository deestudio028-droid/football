"""Unit tests for V4.6 Prospective Outcome Reconciler.

Verifies:
1. Unlocked fixture outcome rejection.
2. Two-stage outcome separation (predictions table remains immutable).
3. Metric calculation and dashboard tracking.
4. Error economics computation (Good vs Bad vs Net gain).
5. Production baseline asset immutability across all 20 pinned hashes.
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

from monitoring.v46_live_prospective_collector import (
    V46LiveProspectiveCollector,
    init_live_prospective_db,
)
from monitoring.v46_outcome_reconciler import (
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
def temp_live_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_p = Path(tmpdir) / "test_live_prospective.sqlite"
        yield db_p

def test_unlocked_fixture_reconciliation_rejection(temp_live_db):
    """Verify reconciler rejects outcome for a fixture with no locked prediction."""
    init_live_prospective_db(temp_live_db)
    reconciler = V46OutcomeReconciler(db_path=temp_live_db)
    
    with pytest.raises(UnlockedFixtureOutcomeError):
        reconciler.reconcile_fixture_outcome(fixture_id=888801, home_goals=2, away_goals=1)

def test_successful_two_stage_reconciliation(temp_live_db):
    """Verify outcome is recorded separately without mutating locked predictions."""
    collector = V46LiveProspectiveCollector(db_path=temp_live_db)
    reconciler = V46OutcomeReconciler(db_path=temp_live_db)
    
    kickoff = datetime.now(timezone.utc) + timedelta(hours=2)
    pred_time = datetime.now(timezone.utc)
    
    # 1. Lock Prediction
    pred = collector.collect_and_lock_prediction(
        fixture_id=888802,
        competition_id=477,
        competition_name="Premier League",
        home_team="Liverpool",
        away_team="Man City",
        scheduled_kickoff=kickoff,
        p_v4=[0.41, 0.27, 0.32],
        p_v42=[0.36, 0.31, 0.33],
        abs_elo_diff=20.0,
        lambda_home=1.1,
        lambda_away=1.0,
        prediction_timestamp=pred_time,
    )
    
    orig_hash = pred["prediction_sha256"]
    assert pred["v4_6_final_decision"] == "D"
    
    # 2. Reconcile Final Match Outcome (1 - 1, Draw)
    res = reconciler.reconcile_fixture_outcome(fixture_id=888802, home_goals=1, away_goals=1)
    assert res["actual_outcome"] == "D"
    assert res["v4_6_correct"] is True
    assert res["v4_correct"] is False # V4 predicted H
    
    # 3. Check Dashboard Metrics
    dash = reconciler.compute_dashboard_metrics()
    assert dash["prospective_reconciled_N"] == 1
    assert dash["v4_accuracy_pct"] == 0.0
    assert dash["v4_6_accuracy_pct"] == 100.0
    assert dash["draw_predictions"] == 1
    assert dash["correct_draws"] == 1
    assert dash["draw_precision_pct"] == 100.0
    assert dash["error_economics"]["good_draw_overrides_GAINED"] == 1
    assert dash["error_economics"]["bad_draw_overrides_SACRIFICED"] == 0
    assert dash["error_economics"]["net_gain"] == 1
    
    # 4. Verify Prediction Table Immutable
    conn = sqlite3.connect(str(temp_live_db))
    cur = conn.cursor()
    cur.execute("SELECT prediction_sha256 FROM live_predictions WHERE fixture_id = 888802")
    cur_hash = cur.fetchone()[0]
    conn.close()
    assert cur_hash == orig_hash

def test_production_asset_integrity():
    """Verify all 20 protected repository assets are bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}"
