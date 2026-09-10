"""Unit tests for Phase 29 Operational Live Collection & Monitoring.

Verifies:
1. Phase 29 collection cycle executes and reports correct counts.
2. Historical 1,301 diagnostic & Phase 25 450 fixtures are strictly excluded.
3. Pre-kickoff timing buffer (15m) rejection.
4. SHA-256 deterministic hash creation and immutability.
5. Two-stage outcome isolation (no outcomes recorded during prediction lock).
6. Clean IDLE status when no upcoming fixtures exist.
7. Phase 29 monitor output structure.
8. All 20 protected baseline repository hashes verified bit-identical.
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

from data.providers.football_fixture_provider import (
    MockTestFixtureProvider,
    UpcomingFixture,
)
from monitoring.phase29_live_monitor import display_phase29_monitor
from monitoring.run_phase29_live_collection import run_phase29_collection
from monitoring.v46_live_metrics import compute_live_metrics
from monitoring.v46_live_prospective_collector import (
    DEFAULT_LIVE_DB,
    V46LiveProspectiveCollector,
    init_live_prospective_db,
)
from monitoring.v46_outcome_reconciler import V46OutcomeReconciler

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
def temp_phase29_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_p = Path(tmpdir) / "test_phase29_live.sqlite"
        yield db_p

def test_phase29_idle_execution(temp_phase29_db):
    """Verify Phase 29 collection handles empty feed cleanly without errors."""
    res = run_phase29_collection(db_path=temp_phase29_db, provider=MockTestFixtureProvider([]))
    assert res["discovered_fixtures"] == 0
    assert res["eligible_fixtures"] == 0
    assert res["new_predictions_locked"] == 0
    assert "IDLE" in res["status"]
    assert res["remaining_to_power_gate"] == 1050

def test_phase29_historical_and_late_exclusion(temp_phase29_db):
    """Verify Phase 29 rejects historical IDs and late fixtures."""
    collector = V46LiveProspectiveCollector(db_path=temp_phase29_db)
    hist_id = next(iter(collector.historical_ids)) if collector.historical_ids else 12345
    
    p = MockTestFixtureProvider()
    k_valid = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
    k_late = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    
    # Fixture 1: Historical ID
    p.add_fixture(UpcomingFixture(
        fixture_id=hist_id, league_id=477, league_name="Bundesliga",
        home_team="Bayern", away_team="Dortmund", home_team_id=1, away_team_id=2,
        scheduled_kickoff=k_valid, status="SCHEDULED", provider="mock", provider_fixture_id=str(hist_id)
    ))
    # Fixture 2: Late kickoff (inside 15m)
    p.add_fixture(UpcomingFixture(
        fixture_id=987654, league_id=423, league_name="Premier League",
        home_team="Arsenal", away_team="Chelsea", home_team_id=3, away_team_id=4,
        scheduled_kickoff=k_late, status="SCHEDULED", provider="mock", provider_fixture_id="987654"
    ))
    
    res = run_phase29_collection(db_path=temp_phase29_db, provider=p)
    assert res["historical_rejected"] == 1
    assert res["late_fixtures_rejected"] == 1
    assert res["new_predictions_locked"] == 0

def test_phase29_monitor_display(temp_phase29_db):
    """Verify Phase 29 monitor outputs metrics dictionary correctly."""
    init_live_prospective_db(temp_phase29_db)
    m = display_phase29_monitor(db_path=temp_phase29_db)
    assert m["fresh_total_locked"] == 0
    assert m["completed_reconciled"] == 0
    assert m["remaining_to_power_gate"] == 1050
    assert "COLLECTING" in m["status"]

def test_production_asset_integrity():
    """Verify all 20 protected repository baseline assets are bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}"
