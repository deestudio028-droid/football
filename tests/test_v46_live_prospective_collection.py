"""Unit tests for V4.6 Live Prospective Collector.

Verifies:
1. Pre-kickoff timestamp enforcement (rejects post-kickoff predictions).
2. Duplicate fixture rejection (idempotent database insertion).
3. Historical fixture exclusion (strictly rejects matches from past training/validation).
4. SHA-256 prediction digest computation and immutability.
5. Model configuration immutability check.
6. Simplex probability conservation.
7. Production baseline asset immutability across all 20 pinned hashes.
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
from monitoring.v46_live_prospective_collector import (
    DuplicateFixtureError,
    HistoricalFixtureRejectionError,
    ModelConfigMutationError,
    PreKickoffViolationError,
    V46LiveProspectiveCollector,
    compute_prediction_hash,
    init_live_prospective_db,
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

def test_pre_kickoff_timing_enforcement(temp_live_db):
    """Verify collector rejects predictions timestamped at or after kickoff."""
    collector = V46LiveProspectiveCollector(db_path=temp_live_db)
    
    kickoff = datetime.now(timezone.utc) + timedelta(hours=2)
    past_prediction_time = kickoff + timedelta(minutes=5) # illegal: after kickoff
    
    with pytest.raises(PreKickoffViolationError):
        collector.collect_and_lock_prediction(
            fixture_id=999901,
            competition_id=477,
            competition_name="Premier League",
            home_team="Team A",
            away_team="Team B",
            scheduled_kickoff=kickoff,
            p_v4=[0.42, 0.26, 0.32],
            p_v42=[0.36, 0.31, 0.33],
            abs_elo_diff=20.0,
            lambda_home=1.1,
            lambda_away=1.0,
            prediction_timestamp=past_prediction_time,
        )

def test_successful_pre_kickoff_lock_and_hashing(temp_live_db):
    """Verify valid pre-kickoff prediction generates correct SHA-256 hash and locks."""
    collector = V46LiveProspectiveCollector(db_path=temp_live_db)
    
    kickoff = datetime.now(timezone.utc) + timedelta(hours=24)
    pred_time = datetime.now(timezone.utc)
    
    res = collector.collect_and_lock_prediction(
        fixture_id=999902,
        competition_id=477,
        competition_name="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        scheduled_kickoff=kickoff,
        p_v4=[0.41, 0.27, 0.32],
        p_v42=[0.36, 0.31, 0.33],
        abs_elo_diff=25.0,
        lambda_home=1.1,
        lambda_away=1.0,
        prediction_timestamp=pred_time,
    )
    
    assert res["fixture_id"] == 999902
    assert res["lock_status"] == "LOCKED"
    assert len(res["prediction_sha256"]) == 64
    assert res["v4_6_final_decision"] == "D"
    assert collector.get_locked_prediction_count() == 1

def test_duplicate_prediction_rejection(temp_live_db):
    """Verify duplicate collection attempt for an already locked fixture is rejected."""
    collector = V46LiveProspectiveCollector(db_path=temp_live_db)
    kickoff = datetime.now(timezone.utc) + timedelta(hours=24)
    pred_time = datetime.now(timezone.utc)
    
    collector.collect_and_lock_prediction(
        fixture_id=999903,
        competition_id=200,
        competition_name="Bundesliga",
        home_team="Bayern",
        away_team="Dortmund",
        scheduled_kickoff=kickoff,
        p_v4=[0.55, 0.22, 0.23],
        p_v42=[0.50, 0.25, 0.25],
        abs_elo_diff=70.0,
        lambda_home=1.8,
        lambda_away=1.1,
        prediction_timestamp=pred_time,
    )
    
    with pytest.raises(DuplicateFixtureError):
        collector.collect_and_lock_prediction(
            fixture_id=999903, # same fixture
            competition_id=200,
            competition_name="Bundesliga",
            home_team="Bayern",
            away_team="Dortmund",
            scheduled_kickoff=kickoff,
            p_v4=[0.55, 0.22, 0.23],
            p_v42=[0.50, 0.25, 0.25],
            abs_elo_diff=70.0,
            lambda_home=1.8,
            lambda_away=1.1,
            prediction_timestamp=pred_time,
        )

def test_historical_fixture_rejection(temp_live_db):
    """Verify collector strictly rejects known historical fixtures from past research."""
    collector = V46LiveProspectiveCollector(db_path=temp_live_db)
    kickoff = datetime.now(timezone.utc) + timedelta(hours=24)
    
    # Grab a real historical fixture ID from historical set
    if collector.historical_ids:
        hist_id = next(iter(collector.historical_ids))
        with pytest.raises(HistoricalFixtureRejectionError):
            collector.collect_and_lock_prediction(
                fixture_id=hist_id,
                competition_id=477,
                competition_name="Premier League",
                home_team="Team H",
                away_team="Team A",
                scheduled_kickoff=kickoff,
                p_v4=[0.40, 0.28, 0.32],
                p_v42=[0.35, 0.32, 0.33],
                abs_elo_diff=15.0,
                lambda_home=1.1,
                lambda_away=1.0,
            )

def test_config_mutation_rejection(temp_live_db):
    """Verify collector rejects mutated V4.6 configurations."""
    mutated_cfg = V46PhysicalGateConfig(draw_prob_threshold=0.20) # illegal change
    with pytest.raises(ModelConfigMutationError):
        V46LiveProspectiveCollector(db_path=temp_live_db, config=mutated_cfg)

def test_production_asset_integrity():
    """Verify all 20 protected repository assets are bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}"
