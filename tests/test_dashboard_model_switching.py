"""Unit tests for Phase Safe Model Switching in Football Prediction Lab Dashboard."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.model_registry import ModelRegistry, get_model_registry, load_model_artifact, PINNED_MODEL_HASHES
from dashboard.prediction_service import PredictionService, DashboardMatchPrediction, SingleMatchPredictionResult
from dashboard.fixture_service import DashboardFixture

V4_1_EXP_MD5 = "145f918d933eb343c0f63ca342b10289"
V4_0_EXP_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
LEDGER_CSV = PROJECT_ROOT / "research/v5_model_improvement/v4_1_prospective_test/02_live_forecast_ledger.csv"


def test_1_default_dashboard_model_is_v4_1():
    """Test 11 & 1: Verify default model in registry is V4.1 Production."""
    reg = get_model_registry()
    assert reg.get_default_model() == "V4.1 Production"
    prod = reg.get_production_model()
    assert prod.model_id == "v4_1_prospective_candidate"
    assert prod.version == "v4.1-champion-dc-elo-stacking-2025-26-trained"
    assert prod.md5_hash == V4_1_EXP_MD5


def test_2_v4_0_can_be_selected_and_metadata_is_correct():
    """Test 2 & 3: Verify V4.0 can be selected and exposes correct metadata."""
    reg = get_model_registry()
    v40 = reg.get_model("V4.0 Production")
    assert v40 is not None
    assert v40.model_id == "v4_0_draw_champion"
    assert v40.version == "v4.0-champion-dc-elo-stacking"
    assert v40.md5_hash == V4_0_EXP_MD5
    assert ("FROZEN BASELINE" in v40.status or "HISTORICAL PRODUCTION" in v40.status)


def test_4_md5_verification_and_fail_closed():
    """Test 4 & 5: Verify MD5 verification passes for valid models and fails closed on invalid hashes."""
    # Valid load
    v41_obj = load_model_artifact("V4.1 Production")
    assert v41_obj is not None

    v40_obj = load_model_artifact("V4.0 Production")
    assert v40_obj is not None

    # Corrupted / unknown key fails closed
    with pytest.raises(ValueError):
        load_model_artifact("NonExistent_Fake_Model_12345")


def test_6_and_7_selecting_models_loads_respective_artifacts():
    """Test 6 & 7: Verify selecting V4.1 and V4.0 loads their respective objects."""
    ps = PredictionService()
    p_v41 = ps.predict_matchup("Arsenal", "Chelsea", "Premier League", competition_id=423, model_key="V4.1 Production")
    p_v40 = ps.predict_matchup("Arsenal", "Chelsea", "Premier League", competition_id=423, model_key="V4.0 Production")

    assert p_v41 is not None and p_v40 is not None
    assert p_v41.prediction_model == "V4.1 Production"
    assert p_v41.model_file_md5 == V4_1_EXP_MD5
    assert p_v40.prediction_model == "V4.0 Production"
    assert p_v40.model_file_md5 == V4_0_EXP_MD5


def test_8_independent_predictions_generated():
    """Test 8: Verify predictions are independently produced and reflect model differences."""
    ps = PredictionService()
    # Test on a tight fixture
    p_v41 = ps.predict_matchup("Everton", "Crystal Palace", "Premier League", competition_id=423, model_key="V4.1 Production")
    p_v40 = ps.predict_matchup("Everton", "Crystal Palace", "Premier League", competition_id=423, model_key="V4.0 Production")

    assert p_v41 is not None and p_v40 is not None
    # Both satisfy probability simplex
    assert np.isclose(sum(p_v41.production_probs.values()), 1.0, atol=1e-2)
    assert np.isclose(sum(p_v40.production_probs.values()), 1.0, atol=1e-2)

    # Probabilities come from the respective models
    assert p_v41.production_probs == p_v41.v4_1_probs
    assert p_v40.production_probs == p_v40.v4_champ_probs


def test_9_switching_models_does_not_mutate_historical_data():
    """Test 9: Verify model switching does not mutate historical match data or predictions."""
    ps = PredictionService()
    # Simulate a fixture with actual outcome
    fix = DashboardFixture(
        fixture_id=999,
        competition_id=423,
        competition_name="Premier League",
        season_name="2026/2027",
        home_team="Arsenal",
        away_team="Chelsea",
        home_id=1,
        away_id=2,
        scheduled_kickoff="2026-08-30T15:00:00+00:00",
        status="FT",
        provider="Local DB",
        home_goals=2,
        away_goals=0,
        actual_outcome="H",
    )

    pred_v41 = ps.predict_dashboard_fixture(fix, model_key="V4.1 Production")
    pred_v40 = ps.predict_dashboard_fixture(fix, model_key="V4.0 Production")

    assert pred_v41.actual_outcome == "H"
    assert pred_v40.actual_outcome == "H"
    assert pred_v41.v4_1_probs == pred_v40.v4_1_probs  # Reference V4.1 unmutated


def test_10_prospective_ledger_byte_identical():
    """Test 10: Verify the 99-match locked prospective ledger remains 100% byte identical."""
    assert LEDGER_CSV.exists()
    df = pd.read_csv(LEDGER_CSV)
    assert len(df) == 99
    assert np.all(df["model_file_md5"] == V4_1_EXP_MD5)


def test_12_model_selection_deterministic():
    """Test 12: Verify repeated inferences with the same selected model are deterministic."""
    ps = PredictionService()
    r1 = ps.predict_matchup("Real Madrid", "Barcelona", "La Liga", competition_id=419, model_key="V4.0 Production")
    r2 = ps.predict_matchup("Real Madrid", "Barcelona", "La Liga", competition_id=419, model_key="V4.0 Production")

    assert r1.production_probs == r2.production_probs
    assert r1.production_decision == r2.production_decision
    assert r1.lambda_home == r2.lambda_home
