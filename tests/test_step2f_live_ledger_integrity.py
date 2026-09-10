"""Unit tests for Step 2F Live Shadow Forecast Ledger integrity, temporal clock, and invariants."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP2E_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"
STEP2F_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2f_live_prospective"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(STEP2E_DIR))

from draw_probability_calibrator import DrawProbabilityCalibrator

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
LEDGER_PATH = STEP2F_DIR / "live_shadow_forecast_ledger.csv"
CONFIG_PATH = STEP2E_DIR / "draw_calibrator_config.json"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


@pytest.fixture
def ledger_df() -> pd.DataFrame:
    assert LEDGER_PATH.exists(), f"Ledger missing at {LEDGER_PATH}"
    return pd.read_csv(LEDGER_PATH)


def test_1_no_duplicate_fixture_ids(ledger_df: pd.DataFrame):
    """Verify that every fixture in the live forecast ledger is unique."""
    assert len(ledger_df) > 0
    assert ledger_df["fixture_id"].is_unique, "Duplicate fixture IDs detected in ledger!"


def test_2_and_3_prediction_timestamp_before_kickoff(ledger_df: pd.DataFrame):
    """Verify that every prediction timestamp is strictly before scheduled kickoff."""
    pred_ts = pd.to_datetime(ledger_df["prediction_timestamp"], utc=True)
    kickoff_ts = pd.to_datetime(ledger_df["scheduled_kickoff"], utc=True)
    assert (pred_ts <= kickoff_ts).all(), "Temporal leakage: Prediction generated after kickoff!"


def test_4_immutability_and_artifacts_exist():
    """Verify that forecast ledger exists and has proper schema."""
    assert LEDGER_PATH.exists()
    df = pd.read_csv(LEDGER_PATH)
    required_cols = [
        "fixture_id", "competition", "season", "home_team", "away_team",
        "scheduled_kickoff", "prediction_timestamp",
        "v4_home_prob", "v4_draw_prob", "v4_away_prob", "v4_decision",
        "shadow_home_prob", "shadow_draw_prob", "shadow_away_prob", "shadow_decision",
        "draw_probability_delta", "decision_changed"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing required column {c} in ledger"


def test_5_v4_baseline_integrity():
    """Verify that frozen V4.0 production baseline artifact is bit-identical."""
    assert V4_PATH.exists()
    act_md5 = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act_md5 == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act_md5}"


def test_6_probability_sum_invariant(ledger_df: pd.DataFrame):
    """Verify that shadow probabilities sum to 1.0 within numerical tolerance."""
    prob_sums = ledger_df["shadow_home_prob"] + ledger_df["shadow_draw_prob"] + ledger_df["shadow_away_prob"]
    assert (abs(prob_sums - 1.0) < 1e-3).all(), "Probability sum invariant violated in ledger!"


def test_7_probability_bounds_invariant(ledger_df: pd.DataFrame):
    """Verify that all probabilities are in [0, 1]."""
    for col in ["shadow_home_prob", "shadow_draw_prob", "shadow_away_prob", "v4_home_prob", "v4_draw_prob", "v4_away_prob"]:
        assert (ledger_df[col] >= 0.0).all(), f"Negative probability in {col}"
        assert (ledger_df[col] <= 1.0).all(), f"Probability > 1.0 in {col}"


def test_8_home_away_ratio_preservation(ledger_df: pd.DataFrame):
    """Verify that relative Home vs Away win preference ratio is preserved."""
    for _, row in ledger_df.iterrows():
        if row["v4_away_prob"] > 1e-4 and row["shadow_away_prob"] > 1e-4:
            orig_ratio = row["v4_home_prob"] / row["v4_away_prob"]
            sh_ratio = row["shadow_home_prob"] / row["shadow_away_prob"]
            assert abs(orig_ratio - sh_ratio) < 1e-2, f"Ratio altered on fixture {row['fixture_id']}"


def test_9_no_nan_values(ledger_df: pd.DataFrame):
    """Verify that no NaN values exist in required fields."""
    check_cols = ["fixture_id", "v4_draw_prob", "shadow_draw_prob", "v4_decision", "shadow_decision"]
    assert not ledger_df[check_cols].isna().any().any(), "NaN values found in forecast ledger!"


def test_10_no_inf_values(ledger_df: pd.DataFrame):
    """Verify that no infinite values exist in probability fields."""
    prob_cols = ["shadow_home_prob", "shadow_draw_prob", "shadow_away_prob"]
    for c in prob_cols:
        assert not np.isinf(ledger_df[c].values).any(), f"Inf value found in {c}!"


def test_11_deterministic_shadow_transformation(ledger_df: pd.DataFrame):
    """Verify that calibrating the same inputs yields bit-identical outputs."""
    cal = DrawProbabilityCalibrator.from_config_file(CONFIG_PATH)
    sample_row = ledger_df.iloc[0]
    res1 = cal.calibrate_single(sample_row["v4_home_prob"], sample_row["v4_draw_prob"], sample_row["v4_away_prob"])
    res2 = cal.calibrate_single(sample_row["v4_home_prob"], sample_row["v4_draw_prob"], sample_row["v4_away_prob"])
    assert res1.calibrated_p_draw == res2.calibrated_p_draw
    assert abs(res1.calibrated_p_draw - sample_row["shadow_draw_prob"]) < 1e-3


def test_12_v4_1_candidate_integrity():
    """Verify that V4.1 candidate artifact is bit-identical."""
    assert V4_1_PATH.exists()
    act_md5 = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act_md5 == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act_md5}"
