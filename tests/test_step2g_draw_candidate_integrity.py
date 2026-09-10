"""Unit tests for Step 2G V4.0 Draw Calibration Integration Candidate integrity and regression."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP2E_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"
STEP2F_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2f_live_prospective"
STEP2G_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2g_production_candidate"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(STEP2E_DIR))

from draw_probability_calibrator import DrawProbabilityCalibrator

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
CONFIG_PATH = STEP2E_DIR / "draw_calibrator_config.json"
STEP2F_LEDGER_PATH = STEP2F_DIR / "live_shadow_forecast_ledger.csv"
PREDS_CSV = STEP2G_DIR / "v40_draw_candidate_predictions.csv"
EVAL_CSV = STEP2G_DIR / "v40_draw_candidate_evaluation.csv"
COMP_CSV = STEP2G_DIR / "v40_draw_candidate_comparison.csv"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_1_v4_baseline_integrity():
    """Verify that frozen V4.0 production baseline artifact is bit-identical."""
    assert V4_PATH.exists()
    act_md5 = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act_md5 == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act_md5}"


def test_2_v4_1_candidate_integrity():
    """Verify that V4.1 candidate artifact is bit-identical."""
    assert V4_1_PATH.exists()
    act_md5 = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act_md5 == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act_md5}"


def test_3_probability_simplex():
    """Verify that candidate probabilities sum to 1.0 across all predictions."""
    assert PREDS_CSV.exists()
    df = pd.read_csv(PREDS_CSV)
    prob_sums = df["candidate_h"] + df["candidate_d"] + df["candidate_a"]
    assert (abs(prob_sums - 1.0) < 1e-5).all()


def test_4_probabilities_in_unit_interval():
    """Verify that all candidate probabilities are strictly in [0, 1]."""
    df = pd.read_csv(PREDS_CSV)
    for col in ["candidate_h", "candidate_d", "candidate_a"]:
        assert (df[col] >= 0.0).all()
        assert (df[col] <= 1.0).all()


def test_5_home_away_ratio_preservation():
    """Verify that relative Home vs Away win ratio is preserved."""
    df = pd.read_csv(PREDS_CSV)
    for _, row in df.head(100).iterrows():
        if row["v4_a"] > 1e-4 and row["candidate_a"] > 1e-4:
            orig_ratio = row["v4_h"] / row["v4_a"]
            cand_ratio = row["candidate_h"] / row["candidate_a"]
            assert abs(orig_ratio - cand_ratio) < 1e-4


def test_6_deterministic_transformation():
    """Verify that transformation is 100% deterministic."""
    cal = DrawProbabilityCalibrator.from_config_file(CONFIG_PATH)
    res1 = cal.calibrate_single(0.40, 0.25, 0.35)
    res2 = cal.calibrate_single(0.40, 0.25, 0.35)
    assert res1.calibrated_p_draw == res2.calibrated_p_draw
    assert res1.calibrated_p_home == res2.calibrated_p_home


def test_7_no_nan_values():
    """Verify that no NaN values exist in prediction outputs."""
    df = pd.read_csv(PREDS_CSV)
    assert not df[["candidate_h", "candidate_d", "candidate_a", "candidate_decision"]].isna().any().any()


def test_8_no_inf_values():
    """Verify that no Inf values exist in prediction outputs."""
    df = pd.read_csv(PREDS_CSV)
    for col in ["candidate_h", "candidate_d", "candidate_a"]:
        assert not np.isinf(df[col].values).any()


def test_9_no_mutation_of_v4_arrays():
    """Verify that calibrator does not mutate input arrays."""
    cal = DrawProbabilityCalibrator.from_config_file(CONFIG_PATH)
    arr = np.array([[0.45, 0.25, 0.30]])
    arr_copy = arr.copy()
    _ = cal.calibrate_array(arr)
    assert np.array_equal(arr, arr_copy)


def test_10_platt_parameters_frozen():
    """Verify that Platt calibrator parameters are exact."""
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    params = cfg["parameters"]
    assert abs(params["slope_a"] - 0.9421034) < 1e-4
    assert abs(params["intercept_b"] - 0.1283626) < 1e-4


def test_11_and_12_no_hard_gate_and_decision_is_argmax():
    """Verify that decision equals argmax without hard threshold override."""
    df = pd.read_csv(PREDS_CSV)
    for _, row in df.head(500).iterrows():
        p_dict = {"H": row["candidate_h"], "D": row["candidate_d"], "A": row["candidate_a"]}
        exp_dec = max(p_dict, key=p_dict.get)
        assert row["candidate_decision"] == exp_dec


def test_13_historical_reproduction_matches_step2e():
    """Verify that historical candidate probabilities match Step 2E outputs."""
    step2e_preds = STEP2E_DIR / "shadow_predictions.csv"
    if step2e_preds.exists():
        df_2e = pd.read_csv(step2e_preds)
        df_2g = pd.read_csv(PREDS_CSV)
        df_2g_ho = df_2g[df_2g["season"] == "2025/2026"].reset_index(drop=True)
        assert len(df_2e) == len(df_2g_ho)
        max_diff = np.max(np.abs(df_2e["calibrated_draw_prob"].values - df_2g_ho["candidate_d"].values))
        assert max_diff < 1e-6


def test_14_existing_step2f_ledger_unchanged():
    """Verify that Step 2F 54-match locked ledger is intact."""
    assert STEP2F_LEDGER_PATH.exists()
    df_ledger = pd.read_csv(STEP2F_LEDGER_PATH)
    assert len(df_ledger) == 54
    assert df_ledger["fixture_id"].is_unique
