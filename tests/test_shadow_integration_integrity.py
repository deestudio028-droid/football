"""Integration tests for Step 2E Shadow Integration integrity, hash checks, and reproduction."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP2E_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(STEP2E_DIR))

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_v4_baseline_integrity():
    """Verify that frozen V4.0 production baseline artifact is bit-identical."""
    assert V4_PATH.exists(), f"V4 artifact missing at {V4_PATH}"
    act_md5 = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act_md5 == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act_md5}"


def test_v4_1_candidate_integrity():
    """Verify that V4.1 candidate artifact is bit-identical."""
    assert V4_1_PATH.exists(), f"V4.1 artifact missing at {V4_1_PATH}"
    act_md5 = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act_md5 == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act_md5}"


def test_shadow_artifacts_exist():
    """Verify that all required shadow deliverables are present."""
    required = [
        "draw_probability_calibrator.py",
        "draw_calibrator_config.json",
        "shadow_predictions.csv",
        "shadow_33_match_audit.csv",
        "shadow_historical_reproduction.csv",
    ]
    for fn in required:
        p = STEP2E_DIR / fn
        assert p.exists(), f"Missing required file: {p}"


def test_historical_reproduction_accuracy():
    """Verify that historical reproduction error is strictly <= 1e-8."""
    repro_path = STEP2E_DIR / "shadow_historical_reproduction.csv"
    df = pd.read_csv(repro_path)
    assert len(df) >= 20
    max_err = df["abs_error_vs_ref"].max()
    assert max_err <= 1e-8, f"Reproduction error exceeded tolerance: {max_err}"


def test_shadow_predictions_invariants():
    """Verify that shadow predictions strictly satisfy probability invariants across all holdout matches."""
    preds_path = STEP2E_DIR / "shadow_predictions.csv"
    df = pd.read_csv(preds_path)
    assert len(df) == 1751

    # Check bounds
    assert (df["calibrated_home_prob"] >= 0.0).all()
    assert (df["calibrated_draw_prob"] >= 0.0).all()
    assert (df["calibrated_away_prob"] >= 0.0).all()
    assert (df["calibrated_home_prob"] <= 1.0).all()
    assert (df["calibrated_draw_prob"] <= 1.0).all()
    assert (df["calibrated_away_prob"] <= 1.0).all()

    # Check sums
    prob_sums = df["calibrated_home_prob"] + df["calibrated_draw_prob"] + df["calibrated_away_prob"]
    assert (abs(prob_sums - 1.0) < 1e-5).all()
