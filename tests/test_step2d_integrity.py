"""Unit tests for Step 2D Final Out-of-Sample Validation integrity and artifacts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
STEP2D_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2d_final_validation"

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


def test_step2d_artifacts_exist():
    """Verify that all required Step 2D CSV and JSON deliverables exist."""
    required_files = [
        "step2d_walkforward_results.csv",
        "step2d_league_results.csv",
        "step2d_probability_buckets.csv",
        "step2d_recent_33_audit.csv",
        "step2d_significance_results.csv",
        "step2d_sensitivity_results.csv",
        "step2d_candidate_config.json",
    ]
    for fn in required_files:
        p = STEP2D_DIR / fn
        assert p.exists(), f"Missing required deliverable: {p}"


def test_step2d_candidate_config_valid():
    """Verify that candidate config JSON is properly formatted."""
    cfg_path = STEP2D_DIR / "step2d_candidate_config.json"
    with open(cfg_path, "r") as f:
        data = json.load(f)
    assert "candidate_id" in data
    assert "parameters" in data
    assert "statistical_validation" in data
    assert data["statistical_validation"]["draw_ece_reduction_pct"] > 50.0
