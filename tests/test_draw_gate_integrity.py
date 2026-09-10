"""Unit tests for Step 2B Draw Gate integrity, hash safety, and gate logic."""
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
CONFIG_PATH = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/draw_gate_config.json"
RESULTS_CSV = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/draw_gate_candidates.csv"

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


def test_draw_gate_config_exists():
    """Verify that draw gate candidate configuration is saved and valid."""
    assert CONFIG_PATH.exists(), f"Config missing at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    assert "parameters" in data
    assert "tau_d" in data["parameters"]
    assert "tau_gap" in data["parameters"]
    assert "tau_conf" in data["parameters"]


def test_draw_gate_candidates_csv_exists():
    """Verify that all 5 candidate strategies are represented in candidate results."""
    assert RESULTS_CSV.exists(), f"CSV missing at {RESULTS_CSV}"
    import pandas as pd
    df = pd.read_csv(RESULTS_CSV)
    assert len(df) >= 15
    strategies = df["Strategy"].unique()
    assert len(strategies) == 5
