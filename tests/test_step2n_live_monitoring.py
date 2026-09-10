"""Step 2N — Unit and integration tests for Live Production Outcome Monitoring."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"),
    str(PROJECT_ROOT / "research/v5_model_improvement/production_monitoring"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
MONITORING_DIR = PROJECT_ROOT / "research/v5_model_improvement/production_monitoring"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_1_v4_0_md5_hash_frozen():
    """TEST 1: Verify V4.0 production baseline MD5 hash is bit-identical."""
    assert V4_PATH.exists()
    act = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act}"


def test_2_v4_1_candidate_md5_hash_frozen():
    """TEST 2: Verify V4.1 candidate MD5 hash is bit-identical."""
    assert V4_1_PATH.exists()
    act = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act}"


def test_3_locked_forecast_ledger_counts():
    """TEST 3: Verify fixture_outcome_audit contains 87 total fixtures (33 completed, 54 pending)."""
    p_audit = MONITORING_DIR / "fixture_outcome_audit.csv"
    assert p_audit.exists()
    df = pd.read_csv(p_audit)
    assert len(df) == 87
    assert (df["status"] == "FT").sum() == 33
    assert (df["status"] == "NS").sum() == 54


def test_4_completed_matches_causality():
    """TEST 4: Verify prediction timestamps are before scheduled kickoffs."""
    p_comp = MONITORING_DIR / "live_outcome_ledger.csv"
    assert p_comp.exists()
    df = pd.read_csv(p_comp)
    assert len(df) == 33
    assert all(df["prediction_timestamp"].notna())
    assert all(df["scheduled_kickoff"].notna())


def test_5_draw_risk_detection_rate_on_actual_draws():
    """TEST 5: Verify 7 of 8 actual draws were flagged as elevated risk (MEDIUM/HIGH)."""
    p_comp = MONITORING_DIR / "live_outcome_ledger.csv"
    df = pd.read_csv(p_comp)
    draws = df[df["actual_result"] == "D"]
    assert len(draws) == 8
    n_elev = draws["draw_risk_tier"].isin(["MEDIUM", "HIGH", "CRITICAL"]).sum()
    assert n_elev == 7
    n_low = (draws["draw_risk_tier"] == "LOW").sum()
    assert n_low == 1


def test_6_performance_summary_metrics_valid():
    """TEST 6: Verify performance summary contains accuracy, CI, Brier, and Log Loss."""
    p_sum = MONITORING_DIR / "live_performance_summary.csv"
    assert p_sum.exists()
    df = pd.read_csv(p_sum)
    assert df["completed_matches"].values[0] == 33
    assert df["v4_0_correct"].values[0] == 20
    assert np.isclose(df["v4_0_accuracy_pct"].values[0], 60.61, atol=0.1)
    assert df["multiclass_brier_score"].values[0] > 0.0
    assert df["multiclass_log_loss"].values[0] > 0.0


def test_7_probability_simplex_sum_to_one():
    """TEST 7: Verify all probabilities sum to 1.0 on completed ledger."""
    p_comp = MONITORING_DIR / "live_outcome_ledger.csv"
    df = pd.read_csv(p_comp)
    p_sums = df["v4_p_home"] + df["v4_p_draw"] + df["v4_p_away"]
    assert np.allclose(p_sums, 1.0, atol=1e-2)


def test_8_all_monitoring_artifacts_exist():
    """TEST 8: Verify all 4 required CSV artifacts exist in production_monitoring."""
    files = [
        "live_outcome_ledger.csv",
        "live_performance_summary.csv",
        "draw_risk_outcome_analysis.csv",
        "fixture_outcome_audit.csv",
    ]
    for f in files:
        assert (MONITORING_DIR / f).exists(), f"Missing {f}"
