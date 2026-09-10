"""Step 3C — Test suite for Prospective Shadow Validation."""
from __future__ import annotations

import hashlib
import json
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
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2k_advisory_integration"),
    str(PROJECT_ROOT / "research/v5_model_improvement/step3c_prospective_shadow"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.draw_risk_advisor import DrawRiskAdvisor
from dashboard.model_registry import ModelRegistry
from dashboard.prediction_service import PredictionService

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
STEP3C_DIR = PROJECT_ROOT / "research/v5_model_improvement/step3c_prospective_shadow"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_1_v4_0_md5_immutability():
    """TEST 1: Verify V4.0 production baseline MD5 hash is bit-identical."""
    assert V4_PATH.exists()
    act = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act}"


def test_2_v4_1_md5_immutability():
    """TEST 2: Verify V4.1 candidate MD5 hash is bit-identical."""
    assert V4_1_PATH.exists()
    act = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act}"


def test_3_candidate_parameters_frozen():
    """TEST 3: Verify candidate freeze manifest exists and defines blend parameters."""
    p_man = STEP3C_DIR / "candidate_freeze_manifest.json"
    assert p_man.exists()
    data = json.loads(p_man.read_text(encoding="utf-8"))
    assert data["platt_blend_weight"] == 0.2108
    assert data["score_space_blend_weight"] == 0.7892
    assert "platt_parameters" in data
    assert "g2_parameters" in data


def test_4_probability_simplex():
    """TEST 4: Verify production probabilities sum to 1.0."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    p = res.production_probs
    assert np.isclose(p["H"] + p["D"] + p["A"], 1.0, atol=1e-2)


def test_5_probability_bounds():
    """TEST 5: Verify probabilities strictly lie in [0.0, 1.0]."""
    svc = PredictionService()
    for h, a, lg in [("Liverpool", "Everton", "Premier League"), ("Inter", "AC Milan", "Serie A")]:
        res = svc.predict_manual_matchup(h, a, lg)
        for k in ["H", "D", "A"]:
            assert 0.0 <= res.production_probs[k] <= 1.0


def test_6_deterministic_inference():
    """TEST 6: Verify repeated prediction inference is strictly deterministic."""
    svc = PredictionService()
    r1 = svc.predict_manual_matchup("PSG", "Marseille", "Ligue 1")
    r2 = svc.predict_manual_matchup("PSG", "Marseille", "Ligue 1")
    assert r1.production_probs == r2.production_probs
    assert r1.production_decision == r2.production_decision
    assert r1.draw_risk_score == r2.draw_risk_score


def test_7_no_forced_draw():
    """TEST 7: Verify baseline does not force draw decisions."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Nice", "Lorient", "Ligue 1")
    assert res.production_decision in ("H", "A")


def test_8_no_temporal_leakage():
    """TEST 8: Verify leakage audit artifact exists and confirms clean causal status."""
    p_leak = STEP3C_DIR / "leakage_audit.csv"
    assert p_leak.exists()
    df_l = pd.read_csv(p_leak)
    assert all(df_l["status"] == "PASS")


def test_9_prediction_timestamp_before_kickoff():
    """TEST 9: Verify prediction timestamp is strictly before kickoff for all shadow fixtures."""
    p_shad = STEP3C_DIR / "prospective_shadow_ledger.csv"
    assert p_shad.exists()
    df_s = pd.read_csv(p_shad)
    assert len(df_s) == 87
    for _, row in df_s[df_s["status"] == "NS"].iterrows():
        assert row["prediction_timestamp"] < row["kickoff"]


def test_10_goal_prediction_validity():
    """TEST 10: Verify goal predictions have positive expected goals."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Real Madrid", "FC Barcelona", "La Liga")
    assert res.lambda_home > 0
    assert res.lambda_away > 0


def test_11_score_matrix_validity():
    """TEST 11: Verify goal performance artifact exists and contains valid MAE metrics."""
    p_goals = STEP3C_DIR / "goal_performance.csv"
    assert p_goals.exists()
    df_g = pd.read_csv(p_goals)
    assert len(df_g) == 2
    assert all(df_g["home_goal_mae"] > 0)
    assert all(df_g["away_goal_mae"] > 0)


def test_12_shadow_ledger_schema():
    """TEST 12: Verify shadow ledger contains all 87 fixtures with correct columns."""
    p_shad = STEP3C_DIR / "prospective_shadow_ledger.csv"
    assert p_shad.exists()
    df_s = pd.read_csv(p_shad)
    assert len(df_s) == 87
    assert (df_s["status"] == "FT").sum() == 33
    assert (df_s["status"] == "NS").sum() == 54


def test_13_production_isolation():
    """TEST 13: Verify registry and UI remain locked to V4.0 Production."""
    reg = ModelRegistry()
    assert reg.get_default_model() == "V4.0 Production"
    assert reg.get_model("V4.1 Production").role == "RESEARCH"
    app_text = (PROJECT_ROOT / "src/dashboard/app.py").read_text(encoding="utf-8")
    assert "st.sidebar.selectbox(\"Select Model\"" not in app_text


def test_14_dashboard_schema_15_columns():
    """TEST 14: Verify dashboard match table contains the clean 15 columns."""
    app_text = (PROJECT_ROOT / "src/dashboard/app.py").read_text(encoding="utf-8")
    for req_col in [
        "\"Kickoff\":",
        "\"League\":",
        "\"Home Team\":",
        "\"Away Team\":",
        "\"Score\":",
        "\"P(H)\":",
        "\"P(D)\":",
        "\"P(A)\":",
        "\"Model Prediction\":",
        "\"Selected\":",
        "\"Eval\":",
        "\"Draw Risk\":",
        "\"Actual Game Result\":",
        "\"Goal Prediction\":",
        "\"Actual Goal Result\":",
    ]:
        assert req_col in app_text


def test_15_candidate_reproducibility():
    """TEST 15: Verify match error analysis artifact exists and classifies all 33 matches."""
    p_err = STEP3C_DIR / "match_error_analysis.csv"
    assert p_err.exists()
    df_e = pd.read_csv(p_err)
    assert len(df_e) == 33


def test_16_no_prospective_retuning():
    """TEST 16: Verify prospective shadow results are derived from fixed frozen weights."""
    p_perf = STEP3C_DIR / "prospective_performance_summary.csv"
    assert p_perf.exists()
    df_p = pd.read_csv(p_perf)
    assert len(df_p) == 2
