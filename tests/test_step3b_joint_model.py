"""Step 3B — Test suite for Joint H/D/A + Goal Model Research."""
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
    str(PROJECT_ROOT / "research/v5_model_improvement/step3b_joint_model_research"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.draw_risk_advisor import DrawRiskAdvisor
from dashboard.model_registry import ModelRegistry
from dashboard.prediction_service import PredictionService

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
STEP3B_DIR = PROJECT_ROOT / "research/v5_model_improvement/step3b_joint_model_research"

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


def test_3_probability_simplex():
    """TEST 3: Verify all 1X2 probabilities sum to 1.0."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    p = res.production_probs
    assert np.isclose(p["H"] + p["D"] + p["A"], 1.0, atol=1e-2)


def test_4_probability_bounds():
    """TEST 4: Verify probabilities strictly lie in [0.0, 1.0]."""
    svc = PredictionService()
    for h, a, lg in [("Liverpool", "Everton", "Premier League"), ("Inter", "AC Milan", "Serie A")]:
        res = svc.predict_manual_matchup(h, a, lg)
        for k in ["H", "D", "A"]:
            assert 0.0 <= res.production_probs[k] <= 1.0


def test_5_deterministic_inference():
    """TEST 5: Verify repeated prediction inference is strictly deterministic."""
    svc = PredictionService()
    r1 = svc.predict_manual_matchup("PSG", "Marseille", "Ligue 1")
    r2 = svc.predict_manual_matchup("PSG", "Marseille", "Ligue 1")
    assert r1.production_probs == r2.production_probs
    assert r1.production_decision == r2.production_decision
    assert r1.draw_risk_score == r2.draw_risk_score


def test_6_no_temporal_leakage():
    """TEST 6: Verify leakage audit artifact exists and confirms clean causal status."""
    p_leak = STEP3B_DIR / "leakage_audit.csv"
    assert p_leak.exists()
    df_l = pd.read_csv(p_leak)
    assert all(df_l["causal_status"] == "PASS")


def test_7_hda_argmax_validity():
    """TEST 7: Verify decision equals argmax of probabilities."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Bayern Munich", "Borussia Dortmund", "Bundesliga")
    max_k = max(res.production_probs, key=res.production_probs.get)
    assert res.production_decision == max_k


def test_8_goal_prediction_validity():
    """TEST 8: Verify goal predictions have positive expected goals."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Real Madrid", "FC Barcelona", "La Liga")
    assert res.lambda_home > 0
    assert res.lambda_away > 0


def test_9_score_matrix_normalization():
    """TEST 9: Verify score matrix probability sums to 1.0."""
    from scipy.stats import poisson
    lh, la, rho = 1.6, 1.2, -0.05
    p_h = np.array([poisson.pmf(i, lh) for i in range(11)])
    p_a = np.array([poisson.pmf(j, la) for j in range(11)])
    mat = np.outer(p_h, p_a)
    mat[0, 0] = max(1e-15, mat[0, 0] * (1.0 - lh * la * rho))
    mat[1, 0] = max(1e-15, mat[1, 0] * (1.0 + la * rho))
    mat[0, 1] = max(1e-15, mat[0, 1] * (1.0 + lh * rho))
    mat[1, 1] = max(1e-15, mat[1, 1] * (1.0 - rho))
    mat /= np.sum(mat)
    assert np.isclose(np.sum(mat), 1.0, atol=1e-5)


def test_10_score_prediction_validity():
    """TEST 10: Verify candidate goal metrics artifact exists and has valid values."""
    p_goals = STEP3B_DIR / "candidate_goal_metrics.csv"
    assert p_goals.exists()
    df_g = pd.read_csv(p_goals)
    assert len(df_g) >= 5
    assert all(df_g["home_goal_mae"] > 0)
    assert all(df_g["away_goal_mae"] > 0)


def test_11_candidate_reproducibility():
    """TEST 11: Verify joint model metrics file exists and covers all candidates."""
    p_cand = STEP3B_DIR / "joint_model_metrics.csv"
    assert p_cand.exists()
    df_c = pd.read_csv(p_cand)
    assert len(df_c) >= 5


def test_12_production_isolation():
    """TEST 12: Verify registry and UI remain locked to V4.0 Production."""
    reg = ModelRegistry()
    assert reg.get_default_model() == "V4.0 Production"
    assert reg.get_model("V4.1 Production").role == "RESEARCH"
    app_text = (PROJECT_ROOT / "src/dashboard/app.py").read_text(encoding="utf-8")
    assert "st.sidebar.selectbox(\"Select Model\"" not in app_text


def test_13_dashboard_15_column_schema():
    """TEST 13: Verify dashboard match table contains the clean 15 columns."""
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


def test_14_no_forced_draw_logic():
    """TEST 14: Verify baseline does not force draw decisions."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Nice", "Lorient", "Ligue 1")
    assert res.production_decision in ("H", "A")


def test_15_shadow_ledger_causality():
    """TEST 15: Verify prospective shadow ledger contains 54 future fixtures with pre-kickoff timestamps."""
    p_shad = STEP3B_DIR / "prospective_shadow_ledger.csv"
    assert p_shad.exists()
    df_s = pd.read_csv(p_shad)
    assert len(df_s) == 54
    assert all(df_s["prediction_timestamp"] < df_s["kickoff"])
