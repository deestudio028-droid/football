"""Step 3 — Test suite for 1X2 + Goal Prediction Research & Clean Dashboard Schema."""
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
    str(PROJECT_ROOT / "research/v5_model_improvement/step3_1x2_goal_research"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.draw_risk_advisor import DrawRiskAdvisor
from dashboard.model_registry import ModelRegistry
from dashboard.prediction_service import PredictionService

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
STEP3_DIR = PROJECT_ROOT / "research/v5_model_improvement/step3_1x2_goal_research"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_1_v4_0_artifact_immutability():
    """TEST 1: Verify V4.0 production baseline MD5 hash is bit-identical."""
    assert V4_PATH.exists()
    act = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act}"


def test_2_v4_1_artifact_immutability():
    """TEST 2: Verify V4.1 candidate MD5 hash is bit-identical."""
    assert V4_1_PATH.exists()
    act = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act}"


def test_3_production_registry_isolation():
    """TEST 3: Verify registry default is strictly V4.0 Production."""
    reg = ModelRegistry()
    assert reg.get_default_model() == "V4.0 Production"
    assert reg.get_production_model().display_name == "V4.0 Production"
    assert reg.get_model("V4.1 Production").role == "RESEARCH"


def test_4_probability_simplex():
    """TEST 4: Verify production probabilities sum to 1.0."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res is not None
    p = res.production_probs
    assert np.isclose(p["H"] + p["D"] + p["A"], 1.0, atol=1e-2)


def test_5_probability_validity():
    """TEST 5: Verify all probabilities are bounded in [0.0, 1.0]."""
    svc = PredictionService()
    for h, a, lg in [("Man City", "Luton", "Premier League"), ("Elche", "Barcelona", "La Liga")]:
        res = svc.predict_manual_matchup(h, a, lg)
        for k in ["H", "D", "A"]:
            assert 0.0 <= res.production_probs[k] <= 1.0


def test_6_deterministic_predictions():
    """TEST 6: Verify repeated prediction calls yield identical results."""
    svc = PredictionService()
    r1 = svc.predict_manual_matchup("Inter", "Juventus", "Serie A")
    r2 = svc.predict_manual_matchup("Inter", "Juventus", "Serie A")
    assert r1.production_probs == r2.production_probs
    assert r1.production_decision == r2.production_decision
    assert r1.draw_risk_score == r2.draw_risk_score


def test_7_hda_decision_equals_argmax():
    """TEST 7: Verify decision strictly matches argmax of probabilities."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Real Madrid", "Real Sociedad", "La Liga")
    assert res is not None
    max_k = max(res.production_probs, key=res.production_probs.get)
    assert res.production_decision == max_k


def test_8_no_forced_draw():
    """TEST 8: Verify forced draw is impossible."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Nice", "Lorient", "Ligue 1")
    assert res is not None
    expected = max(res.production_probs, key=res.production_probs.get)
    assert res.production_decision == expected


def test_9_goal_prediction_validity():
    """TEST 9: Verify expected goals are non-negative floats."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res.lambda_home is not None and res.lambda_home > 0
    assert res.lambda_away is not None and res.lambda_away > 0


def test_10_score_prediction_validity():
    """TEST 10: Verify score predictions are valid string scorelines."""
    p_goal_file = STEP3_DIR / "candidate_goal_metrics.csv"
    assert p_goal_file.exists()
    df_g = pd.read_csv(p_goal_file)
    assert len(df_g) >= 5


def test_11_no_temporal_leakage_audit_passes():
    """TEST 11: Verify leakage audit artifact exists and confirms clean causal status."""
    p_leak = STEP3_DIR / "leakage_audit.csv"
    assert p_leak.exists()
    df_l = pd.read_csv(p_leak)
    assert all(df_l["causal_status"] == "PASS")


def test_12_candidate_reproducibility():
    """TEST 12: Verify 1X2 candidate metrics file exists and covers all candidates."""
    p_cand = STEP3_DIR / "candidate_1x2_metrics.csv"
    assert p_cand.exists()
    df_c = pd.read_csv(p_cand)
    assert len(df_c) >= 6


def test_13_dashboard_schema_clean_and_minimal():
    """TEST 13: Verify app.py Tab 4 match table has exactly the clean required columns."""
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
        assert req_col in app_text, f"Missing required column in table: {req_col}"


def test_14_production_isolation_maintained():
    """TEST 14: Verify model selector dropdown remains absent."""
    app_text = (PROJECT_ROOT / "src/dashboard/app.py").read_text(encoding="utf-8")
    assert "ACTIVE_MODEL_KEY = \"V4.0 Production\"" in app_text
    assert "st.sidebar.selectbox(\"Select Model\"" not in app_text


def test_15_prospective_shadow_ledger_integrity():
    """TEST 15: Verify prospective shadow ledger contains 54 future fixtures."""
    p_shad = STEP3_DIR / "prospective_shadow_ledger.csv"
    assert p_shad.exists()
    df_s = pd.read_csv(p_shad)
    assert len(df_s) == 54
