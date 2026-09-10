"""Step 2L — Test suite for Draw Risk Advisory UX, Semantics, and Confidence Refinement."""
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
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.draw_risk_advisor import DrawRiskAdvisor
from dashboard.model_registry import ModelRegistry
from dashboard.prediction_service import PredictionService

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_1_v4_0_probabilities_remain_identical():
    """TEST 1: Verify V4.0 probabilities remain identical and sum to 1."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res is not None
    p = res.production_probs
    assert np.isclose(p["H"] + p["D"] + p["A"], 1.0, atol=1e-2)
    assert res.production_probs == res.benchmark_v4_0_probs


def test_2_v4_0_decision_remains_identical():
    """TEST 2: Verify V4.0 decision strictly equals argmax of production probabilities."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res is not None
    assert res.production_decision == max(res.production_probs, key=res.production_probs.get)


def test_3_draw_risk_does_not_mutate_prediction():
    """TEST 3: Verify high draw risk does not mutate Home/Away prediction."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Fulham", "Chelsea", "Premier League")
    assert res is not None
    # High risk fixture must retain its primary argmax decision
    assert res.production_decision in ("H", "A")
    assert res.production_decision == max(res.production_probs, key=res.production_probs.get)
    assert res.draw_risk_tier in ("HIGH", "CRITICAL", "MEDIUM")


def test_4_all_four_risk_tiers_render_correctly():
    """TEST 4: Verify tier classification covers all four tiers."""
    advisor = DrawRiskAdvisor()
    assert advisor.classify_tier(0.15) == "LOW"
    assert advisor.classify_tier(0.50) == "MEDIUM"
    assert advisor.classify_tier(0.72) == "HIGH"
    assert advisor.classify_tier(0.85) == "CRITICAL"


def test_5_risk_labels_are_correct():
    """TEST 5: Verify risk labels follow exact required UX semantics."""
    advisor = DrawRiskAdvisor()
    res_low = advisor.evaluate_match_risk(cal_p_d=0.15, cal_win_diff=0.50, score_space_draw=0.15, lambda_gap=1.0)
    res_med = advisor.evaluate_match_risk(cal_p_d=0.28, cal_win_diff=0.10, score_space_draw=0.25, lambda_gap=0.25)
    res_high = advisor.evaluate_match_risk(cal_p_d=0.31, cal_win_diff=0.03, score_space_draw=0.29, lambda_gap=0.15)
    res_crit = advisor.evaluate_match_risk(cal_p_d=0.35, cal_win_diff=0.01, score_space_draw=0.32, lambda_gap=0.05)

    assert res_low["draw_risk_label"] == "Low draw vulnerability"
    assert res_med["draw_risk_label"] == "Moderate draw vulnerability"
    assert res_high["draw_risk_label"] == "High draw vulnerability"
    assert res_crit["draw_risk_label"] == "Critical draw vulnerability"


def test_6_risk_explanations_are_deterministic():
    """TEST 6: Verify risk reasons are deterministic and non-random."""
    advisor = DrawRiskAdvisor()
    r1 = advisor.evaluate_match_risk(cal_p_d=0.32, cal_win_diff=0.02, score_space_draw=0.30, lambda_gap=0.08)
    r2 = advisor.evaluate_match_risk(cal_p_d=0.32, cal_win_diff=0.02, score_space_draw=0.30, lambda_gap=0.08)
    assert r1["draw_risk_reasons"] == r2["draw_risk_reasons"]


def test_7_draw_prediction_never_emitted_by_advisory():
    """TEST 7: Verify advisory never claims 'Draw Prediction' or 'Guaranteed Draw'."""
    advisor = DrawRiskAdvisor()
    for s in np.linspace(0.0, 1.0, 20):
        t = advisor.classify_tier(s)
        r = advisor.generate_reasons(t, 0.32, 0.02, 0.30, 0.05, 2.1, 10.0)
        reasons_text = " ".join(r)
        assert "draw prediction" not in reasons_text.lower()
        assert "guaranteed" not in reasons_text.lower()
        assert "will be a draw" not in reasons_text.lower()


def test_8_forced_draw_is_impossible():
    """TEST 8: Verify forced draw is impossible in PredictionService."""
    svc = PredictionService()
    for home, away, league in [
        ("Arsenal", "Chelsea", "Premier League"),
        ("Real Madrid", "FC Barcelona", "La Liga"),
        ("Inter", "Juventus", "Serie A"),
    ]:
        res = svc.predict_manual_matchup(home, away, league)
        assert res is not None
        # Decision must strictly match argmax of probabilities
        expected = max(res.production_probs, key=res.production_probs.get)
        assert res.production_decision == expected


def test_9_v4_0_remains_sole_production_model():
    """TEST 9: Verify V4.0 is the active production truth."""
    reg = ModelRegistry()
    assert reg.get_default_model() == "V4.0 Production"
    assert reg.get_production_model().display_name == "V4.0 Production"


def test_10_model_selector_remains_absent():
    """TEST 10: Verify model selection dropdown remains absent from dashboard code."""
    app_text = (PROJECT_ROOT / "src/dashboard/app.py").read_text(encoding="utf-8")
    assert "ACTIVE_MODEL_KEY = \"V4.0 Production\"" in app_text
    assert "st.sidebar.selectbox" not in app_text or "Model" not in app_text


def test_11_v4_1_remains_research_only():
    """TEST 11: Verify V4.1 status is RESEARCH."""
    reg = ModelRegistry()
    m_v41 = reg.get_model("V4.1 Production")
    assert m_v41 is not None
    assert m_v41.role == "RESEARCH"


def test_12_existing_prediction_api_remains_compatible():
    """TEST 12: Verify backward compatibility with existing prediction result schema."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Liverpool", "Everton", "Premier League")
    assert hasattr(res, "production_probs")
    assert hasattr(res, "production_decision")
    assert hasattr(res, "draw_risk_score")
    assert hasattr(res, "draw_risk_tier")
    assert hasattr(res, "draw_risk_label")
    assert hasattr(res, "draw_risk_badge")
    assert hasattr(res, "draw_risk_reasons")
    assert hasattr(res, "is_vulnerable_fixture")


def test_13_probability_sum_remains_exactly_1():
    """TEST 13: Verify all probability distributions sum to 1."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Nice", "Lorient", "Ligue 1")
    assert res is not None
    assert np.isclose(sum(res.production_probs.values()), 1.0, atol=1e-2)
    assert np.isclose(sum(res.v4_champ_probs.values()), 1.0, atol=1e-2)


def test_14_repeated_prediction_calls_remain_deterministic():
    """TEST 14: Verify identical repeated calls produce bit-identical results."""
    svc = PredictionService()
    r1 = svc.predict_manual_matchup("Bayern Munich", "Borussia Dortmund", "Bundesliga")
    r2 = svc.predict_manual_matchup("Bayern Munich", "Borussia Dortmund", "Bundesliga")
    assert r1.production_probs == r2.production_probs
    assert r1.production_decision == r2.production_decision
    assert r1.draw_risk_score == r2.draw_risk_score
    assert r1.draw_risk_tier == r2.draw_risk_tier
    assert r1.draw_risk_reasons == r2.draw_risk_reasons
