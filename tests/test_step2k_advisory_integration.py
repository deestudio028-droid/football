"""Comprehensive test suite for Step 2K — Draw Risk Advisory Shadow Integration."""
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
STEP2K_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2k_advisory_integration"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_1_v4_0_md5_remains_frozen():
    """TEST 1: Verify V4.0 production baseline MD5 hash is bit-identical."""
    assert V4_PATH.exists()
    act = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act}"


def test_2_v4_1_candidate_md5_remains_frozen():
    """TEST 2: Verify V4.1 candidate MD5 hash is bit-identical."""
    assert V4_1_PATH.exists()
    act = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act}"


def test_3_original_probabilities_unchanged():
    """TEST 3: Verify original V4.0 probabilities sum to 1 and match baseline."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res is not None
    probs = res.production_probs
    assert np.isclose(probs["H"] + probs["D"] + probs["A"], 1.0, atol=1e-2)
    assert res.benchmark_v4_0_probs == res.v4_champ_probs


def test_4_original_decision_unchanged():
    """TEST 4: Verify production decision strictly equals argmax of production probabilities."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res is not None
    max_k = max(res.production_probs, key=res.production_probs.get)
    assert res.production_decision == max_k


def test_5_risk_score_deterministic():
    """TEST 5: Verify risk score calculation is 100% deterministic."""
    advisor = DrawRiskAdvisor()
    r1 = advisor.evaluate_match_risk(cal_p_d=0.31, cal_win_diff=0.04, score_space_draw=0.29, lambda_gap=0.15)
    r2 = advisor.evaluate_match_risk(cal_p_d=0.31, cal_win_diff=0.04, score_space_draw=0.29, lambda_gap=0.15)
    assert r1["draw_risk_score"] == r2["draw_risk_score"]
    assert r1["draw_risk_tier"] == r2["draw_risk_tier"]


def test_6_risk_tier_deterministic():
    """TEST 6: Verify risk tier categorization boundaries."""
    advisor = DrawRiskAdvisor()
    assert advisor.classify_tier(0.25) == "LOW"
    assert advisor.classify_tier(0.45) == "MEDIUM"
    assert advisor.classify_tier(0.72) == "HIGH"
    assert advisor.classify_tier(0.88) == "CRITICAL"


def test_7_risk_score_in_unit_interval():
    """TEST 7: Verify risk score is strictly bounded in [0.0, 1.0]."""
    advisor = DrawRiskAdvisor()
    for pd_val in [0.10, 0.25, 0.35, 0.50]:
        for gap in [0.01, 0.10, 0.50]:
            r = advisor.evaluate_match_risk(cal_p_d=pd_val, cal_win_diff=gap, score_space_draw=0.25, lambda_gap=gap)
            assert 0.0 <= r["draw_risk_score"] <= 1.0


def test_8_only_valid_risk_tiers_returned():
    """TEST 8: Verify only LOW, MEDIUM, HIGH, CRITICAL tiers are returned."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Real Madrid", "Real Sociedad", "La Liga")
    assert res is not None
    assert res.draw_risk_tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert res.draw_risk_label in (
        "Low draw vulnerability",
        "Moderate draw vulnerability",
        "High draw vulnerability",
        "Critical draw vulnerability",
    )


def test_9_no_production_model_mutation():
    """TEST 9: Verify registry active model is V4.0 Production."""
    reg = ModelRegistry()
    assert reg.get_default_model() == "V4.0 Production"
    assert reg.get_production_model().display_name == "V4.0 Production"
    assert reg.get_production_model().model_id == "v4_0_draw_champion"


def test_10_no_research_model_exposed_in_dropdown():
    """TEST 10: Verify model dropdown remains completely absent from app.py."""
    app_text = (PROJECT_ROOT / "src/dashboard/app.py").read_text(encoding="utf-8")
    assert "ACTIVE_MODEL_KEY = \"V4.0 Production\"" in app_text
    assert "st.sidebar.selectbox" not in app_text or "Model" not in app_text


def test_11_v4_0_remains_the_only_production_model():
    """TEST 11: Verify prediction service uses V4.0 Production by default."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res.prediction_model == "V4.0 Production"


def test_12_existing_prediction_api_compatibility():
    """TEST 12: Verify backward compatibility with existing callers."""
    svc = PredictionService()
    res = svc.predict_manual_matchup("Liverpool", "Everton", "Premier League")
    assert hasattr(res, "production_probs")
    assert hasattr(res, "production_decision")
    assert hasattr(res, "benchmark_v4_0_probs")
    assert hasattr(res, "benchmark_v4_0_decision")
    assert hasattr(res, "draw_risk_score")
    assert hasattr(res, "draw_risk_tier")
    assert hasattr(res, "draw_risk_reasons")


def test_13_risk_explanations_deterministic():
    """TEST 13: Verify risk explanations mention contributing factors."""
    advisor = DrawRiskAdvisor()
    r = advisor.evaluate_match_risk(
        cal_p_d=0.32,
        cal_win_diff=0.03,
        score_space_draw=0.28,
        lambda_gap=0.10,
        tot_goals=2.20,
        elo_gap=25.0,
    )
    assert r["draw_risk_tier"] in ("HIGH", "CRITICAL")
    reasons_str = " ".join(r["draw_risk_reasons"])
    assert "draw probability" in reasons_str.lower()
    assert "margin" in reasons_str.lower() or "parity" in reasons_str.lower()


def test_14_risk_advisory_cannot_force_draw():
    """TEST 14: Hard invariant check - advisory MUST NEVER alter production decision to D."""
    svc = PredictionService()
    # A fixture with high draw risk
    res = svc.predict_manual_matchup("Fulham", "Chelsea", "Premier League")
    assert res is not None
    # Check that decision remains H or A (argmax of probs) and is NOT mutated to D by the advisory
    expected_dec = max(res.production_probs, key=res.production_probs.get)
    assert res.production_decision == expected_dec


def test_15_repeated_execution_produces_identical_output():
    """TEST 15: Verify repeated calls yield identical results."""
    svc = PredictionService()
    r1 = svc.predict_manual_matchup("Inter", "Monza", "Serie A")
    r2 = svc.predict_manual_matchup("Inter", "Monza", "Serie A")
    assert r1.production_probs == r2.production_probs
    assert r1.draw_risk_score == r2.draw_risk_score
    assert r1.draw_risk_tier == r2.draw_risk_tier
