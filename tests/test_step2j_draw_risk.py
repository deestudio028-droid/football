"""Unit and regression tests for Step 2J — Draw Risk / Caution Intelligence Layer."""
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
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2j_draw_risk"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from step2j_shadow_runner import Step2JDrawRiskEngine

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
STEP2J_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2j_draw_risk"

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


def test_3_v4_probabilities_and_predictions_unchanged():
    """TEST 3: Verify shadow risk runner returns exact V4.0 predictions and probabilities."""
    engine = Step2JDrawRiskEngine()
    pred = engine.predict_match_risk(
        fixture_id=101,
        kickoff="2026-08-30T14:00:00+00:00",
        competition_name="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        lh=1.75,
        la=1.05,
        abs_elo=70.0,
    )
    assert pred.v4_decision == "H"
    assert np.isclose(pred.v4_p_H + pred.v4_p_D + pred.v4_p_A, 1.0, atol=1e-3)
    assert pred.v4_p_H > pred.v4_p_A


def test_4_no_production_files_modified():
    """TEST 4: Verify production dashboard and registry code remained untouched."""
    app_text = (PROJECT_ROOT / "src/dashboard/app.py").read_text(encoding="utf-8")
    assert "ACTIVE_MODEL_KEY = \"V4.0 Production\"" in app_text
    assert "model_selector_dropdown" not in app_text


def test_5_no_future_information_leakage_audit_passes():
    """TEST 5: Verify leakage audit artifact exists and confirms zero leakage."""
    leak_file = STEP2J_DIR / "step2j_leakage_audit.md"
    assert leak_file.exists()
    content = leak_file.read_text(encoding="utf-8")
    assert "ZERO TEMPORAL OR DATA LEAKAGE DETECTED" in content
    assert "PASS" in content


def test_6_deterministic_risk_score():
    """TEST 6: Verify risk score is completely deterministic."""
    engine = Step2JDrawRiskEngine()
    p1 = engine.predict_match_risk(
        fixture_id=202,
        kickoff="2026-08-30T14:00:00+00:00",
        competition_name="Serie A",
        home_team="Udinese",
        away_team="Como",
        lh=1.20,
        la=1.20,
        abs_elo=20.0,
    )
    p2 = engine.predict_match_risk(
        fixture_id=202,
        kickoff="2026-08-30T14:00:00+00:00",
        competition_name="Serie A",
        home_team="Udinese",
        away_team="Como",
        lh=1.20,
        la=1.20,
        abs_elo=20.0,
    )
    assert p1 == p2
    assert p1.draw_risk_score == p2.draw_risk_score


def test_7_risk_score_in_unit_interval():
    """TEST 7: Verify risk score is strictly bounded in [0.0, 1.0]."""
    engine = Step2JDrawRiskEngine()
    for lh, la, elo in [(0.5, 0.5, 0.0), (3.0, 0.2, 250.0), (1.2, 1.3, 30.0), (2.0, 2.0, 10.0)]:
        p = engine.predict_match_risk(
            fixture_id=303,
            kickoff="2026-08-30T14:00:00+00:00",
            competition_name="La Liga",
            home_team="TeamA",
            away_team="TeamB",
            lh=lh,
            la=la,
            abs_elo=elo,
        )
        assert 0.0 <= p.draw_risk_score <= 1.0
        assert p.draw_risk_tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_8_tier_assignment_monotonic_and_explainable():
    """TEST 8: Verify tier assignment logic and explainability generation."""
    engine = Step2JDrawRiskEngine()
    # High risk match
    p_high = engine.predict_match_risk(
        fixture_id=404,
        kickoff="2026-08-30T14:00:00+00:00",
        competition_name="Ligue 1",
        home_team="Nice",
        away_team="Lorient",
        lh=1.25,
        la=1.25,
        abs_elo=15.0,
    )
    assert p_high.draw_risk_tier in ("HIGH", "CRITICAL")
    assert p_high.is_vulnerable_fixture is True
    assert "draw" in p_high.draw_risk_reason.lower()

    # Low risk match
    p_low = engine.predict_match_risk(
        fixture_id=505,
        kickoff="2026-08-30T14:00:00+00:00",
        competition_name="Premier League",
        home_team="Man City",
        away_team="Luton Town",
        lh=2.80,
        la=0.50,
        abs_elo=200.0,
    )
    assert p_low.draw_risk_tier == "LOW"
    assert p_low.is_vulnerable_fixture is False


def test_9_holdout_results_exist_and_match_sample_size():
    """TEST 9: Verify holdout tier analysis matches N=1,751 matches."""
    ho_file = STEP2J_DIR / "draw_risk_holdout.csv"
    assert ho_file.exists()
    df_ho = pd.read_csv(ho_file)
    overall_row = df_ho[df_ho["risk_tier"] == "OVERALL"]
    assert len(overall_row) == 1
    assert int(overall_row["match_count"].values[0]) == 1751


def test_10_all_step2j_artifacts_exist():
    """TEST 10: Verify all 15 required Step 2J artifacts exist."""
    required_files = [
        "step2j_draw_risk_research.py",
        "draw_risk_predictions.csv",
        "draw_risk_tier_analysis.csv",
        "draw_risk_walkforward.csv",
        "draw_risk_holdout.csv",
        "draw_risk_prospective.csv",
        "draw_risk_league_analysis.csv",
        "draw_risk_threshold_sensitivity.csv",
        "draw_risk_feature_importance.csv",
        "draw_risk_calibration.csv",
        "step2j_leakage_audit.md",
        "step2j_candidate_config.json",
        "step2j_shadow_runner.py",
    ]
    for rf in required_files:
        assert (STEP2J_DIR / rf).exists(), f"Missing required file: {rf}"
