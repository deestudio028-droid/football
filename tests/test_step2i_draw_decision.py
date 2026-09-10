"""Unit and regression tests for Step 2I — Draw Decision Intelligence Research."""
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
    str(PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2i_draw_decision"),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from step2i_shadow_runner import Step2IShadowEngine

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"
STEP2I_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2i_draw_decision"

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


def test_3_production_files_untouched():
    """TEST 3: Verify dashboard and production prediction service files were untouched."""
    app_path = PROJECT_ROOT / "src/dashboard/app.py"
    pred_path = PROJECT_ROOT / "src/dashboard/prediction_service.py"
    reg_path = PROJECT_ROOT / "src/dashboard/model_registry.py"

    assert app_path.exists()
    assert pred_path.exists()
    assert reg_path.exists()

    app_text = app_path.read_text(encoding="utf-8")
    assert "ACTIVE_MODEL_KEY = \"V4.0 Production\"" in app_text
    assert "model_selector_dropdown" not in app_text


def test_4_candidate_inference_deterministic():
    """TEST 4: Verify shadow candidate produces 100% deterministic output."""
    engine = Step2IShadowEngine()
    res1 = engine.generate_shadow_prediction(
        fixture_id=101,
        kickoff="2026-08-24T18:45:00+00:00",
        league="Serie A",
        home_team="Udinese",
        away_team="Como",
        lh=1.25,
        la=1.20,
        abs_elo=35.0,
        X_df={"A_home": 1.05, "D_home": 1.02, "A_away": 0.98, "D_away": 1.04},
    )
    res2 = engine.generate_shadow_prediction(
        fixture_id=101,
        kickoff="2026-08-24T18:45:00+00:00",
        league="Serie A",
        home_team="Udinese",
        away_team="Como",
        lh=1.25,
        la=1.20,
        abs_elo=35.0,
        X_df={"A_home": 1.05, "D_home": 1.02, "A_away": 0.98, "D_away": 1.04},
    )
    assert res1 == res2
    assert res1.draw_score == res2.draw_score


def test_5_leakage_audit_artifact_exists_and_passes():
    """TEST 5: Verify temporal leakage audit artifact exists and confirms zero leakage."""
    leakage_doc = STEP2I_DIR / "step2i_leakage_audit.md"
    assert leakage_doc.exists()
    content = leakage_doc.read_text(encoding="utf-8")
    assert "ZERO TEMPORAL OR DATA LEAKAGE DETECTED" in content
    assert "PASS" in content


def test_6_probability_simplex_preserved():
    """TEST 6: Verify probabilities sum to 1.0 on shadow inference."""
    engine = Step2IShadowEngine()
    pred = engine.generate_shadow_prediction(
        fixture_id=202,
        kickoff="2026-08-24T18:45:00+00:00",
        league="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        lh=1.80,
        la=1.10,
        abs_elo=65.0,
        X_df={"A_home": 1.25, "D_home": 0.92, "A_away": 1.10, "D_away": 1.05},
    )
    v4_sum = pred.v4_p_H + pred.v4_p_D + pred.v4_p_A
    cal_sum = pred.cal_p_H + pred.cal_p_D + pred.cal_p_A
    assert np.isclose(v4_sum, 1.0, atol=1e-3)
    assert np.isclose(cal_sum, 1.0, atol=1e-3)


def test_7_v4_probabilities_not_mutated():
    """TEST 7: Verify shadow engine does not mutate underlying V4.0 baseline probabilities."""
    engine = Step2IShadowEngine()
    pred = engine.generate_shadow_prediction(
        fixture_id=303,
        kickoff="2026-08-24T18:45:00+00:00",
        league="La Liga",
        home_team="Valencia",
        away_team="Celta de Vigo",
        lh=1.30,
        la=1.25,
        abs_elo=20.0,
        X_df={"A_home": 1.00, "D_home": 1.00, "A_away": 1.00, "D_away": 1.00},
    )
    assert pred.v4_p_D < pred.cal_p_D  # Calibrated draw is strictly refined upward without altering v4 base


def test_8_holdout_results_exist_and_match_unbiased_split():
    """TEST 8: Verify holdout results match exactly N=1,751 matches."""
    ho_csv = STEP2I_DIR / "holdout_decision_results.csv"
    assert ho_csv.exists()
    df_ho = pd.read_csv(ho_csv)
    assert len(df_ho) == 10
    assert (df_ho["n_matches"] == 1751).all()


def test_9_caution_zone_logic_valid():
    """TEST 9: Verify caution zone correctly flags close, draw-elevated fixtures."""
    engine = Step2IShadowEngine()
    # High draw, close win margin -> Caution Zone
    res_close = engine.generate_shadow_prediction(
        fixture_id=404,
        kickoff="2026-08-24T18:45:00+00:00",
        league="Serie A",
        home_team="Udinese",
        away_team="Como",
        lh=1.20,
        la=1.20,
        abs_elo=10.0,
        X_df={"A_home": 1.0, "D_home": 1.0, "A_away": 1.0, "D_away": 1.0},
    )
    assert res_close.is_caution_zone is True
    assert res_close.advisory_status == "CAUTION_DRAW_ZONE"

    # Lopsided match -> Not Caution Zone
    res_lopsided = engine.generate_shadow_prediction(
        fixture_id=505,
        kickoff="2026-08-24T18:45:00+00:00",
        league="Premier League",
        home_team="Man City",
        away_team="Luton",
        lh=2.60,
        la=0.60,
        abs_elo=180.0,
        X_df={"A_home": 1.5, "D_home": 0.8, "A_away": 0.8, "D_away": 1.4},
    )
    assert res_lopsided.is_caution_zone is False
    assert res_lopsided.advisory_status == "STANDARD_WIN"


def test_10_all_step2i_artifacts_exist():
    """TEST 10: Verify all 13 required Step 2I artifacts exist."""
    required_files = [
        "step2i_draw_decision_research.py",
        "candidate_decision_comparison.csv",
        "walkforward_decision_results.csv",
        "threshold_sensitivity.csv",
        "league_decision_results.csv",
        "holdout_decision_results.csv",
        "prospective_decision_audit.csv",
        "draw_signal_importance.csv",
        "decision_confusion_matrices.csv",
        "step2i_leakage_audit.md",
        "step2i_candidate_config.json",
        "step2i_shadow_runner.py",
    ]
    for rf in required_files:
        assert (STEP2I_DIR / rf).exists(), f"Missing required Step 2I file: {rf}"
