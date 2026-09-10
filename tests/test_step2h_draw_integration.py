"""Unit tests for Step 2H — V4.0 Draw-Enhanced Production Pipeline Integration."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP2E_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"
STEP2H_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2h_production_integration"

sys_paths = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"),
    str(STEP2E_DIR),
]
for p in sys_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.model_registry import get_model_registry
from dashboard.prediction_service import PredictionService
from draw_probability_calibrator import DrawProbabilityCalibrator

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


@pytest.fixture(scope="module")
def prediction_service():
    return PredictionService()


def test_1_v4_md5_remains_frozen():
    """TEST 1: V4.0 model artifact MD5 remains 06841f0c03c8597b2b8cd8f8ab064864."""
    assert V4_PATH.exists()
    act = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_MD5, f"V4.0 MD5 violation: expected {EXP_V4_MD5}, got {act}"


def test_2_v4_1_md5_remains_frozen():
    """TEST 2: V4.1 model artifact MD5 remains 145f918d933eb343c0f63ca342b10289."""
    assert V4_1_PATH.exists()
    act = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_1_MD5, f"V4.1 MD5 violation: expected {EXP_V4_1_MD5}, got {act}"


def test_3_v4_production_predictions_unchanged(prediction_service):
    """TEST 3: V4.0 Production predictions remain unchanged."""
    res_v40 = prediction_service.predict_manual_matchup(
        "Arsenal", "Chelsea", "Premier League", model_key="V4.0 Production"
    )
    assert res_v40 is not None
    assert res_v40.production_probs == res_v40.benchmark_v4_0_probs
    assert res_v40.production_decision == res_v40.benchmark_v4_0_decision


def test_4_probabilities_sum_to_one(prediction_service):
    """TEST 4: Draw-Enhanced probabilities sum to approximately 1: H' + D' + A' == 1."""
    res_enh = prediction_service.predict_manual_matchup(
        "Arsenal", "Chelsea", "Premier League", model_key="V4.0 Draw-Enhanced"
    )
    assert res_enh is not None
    p = res_enh.production_probs
    total = p["H"] + p["D"] + p["A"]
    assert abs(total - 1.0) < 1e-2


def test_5_probabilities_in_unit_interval(prediction_service):
    """TEST 5: All probabilities remain within [0, 1]."""
    res_enh = prediction_service.predict_manual_matchup(
        "Arsenal", "Chelsea", "Premier League", model_key="V4.0 Draw-Enhanced"
    )
    assert res_enh is not None
    for k, v in res_enh.production_probs.items():
        assert 0.0 <= v <= 1.0


def test_6_home_away_ratio_preservation(prediction_service):
    """TEST 6: Home/Away ratio preservation: P'(H)/P'(A) == P(H)/P(A)."""
    res_enh = prediction_service.predict_manual_matchup(
        "Fulham", "Chelsea", "Premier League", model_key="V4.0 Draw-Enhanced"
    )
    assert res_enh is not None
    orig_h = res_enh.benchmark_v4_0_probs["H"]
    orig_a = res_enh.benchmark_v4_0_probs["A"]
    enh_h = res_enh.production_probs["H"]
    enh_a = res_enh.production_probs["A"]

    orig_ratio = orig_h / orig_a
    enh_ratio = enh_h / enh_a
    assert abs(orig_ratio - enh_ratio) < 1e-2


def test_7_v4_input_arrays_not_mutated(prediction_service):
    """TEST 7: V4.0 input probability arrays are not mutated."""
    cal = prediction_service._draw_calibrator
    arr = np.array([[0.45, 0.25, 0.30]])
    arr_copy = arr.copy()
    _ = cal.calibrate_array(arr)
    assert np.array_equal(arr, arr_copy)


def test_8_repeated_inference_deterministic(prediction_service):
    """TEST 8: Repeated inference is deterministic."""
    res1 = prediction_service.predict_manual_matchup("Nice", "Lorient", "Ligue 1", model_key="V4.0 Draw-Enhanced")
    res2 = prediction_service.predict_manual_matchup("Nice", "Lorient", "Ligue 1", model_key="V4.0 Draw-Enhanced")
    assert res1.production_probs == res2.production_probs
    assert res1.production_decision == res2.production_decision


def test_9_same_underlying_v4_model_output(prediction_service):
    """TEST 9: V4.0 and V4.0 Draw-Enhanced use the same underlying V4.0 model output."""
    res_v40 = prediction_service.predict_manual_matchup("Nice", "Lorient", "Ligue 1", model_key="V4.0 Production")
    res_enh = prediction_service.predict_manual_matchup("Nice", "Lorient", "Ligue 1", model_key="V4.0 Draw-Enhanced")

    # Benchmark V4.0 probabilities and decisions must be bit-identical
    assert res_v40.benchmark_v4_0_probs == res_enh.benchmark_v4_0_probs
    assert res_v40.benchmark_v4_0_decision == res_enh.benchmark_v4_0_decision
    assert res_v40.v4_draw_enhanced_probs == res_enh.production_probs
    assert res_v40.v4_draw_enhanced_decision == res_enh.production_decision


def test_10_no_hard_draw_threshold_active(prediction_service):
    """TEST 10: No Physical Draw Gate / hard Draw threshold is active."""
    # Decision must be pure argmax over calibrated probabilities
    res_enh = prediction_service.predict_manual_matchup("Troyes", "Paris", "Ligue 1", model_key="V4.0 Draw-Enhanced")
    p = res_enh.production_probs
    expected_dec = max(p, key=p.get)
    assert res_enh.production_decision == expected_dec
