"""Unit tests for DrawProbabilityCalibrator component and boundary behavior."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP2E_DIR = PROJECT_ROOT / "research/v5_model_improvement/step2_draw_research/step2e_shadow_integration"
sys.path.insert(0, str(STEP2E_DIR))

from draw_probability_calibrator import DrawProbabilityCalibrator, CalibratedPredictionResult

CONFIG_PATH = STEP2E_DIR / "draw_calibrator_config.json"


@pytest.fixture
def calibrator() -> DrawProbabilityCalibrator:
    return DrawProbabilityCalibrator.from_config_file(CONFIG_PATH)


def test_calibrator_config_loading(calibrator: DrawProbabilityCalibrator):
    assert calibrator.slope_a > 0.0
    assert abs(calibrator.slope_a - 0.942103) < 1e-4
    assert abs(calibrator.intercept_b - 0.128362) < 1e-4
    assert calibrator.version == "step2e-shadow-1"


def test_standard_calibration_invariants(calibrator: DrawProbabilityCalibrator):
    """Test normal probability distribution."""
    res = calibrator.calibrate_single(0.45, 0.25, 0.30)
    assert 0.0 <= res.calibrated_p_home <= 1.0
    assert 0.0 <= res.calibrated_p_draw <= 1.0
    assert 0.0 <= res.calibrated_p_away <= 1.0

    # Sum equals 1.0
    prob_sum = res.calibrated_p_home + res.calibrated_p_draw + res.calibrated_p_away
    assert abs(prob_sum - 1.0) < 1e-6

    # Home/Away ratio preserved
    orig_ratio = 0.45 / 0.30
    cal_ratio = res.calibrated_p_home / res.calibrated_p_away
    assert abs(orig_ratio - cal_ratio) < 1e-5


def test_boundary_draw_near_zero(calibrator: DrawProbabilityCalibrator):
    """Test boundary condition where P(D) -> 0."""
    res = calibrator.calibrate_single(0.70, 0.01, 0.29)
    assert res.calibrated_p_draw >= 0.0
    assert not np.isnan(res.calibrated_p_draw)
    assert not np.isinf(res.calibrated_p_draw)
    prob_sum = res.calibrated_p_home + res.calibrated_p_draw + res.calibrated_p_away
    assert abs(prob_sum - 1.0) < 1e-6


def test_boundary_draw_near_one(calibrator: DrawProbabilityCalibrator):
    """Test boundary condition where P(D) -> 1."""
    res = calibrator.calibrate_single(0.01, 0.98, 0.01)
    assert res.calibrated_p_draw <= 1.0
    assert not np.isnan(res.calibrated_p_draw)
    assert not np.isinf(res.calibrated_p_draw)
    prob_sum = res.calibrated_p_home + res.calibrated_p_draw + res.calibrated_p_away
    assert abs(prob_sum - 1.0) < 1e-6


def test_boundary_home_away_equality(calibrator: DrawProbabilityCalibrator):
    """Test condition where P(H) == P(A)."""
    res = calibrator.calibrate_single(0.35, 0.30, 0.35)
    assert abs(res.calibrated_p_home - res.calibrated_p_away) < 1e-6
    prob_sum = res.calibrated_p_home + res.calibrated_p_draw + res.calibrated_p_away
    assert abs(prob_sum - 1.0) < 1e-6


def test_boundary_extreme_home_dominance(calibrator: DrawProbabilityCalibrator):
    """Test condition where P(H) >> P(A)."""
    res = calibrator.calibrate_single(0.90, 0.08, 0.02)
    assert res.calibrated_p_home > res.calibrated_p_away
    orig_ratio = 0.90 / 0.02
    cal_ratio = res.calibrated_p_home / res.calibrated_p_away
    assert abs(orig_ratio - cal_ratio) < 1e-4


def test_determinism_and_immutability(calibrator: DrawProbabilityCalibrator):
    """Verify that same inputs always produce identical results and no input mutation occurs."""
    input_tuple = (0.40, 0.28, 0.32)
    res1 = calibrator.calibrate_single(*input_tuple)
    res2 = calibrator.calibrate_single(*input_tuple)

    assert res1.calibrated_p_draw == res2.calibrated_p_draw
    assert res1.calibrated_p_home == res2.calibrated_p_home
    assert res1.calibrated_p_away == res2.calibrated_p_away


def test_batch_array_calibration(calibrator: DrawProbabilityCalibrator):
    """Verify that batch array calibration matches single item calibration."""
    arr = np.array([
        [0.45, 0.25, 0.30],
        [0.35, 0.30, 0.35],
        [0.60, 0.20, 0.20],
    ])
    cal_arr = calibrator.calibrate_array(arr)
    assert cal_arr.shape == (3, 3)

    for i in range(3):
        res_single = calibrator.calibrate_single(arr[i, 0], arr[i, 1], arr[i, 2])
        assert abs(cal_arr[i, 1] - res_single.calibrated_p_draw) < 1e-5
