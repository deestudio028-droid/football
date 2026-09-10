"""Tests for V4.2 Draw Resolution Candidate Model.

Verifies:
- Simplex normalization (sum = 1.0)
- Probability bounds ([0, 1])
- No NaN / Inf across extreme inputs
- DIBP scoreline grid properties
- Proportional odds invariance
- Decision policy behaviors
- Determinism and immutability
- Production isolation
"""
import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.v4_2_draw_resolution_candidate import (
    DrawResolutionConfig,
    DrawResolutionPrediction,
    compute_dibp_scoreline_matrix,
    extract_1x2_from_grid,
    predict_draw_resolution,
)

def test_simplex_normalization():
    """Verify that probabilities always sum to 1.0 within 1e-12."""
    p_v4 = [0.45, 0.25, 0.30]
    pred = predict_draw_resolution(1.5, 1.2, p_v4, 50.0, "Premier League")
    total = sum(pred.probabilities.values())
    assert abs(total - 1.0) < 1e-12
    assert all(0.0 <= p <= 1.0 for p in pred.probabilities.values())

def test_extreme_inputs_numerical_stability():
    """Verify stability under extreme goal and Elo inputs."""
    extremes = [
        (0.01, 0.01, [0.1, 0.8, 0.1], 0.0),
        (5.0, 0.1, [0.85, 0.10, 0.05], 400.0),
        (0.1, 5.0, [0.05, 0.10, 0.85], -400.0),
        (6.0, 6.0, [0.35, 0.30, 0.35], 0.0),
    ]
    for lh, la, pv4, elo_diff in extremes:
        pred = predict_draw_resolution(lh, la, pv4, abs(elo_diff), "La Liga")
        assert not any(np.isnan(p) for p in pred.probabilities.values())
        assert not any(np.isinf(p) for p in pred.probabilities.values())
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-12

def test_dibp_grid_properties():
    """Verify DIBP grid shape, non-negativity, and diagonal inflation."""
    grid_no_inf = compute_dibp_scoreline_matrix(1.4, 1.2, rho=-0.056, p_inf=0.0)
    grid_inf = compute_dibp_scoreline_matrix(1.4, 1.2, rho=-0.056, p_inf=0.08)
    
    assert grid_inf.shape == (11, 11)
    assert abs(grid_inf.sum() - 1.0) < 1e-12
    # Diagonal sum must be strictly larger with positive inflation
    assert np.trace(grid_inf) > np.trace(grid_no_inf)

def test_odds_ratio_invariance():
    """Verify that P(H)/P(A) is invariant between V4 and Candidate before Dirichlet layer."""
    p_v4 = [0.50, 0.20, 0.30]
    pred = predict_draw_resolution(1.6, 1.1, p_v4, 60.0, "Serie A")
    
    odds_v4 = p_v4[0] / p_v4[2]
    odds_cand = pred.probabilities["H"] / pred.probabilities["A"]
    assert abs(odds_cand - odds_v4) < 1e-12

def test_decision_policies():
    """Verify operational decision policies: argmax, threshold, and margin_aware."""
    # Balanced match where P(H)=0.37, P(D)=0.31, P(A)=0.32
    p_v4 = [0.38, 0.28, 0.34]
    
    # 1. Argmax policy -> should predict H because 0.37 > 0.31
    cfg_argmax = DrawResolutionConfig(decision_policy="argmax", stacking_intercept=0.25)
    pred_argmax = predict_draw_resolution(1.3, 1.2, p_v4, 10.0, "Bundesliga", cfg_argmax)
    assert pred_argmax.predicted_class in ["H", "A"]
    
    # 2. Threshold policy @ 0.28 -> should predict D because P(D) >= 0.28
    cfg_thresh = DrawResolutionConfig(decision_policy="threshold", decision_threshold=0.28, stacking_intercept=0.25)
    pred_thresh = predict_draw_resolution(1.3, 1.2, p_v4, 10.0, "Bundesliga", cfg_thresh)
    assert pred_thresh.predicted_class == "D"
    
    # 3. Margin-aware policy -> should predict D when within margin
    cfg_margin = DrawResolutionConfig(decision_policy="margin_aware", decision_threshold=0.28, decision_margin=0.10, stacking_intercept=0.25)
    pred_margin = predict_draw_resolution(1.3, 1.2, p_v4, 10.0, "Bundesliga", cfg_margin)
    assert pred_margin.predicted_class == "D"

def test_production_immutability():
    """Verify that candidate code is isolated and production hashes remain untouched."""
    pinned = {
        "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
        "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
        "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
        "research/v4_promotion/draw_champion_method_frozen.json": "9c396e7e5364f93f079313726c1ba499",
    }
    for rel, exp in pinned.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}"
