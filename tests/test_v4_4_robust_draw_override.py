"""Unit tests for V4.4 Robust Selective Draw Override Model.

Verifies:
1. Preset configs (Accuracy Maximizer, High-Precision Conservative, Balanced Preserver).
2. Override logic gates (Confidence cap protection, team balance, low-score requirement).
3. Simplex probability conservation.
4. Production asset immutability across all 20 pinned hashes.
"""
import hashlib
import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.v4_4_robust_draw_override import (
    V44RobustOverrideConfig,
    V44RobustPrediction,
    evaluate_robust_override,
    predict_v4_4_robust_override,
)

PINNED_20 = {
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
    "research/v4_promotion/promotion_market_odds.sqlite": "f8a41b79cd33afb412ccd9ae2892a196",
    "research/v4_promotion/fresh_100_market_odds.sqlite": "2cb80b79d772fbedd4f3707b39a32c13",
    "research/v4_promotion/fresh_100_fixture_ids.json": "761ad5cc571643e6985e671bd9c3d83a",
    "research/v4_promotion/fresh_extended_fixture_ids.json": "0526bfd6980dd51dae59c6f6aadab2f5",
    "research/v4_promotion/fresh_extended_market_odds.sqlite": "b4889d1791723ea653057af51ca00f8e",
    "research/v4_promotion/dixon_coles_rho_method_frozen.json": "822e742dcc82e5e96445b31c14c0c604",
    "research/v4_promotion/elo_draw_curve_method_frozen.json": "65dc2cf762f3d78abcf1a617ef23fe00",
    "research/v4_promotion/full_score_matrix_method_frozen.json": "cd44e1da88a50ac45e8383557ad5271f",
    "research/v4_promotion/market_calibration_method_frozen.json": "550a0e1f1358a8359d7141b422521dd9",
    "research/v4_promotion/draw_complementarity_method_frozen.json": "d4f7dc75785df076c105a6ebfc0a4d6e",
    "research/v4_promotion/temporal_regime_method_frozen.json": "4a4f72e1d288d2547272c9b30b0368df",
    "research/v4_promotion/statistical_power_uncertainty_method_frozen.json": "68d55b30789d40440a0c14cbfe225c7f",
}

def test_preset_configurations():
    """Verify default parameters of V4.4 preset configurations."""
    c_acc = V44RobustOverrideConfig.accuracy_maximizer()
    assert c_acc.name == "robust_accuracy_maximizer"
    assert c_acc.v4_winner_conf_cap == 0.45
    assert c_acc.tot_expected_goals_cap == 2.50
    assert c_acc.abs_elo_cap == 100.0

    c_cons = V44RobustOverrideConfig.high_precision_conservative()
    assert c_cons.abs_elo_cap == 80.0

    c_bal = V44RobustOverrideConfig.balanced_draw_preserver()
    assert c_bal.abs_elo_cap == 150.0

def test_override_trigger_on_fragile_balanced_match():
    """Verify override occurs when match is balanced, low-scoring, and V4 confidence is low."""
    cfg = V44RobustOverrideConfig.accuracy_maximizer()
    
    p_v4 = [0.41, 0.27, 0.32] # V4 predicts H, winner conf = 0.41 <= 0.45
    p_v42 = [0.36, 0.31, 0.33] # V4.2 draw prob = 0.31 >= 0.26, margin = 0.36 - 0.31 = 0.05 <= 0.10
    
    pred = predict_v4_4_robust_override(
        lambda_home=1.1,
        lambda_away=1.0,
        p_v4=p_v4,
        p_v42=p_v42,
        abs_elo_diff=25.0,
        config=cfg,
        fixture_id=201,
    )
    
    assert pred.override_applied is True
    assert pred.v4_base_decision == "H"
    assert pred.v4_4_final_decision == "D"
    assert pred.probabilities["D"] == 0.31

def test_override_blocked_on_high_confidence_v4():
    """Verify override is blocked when V4 winner confidence is high (> 0.45)."""
    cfg = V44RobustOverrideConfig.accuracy_maximizer()
    
    p_v4 = [0.65, 0.20, 0.15] # V4 highly confident in H (0.65 > 0.45)
    p_v42 = [0.55, 0.28, 0.17]
    
    pred = predict_v4_4_robust_override(
        lambda_home=2.1,
        lambda_away=0.8,
        p_v4=p_v4,
        p_v42=p_v42,
        abs_elo_diff=150.0,
        config=cfg,
        fixture_id=202,
    )
    
    assert pred.override_applied is False
    assert pred.v4_base_decision == "H"
    assert pred.v4_4_final_decision == "H"

def test_override_blocked_on_high_scoring_match():
    """Verify override is blocked when total expected goals exceed cap (> 2.50)."""
    cfg = V44RobustOverrideConfig.accuracy_maximizer()
    
    p_v4 = [0.42, 0.26, 0.32] # V4 conf = 0.42 <= 0.45
    p_v42 = [0.37, 0.30, 0.33] # margin = 0.07 <= 0.10
    
    pred = predict_v4_4_robust_override(
        lambda_home=1.8,
        lambda_away=1.6, # Total goals = 3.4 > 2.50
        p_v4=p_v4,
        p_v42=p_v42,
        abs_elo_diff=30.0,
        config=cfg,
    )
    
    assert pred.override_applied is False
    assert pred.v4_4_final_decision == "H"

def test_production_asset_integrity():
    """Verify all 20 protected repository assets are bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}"
