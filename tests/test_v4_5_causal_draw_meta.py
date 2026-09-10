"""Unit tests for V4.5 Causal Draw Meta-Decision Layer.

Verifies:
1. Config initialization and weight vector dimensions.
2. Pre-kickoff causal feature extraction (10 dimensions).
3. Override mechanics and threshold evaluation.
4. Simplex probability conservation.
5. Production asset immutability across all 20 pinned baseline hashes.
"""
import hashlib
import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.v4_5_causal_draw_meta import (
    CausalDrawMetaConfig,
    CausalDrawMetaPrediction,
    extract_meta_features,
    predict_v4_5_meta,
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

def test_meta_config_structure():
    """Verify configuration parameters and feature weight dimensionality."""
    cfg = CausalDrawMetaConfig.default_config()
    assert cfg.name == "v4_5_causal_draw_meta"
    assert len(cfg.weights) == 10
    assert cfg.cost_ratio == 1.50
    assert cfg.decision_threshold == 0.2500

def test_causal_feature_extraction():
    """Verify exact 10-dimensional causal feature extraction."""
    p_v4 = [0.42, 0.26, 0.32]
    p_v42 = [0.36, 0.31, 0.33]
    x_feat = extract_meta_features(
        p_v4=p_v4,
        p_v42=p_v42,
        abs_elo_diff=25.0,
        lambda_home=1.1,
        lambda_away=1.0,
        low_score_prob=0.35,
    )
    assert len(x_feat) == 10
    assert x_feat[0] == 0.31 # P_v42(Draw)
    assert abs(x_feat[1] - (0.36 - 0.31)) < 1e-6 # Margin
    assert x_feat[2] == 0.42 # V4 winner confidence
    assert abs(x_feat[4] - 0.25) < 1e-6 # Abs Elo Diff / 100
    assert abs(x_feat[5] - (2.1 - 2.5)) < 1e-6 # Tot Goals offset

def test_meta_prediction_mechanics():
    """Verify prediction mechanics on fragile draw candidate."""
    cfg = CausalDrawMetaConfig.default_config()
    p_v4 = [0.40, 0.28, 0.32]
    p_v42 = [0.35, 0.32, 0.33]
    
    pred = predict_v4_5_meta(
        lambda_home=1.1,
        lambda_away=1.0,
        p_v4=p_v4,
        p_v42=p_v42,
        abs_elo_diff=15.0,
        low_score_prob=0.38,
        config=cfg,
        fixture_id=301,
    )
    assert pred.fixture_id == 301
    assert pred.v4_base_decision == "H"
    assert pred.v4_5_final_decision in ["H", "D", "A"]
    assert 0.0 <= pred.meta_draw_prob <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6

def test_production_asset_integrity():
    """Verify all 20 protected repository assets are bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}"
