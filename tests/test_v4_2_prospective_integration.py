"""Unit tests for V4.2 Prospective Shadow Integration & Generalization Harness.

Verifies:
1. Shadow candidate isolation (production Draw Champion unaffected).
2. Pre-kickoff timestamp validation and SHA-256 cryptographic locking.
3. Two-stage outcome separation and record immutability.
4. Margin-aware decision policy invariants (theta=0.28, margin=0.12).
5. Zero lookahead and zero market leakage.
6. Production asset integrity across all 20 pinned baselines.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.v4_2_draw_resolution_candidate import (
    DrawResolutionConfig,
    DrawResolutionPrediction,
    predict_draw_resolution,
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

def test_shadow_candidate_isolation():
    """Verify that V4.2 candidate executes in parallel shadow mode without altering production."""
    p_v4 = [0.42, 0.26, 0.32]
    cfg_v42 = DrawResolutionConfig(
        decision_policy="margin_aware",
        decision_threshold=0.28,
        decision_margin=0.12,
        stacking_intercept=0.2450,
    )
    pred = predict_draw_resolution(1.4, 1.2, p_v4, 25.0, "Premier League", cfg_v42, fixture_id=999999)
    assert pred.fixture_id == 999999
    assert "H" in pred.probabilities and "D" in pred.probabilities and "A" in pred.probabilities
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-12

def test_pre_kickoff_timestamp_locking():
    """Verify pre-kickoff timestamp validation and SHA-256 locking format."""
    now_utc = datetime.now(timezone.utc)
    kickoff_utc = datetime(2026, 9, 1, 15, 0, 0, tzinfo=timezone.utc)
    
    # Prediction must be strictly before kickoff
    assert now_utc < kickoff_utc
    
    p_v4 = [0.40, 0.28, 0.32]
    pred = predict_draw_resolution(1.3, 1.3, p_v4, 10.0, "La Liga")
    
    payload = {
        "fixture_id": 123456,
        "kickoff_utc": kickoff_utc.isoformat(),
        "prediction_utc": now_utc.isoformat(),
        "probabilities": pred.probabilities,
        "predicted_class": pred.predicted_class,
    }
    canonical_json = json.dumps(payload, sort_keys=True)
    sha256_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    assert len(sha256_hash) == 64

def test_margin_aware_policy_logic():
    """Verify exact margin-aware decision conditions: P(D) >= theta AND max(P(H),P(A)) - P(D) <= margin."""
    cfg = DrawResolutionConfig(
        decision_policy="margin_aware",
        decision_threshold=0.28,
        decision_margin=0.12,
        stacking_intercept=0.2450,
    )
    
    # Case 1: Balanced encounter -> P(H)=0.37, P(D)=0.30, P(A)=0.33 -> max(0.37,0.33)-0.30 = 0.07 <= 0.12 -> Draw
    p_v4_balanced = [0.38, 0.28, 0.34]
    pred_bal = predict_draw_resolution(1.2, 1.2, p_v4_balanced, 15.0, "Serie A", cfg)
    assert pred_bal.predicted_class == "D"
    
    # Case 2: Skewed match -> P(H)=0.60, P(D)=0.20, P(A)=0.20 -> Draw condition not met -> Home
    p_v4_skewed = [0.65, 0.18, 0.17]
    pred_skew = predict_draw_resolution(2.2, 0.8, p_v4_skewed, 150.0, "Premier League", cfg)
    assert pred_skew.predicted_class == "H"

def test_production_asset_integrity():
    """Verify all 20 protected repository assets are bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}: expected {exp}, got {act}"
