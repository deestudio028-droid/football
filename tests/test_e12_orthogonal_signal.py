"""Unit tests for Phase 37 Experiment E12: Orthogonal Signal & Bounded Lambda Adjustment."""
import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e12_orthogonal_signal"))
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"))

from bounded_lambda_adjustment import apply_bounded_lambda_adjustment
from residual_models import BoundedResidualModel

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


def test_20_protected_hashes_unmodified():
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Protected hash mismatch on {rel}"


def test_bounded_adjustment_clipping():
    lam_h = np.array([1.50, 1.20, 2.00])
    lam_a = np.array([1.00, 1.40, 0.80])
    dh = np.array([0.50, -0.80, 0.02])
    da = np.array([-0.30, 0.40, -0.01])

    bound = 0.05
    probs, lh_adj, la_adj = apply_bounded_lambda_adjustment(lam_h, lam_a, dh, da, bound=bound, rho=-0.08)

    assert np.all(lh_adj <= lam_h * np.exp(bound) + 1e-6)
    assert np.all(lh_adj >= lam_h * np.exp(-bound) - 1e-6)
    assert np.all(la_adj <= lam_a * np.exp(bound) + 1e-6)
    assert np.all(la_adj >= lam_a * np.exp(-bound) - 1e-6)

    assert probs.shape == (3, 3)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_residual_model_regularization():
    np.random.seed(42)
    N = 100
    X_tr = np.random.randn(N, 5)
    lam_h = np.ones(N) * 1.5
    lam_a = np.ones(N) * 1.2
    goals_h = np.random.poisson(1.5, size=N)
    goals_a = np.random.poisson(1.2, size=N)

    model = BoundedResidualModel(model_type="ridge", alpha=100.0)
    model.fit(X_tr, lam_h, lam_a, goals_h, goals_a)

    X_te = np.random.randn(10, 5)
    dh, da = model.predict_deltas(X_te)
    assert len(dh) == 10
    assert len(da) == 10
    assert np.all(np.abs(dh) < 1.0)
    assert np.all(np.abs(da) < 1.0)
