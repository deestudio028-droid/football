"""Unit tests for Phase 42 Experiment E17: Forecast Status Optimization & Decision Quality."""
import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e17_forecast_status"))

from status_targets import compute_status_evaluation_targets
from status_models import AnalyticalStatusScorer, GradientBoostingStatusRegressor
from status_engine import WalkForwardStatusClassifier, compute_status_coverage_curve

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


def test_status_evaluation_targets():
    probs = np.array([[0.65, 0.25, 0.10], [0.30, 0.40, 0.30]])
    y_true = np.array(["H", "D"])

    df_t = compute_status_evaluation_targets(y_true, probs)

    assert len(df_t) == 2
    assert "match_rps" in df_t.columns
    assert "match_log_loss" in df_t.columns
    assert "top1_correct" in df_t.columns
    assert df_t.iloc[0]["top1_correct"] == 1
    assert df_t.iloc[1]["top1_correct"] == 1


def test_walk_forward_status_classifier():
    scores = np.array([0.95, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05])
    classifier = WalkForwardStatusClassifier(p_strong=75.0, p_lean=50.0, p_caution=25.0).fit_thresholds(scores)
    statuses = classifier.assign_status(scores)

    assert len(statuses) == 10
    assert "STRONG" in statuses
    assert "LEAN" in statuses
    assert "CAUTION" in statuses
    assert "AVOID" in statuses


def test_analytical_status_scorer():
    scorer = AnalyticalStatusScorer()
    norm_ent = np.array([0.15, 0.95])
    max_p = np.array([0.85, 0.36])
    margin = np.array([0.70, 0.02])

    scores = scorer.compute_score(norm_ent, max_p, margin)

    assert len(scores) == 2
    assert scores[0] > scores[1]
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_status_coverage_curve():
    scores = np.linspace(0.1, 0.9, 10)
    probs_e10 = np.tile([0.5, 0.3, 0.2], (10, 1))
    y_true = np.array(["H", "D", "H", "A", "H", "D", "H", "A", "H", "D"])

    df_cov = compute_status_coverage_curve(scores, probs_e10, y_true, coverage_levels=[1.0, 0.80, 0.50])

    assert len(df_cov) == 3
    assert df_cov.iloc[0]["retained_matches"] == 10
    assert df_cov.iloc[1]["retained_matches"] == 8
    assert df_cov.iloc[2]["retained_matches"] == 5
