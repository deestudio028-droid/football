"""Unit tests for Step 1: V4.1 Prospective Candidate Training."""
import hashlib
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"))

from dixon_coles_engine import compute_dixon_coles_matrix_fast, compute_1x2_from_score_matrix

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

V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"


def test_20_protected_hashes_unmodified():
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Protected hash mismatch on {rel}"


def test_v4_1_artifact_integrity():
    assert V4_1_PATH.exists(), f"V4.1 artifact not found at {V4_1_PATH}"
    with open(V4_1_PATH, "rb") as f:
        payload = pickle.load(f)

    assert isinstance(payload, dict)
    assert payload["model_name"] == "poisson_venue_elo_online_ad"
    assert payload["model_version"] == "v4.1-champion-dc-elo-stacking-2025-26-trained"
    assert payload["feature_version"] == "v1.0"
    assert payload["n_features"] == 91
    assert len(payload["feature_columns"]) == 91
    assert payload["training_rows"] == 10734
    assert len(payload["training_seasons"]) == 6
    assert "2025/2026" in payload["training_seasons"]
    assert payload["holdout_used_in_training"] is True


def test_v4_1_deterministic_inference():
    with open(V4_1_PATH, "rb") as f:
        payload = pickle.load(f)

    prep = payload["preprocessor"]
    mh = payload["model_home_goals"]
    ma = payload["model_away_goals"]

    # Dummy test matrix matching 91 features
    dummy_x = pd.DataFrame(np.ones((5, 91)), columns=payload["feature_columns"])
    dummy_x["competition_id"] = [423, 419, 477, 499, 200]

    E = prep.transform(dummy_x)
    assert E.shape == (5, 95)

    lh = mh.predict(E)
    la = ma.predict(E)
    assert np.all(np.isfinite(lh)) and np.all(lh > 0)
    assert np.all(np.isfinite(la)) and np.all(la > 0)

    # Compute probabilities
    probs = np.zeros((5, 3))
    for i in range(5):
        M = compute_dixon_coles_matrix_fast(lh[i], la[i], rho=-0.08)
        probs[i] = compute_1x2_from_score_matrix(M)

    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
