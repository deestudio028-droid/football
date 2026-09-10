"""Unit tests for Phase Prospective: V4.1 Information-Clock and Baseline Causality Audit."""
import hashlib
import pickle
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.online_attack_defense import fit_baseline_rates, compute_ad_states, BaselineRates

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
V4_1_EXPECTED_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_20_protected_hashes_unmodified():
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Protected hash mismatch on {rel}"


def test_v4_1_candidate_hash_unmodified():
    act = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act == V4_1_EXPECTED_MD5, f"V4.1 hash mismatch: expected {V4_1_EXPECTED_MD5}, got {act}"


def test_two_pass_timestamp_causality():
    # Construct a synthetic 2-match sequence at timestamp T=100 and T=200
    df_test = pd.DataFrame([
        {"fixture_id": 1, "unix": 100, "home_id": 10, "away_id": 20, "home_goals": 5.0, "away_goals": 0.0, "status": "FT"},
        {"fixture_id": 2, "unix": 100, "home_id": 30, "away_id": 40, "home_goals": 0.0, "away_goals": 3.0, "status": "FT"},
        {"fixture_id": 3, "unix": 200, "home_id": 10, "away_id": 30, "home_goals": 1.0, "away_goals": 1.0, "status": "FT"},
    ])
    base = BaselineRates(mu_home=1.5, mu_away=1.2, n_train=10)
    states = compute_ad_states(df_test, lr=0.02, baseline=base).set_index("fixture_id")

    # Match 1 and 2 happen at same timestamp: states must be initial 0.0
    assert states.loc[1, "A_home"] == 0.0
    assert states.loc[2, "A_away"] == 0.0

    # Match 3 at T=200 must reflect the update from Match 1 (Team 10 scored 5 goals > 1.5, so A_home > 0)
    assert states.loc[3, "A_home"] > 0.0


def test_zero_prospective_2026_contamination():
    with open(V4_1_PATH, "rb") as f:
        payload = pickle.load(f)

    assert payload["training_rows"] == 10734
    assert payload["information_cutoff"]["live_data_included"] is False
    assert payload["information_cutoff"]["cutoff_date"] == "2026-05-24T19:45:00.000000Z"
