"""Unit tests for Phase Prospective: Step 2 Live Forecast Lock."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
V4_1_EXPECTED_SHA256 = "cbb00b32c3ce4dbf600564a95d11b67a0c236f6b34cd184a9ca8abcc8ab94ac8"

TEST_DIR = PROJECT_ROOT / "research/v5_model_improvement/v4_1_prospective_test"
LEDGER_CSV = TEST_DIR / "02_live_forecast_ledger.csv"
INTEGRITY_JSON = TEST_DIR / "03_prediction_integrity.json"
CLOCK_CSV = TEST_DIR / "04_information_clock_log.csv"


def test_20_protected_hashes_unmodified():
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Protected hash mismatch on {rel}"


def test_v4_1_model_artifact_hashes():
    act_md5 = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    act_sha256 = hashlib.sha256(V4_1_PATH.read_bytes()).hexdigest()
    assert act_md5 == V4_1_EXPECTED_MD5, f"V4.1 MD5 mismatch: expected {V4_1_EXPECTED_MD5}, got {act_md5}"
    assert act_sha256 == V4_1_EXPECTED_SHA256, f"V4.1 SHA256 mismatch: expected {V4_1_EXPECTED_SHA256}, got {act_sha256}"


def test_ledger_exists_and_unempty():
    assert LEDGER_CSV.exists(), f"Ledger file not found at {LEDGER_CSV}"
    df = pd.read_csv(LEDGER_CSV)
    assert len(df) > 0, "Forecast ledger is empty"
    assert len(df) == 99, f"Expected 99 locked predictions, got {len(df)}"


def test_probability_simplex_satisfied():
    df = pd.read_csv(LEDGER_CSV)
    probs_sum = df["p_home"] + df["p_draw"] + df["p_away"]
    assert np.allclose(probs_sum, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"
    assert np.all(df["p_home"] >= 0.0) and np.all(df["p_home"] <= 1.0)
    assert np.all(df["p_draw"] >= 0.0) and np.all(df["p_draw"] <= 1.0)
    assert np.all(df["p_away"] >= 0.0) and np.all(df["p_away"] <= 1.0)


def test_strictly_pre_kickoff_predictions():
    df = pd.read_csv(CLOCK_CSV)
    assert np.all(df["pre_kickoff_lead_seconds"] > 0), "Found predictions made after kickoff"


def test_no_duplicate_fixture_predictions():
    df = pd.read_csv(LEDGER_CSV)
    assert len(df["fixture_id"]) == len(df["fixture_id"].unique()), "Found duplicate fixture predictions in ledger"


def test_integrity_manifest_structure():
    assert INTEGRITY_JSON.exists(), f"Integrity manifest not found at {INTEGRITY_JSON}"
    with open(INTEGRITY_JSON) as f:
        data = json.load(f)

    assert data["candidate_model"]["md5_hash"] == V4_1_EXPECTED_MD5
    assert data["candidate_model"]["sha256_hash"] == V4_1_EXPECTED_SHA256
    assert data["frozen_production_model"]["md5_hash"] == "06841f0c03c8597b2b8cd8f8ab064864"
    assert data["integrity_checks"]["probability_simplex_satisfied_all"] is True
    assert data["integrity_checks"]["strictly_pre_kickoff_all"] is True
    assert data["integrity_checks"]["zero_outcome_leakage"] is True
