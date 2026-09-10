"""Unit tests for Phase Prospective: V4.1 Dashboard Production Promotion."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.model_registry import ModelRegistry, get_model_registry
from dashboard.prediction_service import PredictionService, DashboardMatchPrediction, SingleMatchPredictionResult
from dashboard.fixture_service import DashboardFixture

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
V4_1_EXP_MD5 = "145f918d933eb343c0f63ca342b10289"
V4_0_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_0_EXP_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
LEDGER_CSV = PROJECT_ROOT / "research/v5_model_improvement/v4_1_prospective_test/02_live_forecast_ledger.csv"


def test_20_protected_hashes_unmodified():
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Protected hash mismatch on {rel}"


def test_model_registry_production_model():
    reg = get_model_registry()
    prod = reg.get_production_model()
    assert prod.model_id == "v4_1_prospective_candidate"
    assert prod.version == "v4.1-champion-dc-elo-stacking-2025-26-trained"
    assert prod.role == "PRODUCTION"
    assert "PROSPECTIVE LIVE EVALUATION ACTIVE" in prod.status
    assert prod.md5_hash == V4_1_EXP_MD5


def test_model_registry_baseline_benchmark_model():
    reg = get_model_registry()
    bench = reg.get_baseline_model()
    assert bench.model_id == "v4_0_draw_champion"
    assert bench.role == "BENCHMARK"
    assert "FROZEN BASELINE" in bench.status


def test_prediction_service_production_inference():
    ps = PredictionService()
    res = ps.predict_matchup("Arsenal", "Chelsea", "Premier League", competition_id=423)
    assert res is not None

    # Check V4.1 Metadata
    assert res.model_name == "v4_1_prospective_candidate"
    assert res.model_version == "v4.1-champion-dc-elo-stacking-2025-26-trained"
    assert res.model_file_md5 == V4_1_EXP_MD5
    assert res.information_cutoff == "2026-05-24T19:45:00.000000Z"
    assert res.evaluation_status == "PROSPECTIVE 2026 LIVE EVALUATION ACTIVE"

    # Check V4.1 Probabilities and Simplex
    p_h, p_d, p_a = res.v4_1_probs["H"], res.v4_1_probs["D"], res.v4_1_probs["A"]
    assert np.isclose(p_h + p_d + p_a, 1.0, atol=1e-2)
    assert p_h >= 0.0 and p_d >= 0.0 and p_a >= 0.0
    assert res.production_probs == res.v4_1_probs
    assert res.production_decision == res.v4_1_decision

    # Check V4.0 Benchmark remains accessible
    assert "H" in res.benchmark_v4_0_probs
    assert res.benchmark_v4_0_decision in ["H", "D", "A"]


def test_deterministic_inference():
    ps = PredictionService()
    res1 = ps.predict_matchup("Real Madrid", "Barcelona", "La Liga", competition_id=419)
    res2 = ps.predict_matchup("Real Madrid", "Barcelona", "La Liga", competition_id=419)

    assert res1.v4_1_probs == res2.v4_1_probs
    assert res1.v4_1_decision == res2.v4_1_decision
    assert res1.lambda_home == res2.lambda_home
    assert res1.lambda_away == res2.lambda_away


def test_prospective_99_match_ledger_unmodified():
    assert LEDGER_CSV.exists()
    df = pd.read_csv(LEDGER_CSV)
    assert len(df) == 99
    assert np.all(df["model_file_md5"] == V4_1_EXP_MD5)
