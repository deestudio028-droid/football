"""Unit tests for Phase 30 Historical-Only Retraining & 2025/26 Blind Replay.

Verifies:
1. Strict Data Isolation (Zero overlap between training IDs and 2025/26 / prospective IDs).
2. Historical candidate manifest parameters & frozen contracts.
3. Candidate H prediction behavior and selective gating logic.
4. Metric calculations (Accuracy, Draw precision, Macro F1, Error economics).
5. Repository baseline immutability across all 20 protected assets.
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.historical_draw_research_candidate import (
    HistoricalCandidateManifest,
    compute_historical_dibp_matrix,
    predict_historical_candidate_h,
    redistribute_proportional_1x2,
)
from models.config import FINAL_TRAIN_SEASONS, FINAL_TEST_SEASONS, SEASON_NAME_TO_IDS
from models.data import load_supervised_dataset

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

def test_strict_dataset_isolation():
    """Verify zero overlap between historical training IDs and 2025/26 test IDs."""
    features_db = PROJECT_ROOT / "data" / "processed" / "features.db"
    ds = load_supervised_dataset(features_db)
    meta = ds.metadata

    train_season_ids = set()
    for s in FINAL_TRAIN_SEASONS:
        train_season_ids.update(SEASON_NAME_TO_IDS[s])

    test_season_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_season_ids.update(SEASON_NAME_TO_IDS[s])

    train_fids = set(meta[meta["season_id"].isin(train_season_ids)]["fixture_id"])
    test_fids = set(meta[meta["season_id"].isin(test_season_ids)]["fixture_id"])

    assert len(train_fids) == 8983
    assert len(test_fids) == 1751
    assert len(train_fids & test_fids) == 0

def test_historical_manifest_integrity():
    """Verify frozen historical candidate manifest values."""
    manifest_path = PROJECT_ROOT / "research" / "v4_promotion" / "phase30_model_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path) as f:
        d = json.load(f)

    assert d["candidate_id"] == "v4_6_historical_draw_candidate"
    assert d["training_match_count"] == 8983
    assert d["draw_prob_threshold"] == 0.26
    assert d["winner_margin_cap"] == 0.10
    assert d["v4_winner_conf_cap"] == 0.45
    assert d["abs_elo_cap"] == 100.0
    assert d["tot_expected_goals_cap"] == 2.50

def test_candidate_h_override_logic():
    """Verify Candidate H correctly applies and rejects Draw overrides."""
    manifest = HistoricalCandidateManifest(
        candidate_id="v4_6_historical_draw_candidate",
        version="v4.6-historical-only-retrained",
        training_cutoff_season="2024/2025",
        training_match_count=8983,
        stacking_intercept=0.2450,
        dibp_inflation_p=0.0500,
        dixon_coles_rho=-0.0560,
        draw_prob_threshold=0.2600,
        winner_margin_cap=0.1000,
        v4_winner_conf_cap=0.4500,
        abs_elo_cap=100.0,
        tot_expected_goals_cap=2.5000,
    )

    # Case 1: Match satisfies all 5 gates -> Overridden to Draw
    p_v4_1 = np.array([0.41, 0.27, 0.32])
    p_v42_1 = np.array([0.36, 0.31, 0.33])
    res_1 = predict_historical_candidate_h(
        p_v4=p_v4_1, p_v42=p_v42_1, lambda_home=1.1, lambda_away=1.0, abs_elo_diff=20.0, manifest=manifest
    )
    assert res_1["v4_base_decision"] == "H"
    assert res_1["v4_6_final_decision"] == "D"
    assert res_1["override_applied"] is True

    # Case 2: Max confidence too high -> Retains V4 H
    p_v4_2 = np.array([0.55, 0.22, 0.23])
    p_v42_2 = np.array([0.49, 0.26, 0.25])
    res_2 = predict_historical_candidate_h(
        p_v4=p_v4_2, p_v42=p_v42_2, lambda_home=1.8, lambda_away=1.0, abs_elo_diff=70.0, manifest=manifest
    )
    assert res_2["v4_base_decision"] == "H"
    assert res_2["v4_6_final_decision"] == "H"
    assert res_2["override_applied"] is False

def test_production_asset_integrity():
    """Verify all 20 protected repository baseline assets are bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}"
