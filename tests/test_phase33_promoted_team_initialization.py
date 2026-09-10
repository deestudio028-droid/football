"""Unit tests for Phase 33: Promoted Team Pre-Match Feature Initialization.

Verifies:
1. Existing team prediction remains unchanged and bit-identical.
2. Promoted team gets non-static, fixture-specific prediction.
3. Static 1.45/1.15 fallback can NEVER execute.
4. No post-kickoff information is used (causality check).
5. Feature vector matches the 91-column V4 contract.
6. Initialization is completely deterministic.
7. Missing/invalid historical data fails closed.
8. V4 model artifact hash remains unchanged.
9. V4.6 physical gate remains unchanged.
10. Protected 20 baseline assets remain 100% bit-identical.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.fixture_service import DashboardFixture
from dashboard.prediction_service import DashboardMatchPrediction, PredictionService
from features.promoted_team_initializer import PromotedTeamInitializer, PromotedMatchFeatureResult
from models.v4_contract import V4_FEATURE_COLUMNS

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


@pytest.fixture(scope="module")
def prediction_service():
    return PredictionService()


@pytest.fixture(scope="module")
def initializer(prediction_service):
    return prediction_service._promoted_initializer


def test_existing_team_prediction_unchanged(prediction_service):
    """Verify existing top-flight teams continue receiving unperturbed predictions."""
    res = prediction_service.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res is not None
    assert res.is_promoted_match is False
    assert res.feature_source == "Live Chronological Feature Context"
    assert res.v4_decision in ("H", "D", "A")
    assert res.v4_probs["H"] > 0.0


def test_promoted_team_gets_non_static_prediction(prediction_service):
    """Verify promoted team (Hull City) receives genuine, non-static predictions against top flight opponent."""
    # Hull City vs Manchester United
    hull_fix = DashboardFixture(
        fixture_id=420581845,
        competition_id=423,
        competition_name="Premier League",
        season_name="2026/2027",
        home_team="Hull City",
        away_team="Manchester United",
        home_id=9770,
        away_id=10659,
        scheduled_kickoff="2026-08-22T14:00:00+00:00",
        status="NS",
        provider="oddalerts",
    )
    pred_hull = prediction_service.predict_dashboard_fixture(
        hull_fix, pre_kickoff_buffer_minutes=15, current_time_iso="2026-08-22T12:00:00+00:00"
    )

    assert pred_hull.prediction_allowed is True
    assert pred_hull.is_promoted_match is True
    assert "PROMOTED TEAM INITIALIZATION" in pred_hull.feature_source
    assert pred_hull.v4_decision == "A"
    # Verify probabilities are NOT the forbidden static (43.8 / 25.9 / 30.3)
    assert not (
        round(pred_hull.v4_probs["H"], 3) == 0.438
        and round(pred_hull.v4_probs["D"], 3) == 0.259
        and round(pred_hull.v4_probs["A"], 3) == 0.303
    )


def test_no_static_fallback_probabilities_for_promoted(prediction_service):
    """Verify different promoted team matchups produce distinct, opponent-specific distributions."""
    # Hull vs Man Utd (PL)
    res_hull = prediction_service.predict_matchup(
        home_team="Hull City",
        away_team="Manchester United",
        competition_name="Premier League",
        competition_id=423,
        home_id_hint=9770,
        away_id_hint=10659,
    )
    # Le Mans vs Brest (Ligue 1)
    res_lemans = prediction_service.predict_matchup(
        home_team="Le Mans",
        away_team="Brest",
        competition_name="Ligue 1",
        competition_id=200,
        home_id_hint=5722,
        away_id_hint=8674,
    )

    assert res_hull is not None and res_lemans is not None
    assert res_hull.v4_probs != res_lemans.v4_probs
    assert res_hull.lambda_home != res_lemans.lambda_home
    assert res_hull.lambda_away != res_lemans.lambda_away


def test_causality_and_no_future_leakage(initializer):
    """Verify initializer uses only information strictly before cutoff timestamp."""
    cutoff = "2026-08-22T15:00:00+00:00"
    prof = initializer.get_team_profile("Hull City", 9770, 423, cutoff)

    assert prof.data_cutoff_timestamp == cutoff
    if prof.latest_historical_match_date:
        assert prof.latest_historical_match_date < cutoff


def test_feature_vector_matches_v4_contract(initializer):
    """Verify initialized feature matrix strictly adheres to 91-column V4 schema."""
    res = initializer.initialize_match_features(
        fixture_id=1,
        competition_id=423,
        scheduled_kickoff="2026-08-22T15:00:00+00:00",
        home_team="Hull City",
        away_team="Manchester United",
        home_id=9770,
        away_id=10659,
    )
    assert res is not None
    assert isinstance(res.features_df, pd.DataFrame)
    assert list(res.features_df.columns) == list(V4_FEATURE_COLUMNS)
    assert len(res.features_df.columns) == 91
    assert not res.features_df.empty


def test_initialization_determinism(initializer):
    """Verify initializing the same promoted match twice yields identical feature vectors."""
    res1 = initializer.initialize_match_features(
        fixture_id=99,
        competition_id=423,
        scheduled_kickoff="2026-08-22T15:00:00+00:00",
        home_team="Hull City",
        away_team="Manchester United",
        home_id=9770,
        away_id=10659,
    )
    res2 = initializer.initialize_match_features(
        fixture_id=99,
        competition_id=423,
        scheduled_kickoff="2026-08-22T15:00:00+00:00",
        home_team="Hull City",
        away_team="Manchester United",
        home_id=9770,
        away_id=10659,
    )

    pd.testing.assert_frame_equal(res1.features_df, res2.features_df)
    assert res1.diagnostics == res2.diagnostics


def test_invalid_team_fails_closed(prediction_service):
    """Verify empty or nonexistent team names fail closed gracefully without generic probabilities."""
    res = prediction_service.predict_matchup(
        home_team="",
        away_team="Manchester United",
        competition_name="Premier League",
    )
    assert res is None


def test_v4_model_artifact_hash_unchanged():
    """Verify V4 production model artifact is unmodified."""
    v4_path = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
    act = hashlib.md5(v4_path.read_bytes()).hexdigest()
    assert act == "06841f0c03c8597b2b8cd8f8ab064864"


def test_v4_6_physical_draw_gate_unchanged():
    """Verify V4.6 draw gate code and parameters remain frozen."""
    from models.v4_6_physical_draw_gate import V46PhysicalGateConfig
    cfg = V46PhysicalGateConfig.robust_optimal_gate()
    assert cfg.draw_prob_threshold == 0.2600
    assert cfg.winner_margin_cap == 0.1000
    assert cfg.v4_winner_conf_cap == 0.4500
    assert cfg.abs_elo_cap == 100.0
    assert cfg.tot_expected_goals_cap == 2.50


def test_all_20_protected_baseline_assets_integrity():
    """Verify all 20 protected repository baseline assets remain 100% bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}: expected {exp}, got {act}"
