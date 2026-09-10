"""Unit tests for Phase 31: Model Evaluation & 2026/27 Dashboard.

Verifies:
1. Model registry correctly loads and inspects production / shadow models.
2. Fixture service provides supported leagues, team lists, and 2026/27 discovery.
3. Prediction service computes multi-model pre-match inference and gate checks.
4. Metrics service calculates error economics and agreement.
5. Evaluation service builds formal promotion review reports.
6. Baseline assets integrity is maintained across all 20 pinned files.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.evaluation_service import EvaluationService
from dashboard.fixture_service import FixtureService
from dashboard.metrics_service import MetricsService
from dashboard.model_registry import ModelRegistry, get_model_registry
from dashboard.prediction_service import PredictionService

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
def fixture_service():
    return FixtureService()


def test_model_registry_integrity():
    """Verify registry correctly exposes models, metadata, and frozen statuses."""
    registry = get_model_registry()
    prod = registry.get_production_model()
    shadow = registry.get_shadow_model()

    assert prod.model_id == "v4_1_prospective_candidate"
    assert "FROZEN" in prod.status
    assert prod.md5_hash == "145f918d933eb343c0f63ca342b10289"

    assert shadow.model_id == "v4_6_physical_draw_gate"
    assert shadow.status == "FROZEN"
    assert len(registry.list_all_models()) >= 5


def test_fixture_service_leagues_and_teams(fixture_service):
    """Verify fixture service queries 5 target competitions and team rosters."""
    leagues = fixture_service.get_supported_leagues()

    assert len(leagues) == 5
    assert 423 in leagues  # Premier League
    assert 419 in leagues  # La Liga
    assert 477 in leagues  # Bundesliga
    assert 499 in leagues  # Serie A
    assert 200 in leagues  # Ligue 1

    pl_teams = fixture_service.get_teams_by_league(423)
    assert len(pl_teams) > 0
    assert "Arsenal" in pl_teams or "Liverpool" in pl_teams or "Manchester City" in pl_teams

    disc = fixture_service.discover_2026_27_fixtures()
    assert "total_discovered" in disc
    assert "completed_ft" in disc


def test_prediction_service_single_match(prediction_service):
    """Verify prediction service computes probabilities and gate check conditions."""
    res = prediction_service.predict_manual_matchup(
        home_team="Arsenal",
        away_team="Chelsea",
        competition_name="Premier League",
        approx_lambda_home=1.35,
        approx_lambda_away=1.10,
        approx_elo_diff=25.0,
    )

    assert res is not None
    assert res.home_team == "Arsenal"
    assert res.away_team == "Chelsea"
    assert res.v4_decision in ("H", "D", "A")
    assert res.v4_6_decision in ("H", "D", "A")
    assert isinstance(res.v4_6_gate_checks, dict)
    assert len(res.v4_6_gate_checks) == 5


def test_metrics_service_error_economics_and_agreement():
    """Verify error economics and model agreement calculations."""
    y_true = ["H", "D", "A", "D", "H"]
    dec_v4 = ["H", "H", "A", "A", "H"]
    dec_v46 = ["H", "D", "A", "D", "D"]
    dec_h = ["H", "D", "A", "D", "D"]

    ee = MetricsService.calculate_error_economics(y_true, dec_v4, dec_v46)
    assert ee["good_draw_overrides_GAINED"] == 2  # Match 1 (D) & Match 3 (D)
    assert ee["bad_draw_overrides_SACRIFICED"] == 1  # Match 4 (H -> D, actual H)
    assert ee["net_gain"] == 1

    agree = MetricsService.calculate_model_agreement(dec_v4, dec_v46, dec_h)
    assert agree["v46_vs_h_agreement_pct"] == 100.0
    assert agree["v4_vs_v46_divergence_count"] == 3


def test_evaluation_service_promotion_report():
    """Verify evaluation service generates formal promotion report."""
    eval_service = EvaluationService()
    report = eval_service.generate_promotion_review_report()

    assert "Formal Model Promotion Review Artifact" in report
    assert ("v4_1_prospective_candidate" in report or "v4_0_draw_champion" in report)
    assert "v4_6_physical_draw_gate" in report


def test_production_asset_integrity():
    """Verify all 20 protected repository baseline assets remain 100% bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}: expected {exp}, got {act}"
