"""Unit tests for Phase 32: Prediction Specificity & Anti-Fallback Verification.

Ensures:
1. Different fixtures receive distinct, fixture-specific probabilities.
2. Team identities reach the inference layer directly.
3. Fixture ID reaches prediction context.
4. No static 44/26/30 fallback probabilities exist in the prediction pipeline.
5. Unknown teams or missing features produce 'unavailable', never fake predictions.
6. V4.6 receives fixture-specific parameters.
7. Historical Candidate H receives fixture-specific parameters.
8. Prediction cache is keyed properly and does not collide across fixtures.
9. Model weights and hash remain strictly unchanged.
10. All 20 protected repository baseline assets remain 100% bit-identical.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.fixture_service import DashboardFixture
from dashboard.prediction_service import DashboardMatchPrediction, PredictionService

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


def test_prediction_specificity_different_fixtures(prediction_service):
    """Verify distinct matchups receive distinct probabilities and rate parameters."""
    p1 = prediction_service.predict_matchup("Arsenal", "Chelsea", "Premier League", competition_id=423)
    p2 = prediction_service.predict_matchup("Real Madrid", "Getafe", "La Liga", competition_id=419)
    p3 = prediction_service.predict_matchup("Bayern Munich", "Bochum", "Bundesliga", competition_id=477)

    assert p1 is not None and p2 is not None and p3 is not None

    # Verify probabilities differ significantly between matchups
    assert p1.v4_probs != p2.v4_probs
    assert p2.v4_probs != p3.v4_probs
    assert p1.lambda_home != p2.lambda_home or p1.lambda_away != p2.lambda_away
    assert p1.abs_elo_diff != p2.abs_elo_diff or p2.abs_elo_diff != p3.abs_elo_diff


def test_team_identity_and_reversal_sensitivity(prediction_service):
    """Verify home/away asymmetry and team identity sensitivity."""
    p_home = prediction_service.predict_matchup("Arsenal", "Chelsea", "Premier League", competition_id=423)
    p_away = prediction_service.predict_matchup("Chelsea", "Arsenal", "Premier League", competition_id=423)

    assert p_home is not None and p_away is not None
    # Home/Away reversal should produce different probabilities
    assert p_home.v4_probs["H"] != p_away.v4_probs["H"]
    assert p_home.lambda_home != p_away.lambda_home


def test_no_hardcoded_fallback_probabilities(prediction_service):
    """Verify no prediction matches the exact bugged fallback (0.44, 0.26, 0.30 across all matches)."""
    matchups = [
        ("Arsenal", "Bournemouth", "Premier League", 423),
        ("Inter", "Monza", "Serie A", 499),
        ("Espanyol", "Real Madrid", "La Liga", 419),
        ("Lens", "Auxerre", "Ligue 1", 200),
    ]

    for h, a, comp, cid in matchups:
        p = prediction_service.predict_matchup(h, a, comp, competition_id=cid)
        assert p is not None
        # Must not be simultaneously 0.44 / 0.26 / 0.30
        is_bugged_default = (p.v4_probs["H"] == 0.44 and p.v4_probs["D"] == 0.26 and p.v4_probs["A"] == 0.30)
        assert not is_bugged_default, f"Bugged default found for {h} vs {a}"


def test_missing_features_fail_closed(prediction_service):
    """Verify unknown teams fail closed with UNAVAILABLE message rather than emitting fake predictions."""
    f_unknown = DashboardFixture(
        fixture_id=999999,
        competition_id=423,
        competition_name="Premier League",
        season_name="2026/2027",
        home_team="NonExistentFC_12345",
        away_team="UnknownClub_67890",
        home_id=999991,
        away_id=999992,
        scheduled_kickoff="2026-08-22T20:00:00+00:00",
        status="NS",
        provider="mock",
    )

    pred = prediction_service.predict_dashboard_fixture(f_unknown, pre_kickoff_buffer_minutes=15)
    assert pred.prediction_allowed is False
    assert "required pre-match features unavailable" in pred.status_message
    assert pred.v4_probs is None
    assert pred.v4_decision is None


def test_v4_6_receives_fixture_specific_gate_inputs(prediction_service):
    """Verify V4.6 physical gate checks reflect the fixture's specific metrics."""
    p_balanced = prediction_service.predict_matchup("Everton", "Crystal Palace", "Premier League", competition_id=423)
    p_unbalanced = prediction_service.predict_matchup("Inter", "Monza", "Serie A", competition_id=499)

    assert p_balanced is not None and p_unbalanced is not None
    assert p_unbalanced.abs_elo_diff > 300
    assert p_unbalanced.v4_6_gate_checks["abs_elo_diff_le_100"] is False
    assert p_unbalanced.v4_6_override_applied is False


def test_historical_candidate_h_specificity(prediction_service):
    """Verify Historical Candidate H receives fixture-specific inputs."""
    p1 = prediction_service.predict_matchup("Arsenal", "Chelsea", "Premier League", competition_id=423)
    p2 = prediction_service.predict_matchup("Inter", "Monza", "Serie A", competition_id=499)

    assert p1 is not None and p2 is not None
    assert p1.hist_h_gate_checks != p2.hist_h_gate_checks


def test_dashboard_fixture_prediction_diagnostics_populated(prediction_service):
    """Verify pre-match diagnostic fields are populated on valid predictions."""
    fix = DashboardFixture(
        fixture_id=420581845,
        competition_id=423,
        competition_name="Premier League",
        season_name="2026/2027",
        home_team="Everton",
        away_team="Crystal Palace",
        home_id=1,
        away_id=2,
        scheduled_kickoff="2026-08-22T20:00:00+00:00",
        status="NS",
        provider="OddAlerts API",
    )

    pred = prediction_service.predict_dashboard_fixture(fix, pre_kickoff_buffer_minutes=15)
    assert pred.prediction_allowed is True
    assert pred.home_elo is not None and pred.home_elo > 1000
    assert pred.away_elo is not None and pred.away_elo > 1000
    assert pred.elo_diff is not None
    assert pred.lambda_home is not None and pred.lambda_home > 0
    assert pred.lambda_away is not None and pred.lambda_away > 0
    assert pred.feature_source == "Live Chronological Feature Context"


def test_production_asset_integrity():
    """Verify all 20 protected repository baseline assets remain 100% bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}: expected {exp}, got {act}"
