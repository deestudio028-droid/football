"""Unit tests for Phase 32: Real Fixture API & Streamlit Dashboard Integration.

Verifies:
1. API provider normalization and attribute mapping.
2. Strict 5-league filtering (Premier League, La Liga, Bundesliga, Serie A, Ligue 1).
3. Matchday fixture discovery for specific dates.
4. Graceful handling of network and provider connection failures.
5. Empty provider response handling.
6. Pre-kickoff safety buffer enforcement (t_pred <= t_kickoff - 15m).
7. Prevention of retrospective prediction generation after kickoff.
8. Evaluation of completed match outcomes when prediction is locked.
9. Dynamic switching between OddAlerts API and Local Database providers.
10. Strict redaction of API credentials and tokens from all user-facing outputs.
11. Deduplication of duplicate fixture IDs returned by feeds.
12. Baseline model immutability across all 20 protected repository assets.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.fixture_service import DashboardFixture, FixtureService, TARGET_LEAGUES
from dashboard.prediction_service import DashboardMatchPrediction, PredictionService
from data.providers.football_fixture_provider import (
    MockTestFixtureProvider,
    ProviderConnectionError,
    UpcomingFixture,
)

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


def test_provider_fixture_normalization():
    """Verify provider correctly parses and normalizes fixture records."""
    mock = MockTestFixtureProvider([
        UpcomingFixture(
            fixture_id=9001,
            league_id=423,
            league_name="Premier League",
            home_team="Arsenal",
            away_team="Chelsea",
            home_team_id=1,
            away_team_id=2,
            scheduled_kickoff="2026-08-22T15:00:00+00:00",
            status="NS",
            provider="mock",
            provider_fixture_id="9001",
        )
    ])
    service = FixtureService(custom_provider=mock)
    fixtures, meta = service.get_todays_matches(date_str="2026-08-22", provider_name="mock")

    assert len(fixtures) == 1
    f = fixtures[0]
    assert f.fixture_id == 9001
    assert f.home_team == "Arsenal"
    assert f.away_team == "Chelsea"
    assert f.competition_name == "Premier League"
    assert f.status == "NS"


def test_5_league_filtering():
    """Verify fixtures outside the 5 target competitions are strictly excluded."""
    mock = MockTestFixtureProvider([
        UpcomingFixture(fixture_id=1, league_id=423, league_name="Premier League", home_team="A", away_team="B", home_team_id=1, away_team_id=2, scheduled_kickoff="2026-08-22T15:00:00+00:00", status="NS", provider="mock", provider_fixture_id="1"),
        UpcomingFixture(fixture_id=2, league_id=999, league_name="Unknown League", home_team="C", away_team="D", home_team_id=3, away_team_id=4, scheduled_kickoff="2026-08-22T15:00:00+00:00", status="NS", provider="mock", provider_fixture_id="2"),
        UpcomingFixture(fixture_id=3, league_id=419, league_name="La Liga", home_team="E", away_team="F", home_team_id=5, away_team_id=6, scheduled_kickoff="2026-08-22T17:00:00+00:00", status="NS", provider="mock", provider_fixture_id="3"),
    ])
    service = FixtureService(custom_provider=mock)
    fixtures, meta = service.get_todays_matches(date_str="2026-08-22", provider_name="mock")

    assert len(fixtures) == 2
    assert set(f.competition_id for f in fixtures) == {423, 419}


def test_matchday_fixture_discovery_by_date():
    """Verify fixtures are correctly filtered by calendar date."""
    mock = MockTestFixtureProvider([
        UpcomingFixture(fixture_id=1, league_id=423, league_name="Premier League", home_team="A", away_team="B", home_team_id=1, away_team_id=2, scheduled_kickoff="2026-08-22T15:00:00+00:00", status="NS", provider="mock", provider_fixture_id="1"),
        UpcomingFixture(fixture_id=2, league_id=423, league_name="Premier League", home_team="C", away_team="D", home_team_id=3, away_team_id=4, scheduled_kickoff="2026-08-23T15:00:00+00:00", status="NS", provider="mock", provider_fixture_id="2"),
    ])
    service = FixtureService(custom_provider=mock)

    f22, _ = service.get_todays_matches(date_str="2026-08-22", provider_name="mock")
    f23, _ = service.get_todays_matches(date_str="2026-08-23", provider_name="mock")

    assert len(f22) == 1 and f22[0].fixture_id == 1
    assert len(f23) == 1 and f23[0].fixture_id == 2


def test_api_failure_graceful_handling():
    """Verify connection errors return informative metadata without crashing."""
    class FailingProvider(MockTestFixtureProvider):
        def get_fixtures_by_date(self, *args, **kwargs):
            raise ProviderConnectionError("OddAlerts host timed out")

    service = FixtureService(custom_provider=FailingProvider())
    fixtures, meta = service.get_todays_matches(date_str="2026-08-22", provider_name="mock")

    assert fixtures == []
    assert meta["status"] == "CONNECTION_ERROR"
    assert "timed out" in meta["error_message"]


def test_empty_api_response_handling():
    """Verify empty provider responses return clean NO_FIXTURES metadata."""
    mock = MockTestFixtureProvider([])
    service = FixtureService(custom_provider=mock)
    fixtures, meta = service.get_todays_matches(date_str="2026-08-22", provider_name="mock")

    assert fixtures == []
    assert meta["status"] == "NO_FIXTURES"
    assert "No fixtures returned" in meta["error_message"]


def test_kickoff_buffer_and_timing_gate(prediction_service):
    """Verify pre-kickoff safety buffer allows pre-match predictions and rejects late ones."""
    now_str = "2026-08-22T12:00:00+00:00"
    future_fix = DashboardFixture(
        fixture_id=101, competition_id=423, competition_name="Premier League", season_name="2026/2027",
        home_team="Arsenal", away_team="Chelsea", home_id=1, away_id=2,
        scheduled_kickoff="2026-08-22T14:00:00+00:00", status="NS", provider="mock",
    )
    pred_future = prediction_service.predict_dashboard_fixture(future_fix, pre_kickoff_buffer_minutes=15, current_time_iso=now_str)
    assert pred_future.prediction_allowed is True
    assert pred_future.v4_decision in ("H", "D", "A")

    # Case 2: Kickoff in 5 minutes (within 15m buffer) -> Rejected
    late_fix = DashboardFixture(
        fixture_id=102, competition_id=423, competition_name="Premier League", season_name="2026/2027",
        home_team="Arsenal", away_team="Chelsea", home_id=1, away_id=2,
        scheduled_kickoff="2026-08-22T12:10:00+00:00", status="NS", provider="mock",
    )
    pred_late = prediction_service.predict_dashboard_fixture(late_fix, pre_kickoff_buffer_minutes=15, current_time_iso=now_str)
    assert pred_late.prediction_allowed is False
    assert "kickoff passed before prediction lock" in pred_late.status_message


def test_completed_match_evaluation(prediction_service):
    """Verify finished matches evaluate correctness against actual outcome."""
    ft_fix = DashboardFixture(
        fixture_id=103, competition_id=423, competition_name="Premier League", season_name="2026/2027",
        home_team="Arsenal", away_team="Chelsea", home_id=1, away_id=2,
        scheduled_kickoff="2026-08-22T10:00:00+00:00", status="FT", provider="mock",
        home_goals=2, away_goals=1, actual_outcome="H",
    )
    pred_ft = prediction_service.predict_dashboard_fixture(ft_fix, pre_kickoff_buffer_minutes=15)
    assert pred_ft.prediction_allowed is True
    assert pred_ft.status_message == "EVALUATED"
    assert pred_ft.actual_outcome == "H"
    assert pred_ft.v4_correct == (pred_ft.v4_decision == "H")
    assert pred_ft.v4_6_correct == (pred_ft.v4_6_decision == "H")


def test_provider_switching(fixture_service):
    """Verify switching between OddAlerts API and Local Database."""
    f_db, meta_db = fixture_service.get_todays_matches(date_str="2026-08-22", provider_name="local_db")
    assert meta_db["provider_key"] == "local_db"
    assert meta_db["provider_display"] == "Local Database"

    f_api, meta_api = fixture_service.get_todays_matches(date_str="2026-08-22", provider_name="oddalerts")
    assert meta_api["provider_key"] == "oddalerts"
    assert meta_api["provider_display"] == "OddAlerts API"


def test_credential_redaction(fixture_service):
    """Verify no API tokens or keys appear in fixture metadata or error messages."""
    fixtures, meta = fixture_service.get_todays_matches(date_str="2026-08-22", provider_name="oddalerts")

    meta_str = json.dumps(meta)
    assert "WJRX" not in meta_str
    assert "api_token" not in meta_str

    for f in fixtures:
        f_str = str(f)
        assert "WJRX" not in f_str
        assert "api_token" not in f_str


def test_duplicate_fixture_deduplication():
    """Verify duplicated fixture records in feed are deduplicated to exactly one."""
    mock = MockTestFixtureProvider([
        UpcomingFixture(fixture_id=501, league_id=423, league_name="Premier League", home_team="A", away_team="B", home_team_id=1, away_team_id=2, scheduled_kickoff="2026-08-22T15:00:00+00:00", status="NS", provider="mock", provider_fixture_id="501"),
        UpcomingFixture(fixture_id=501, league_id=423, league_name="Premier League", home_team="A", away_team="B", home_team_id=1, away_team_id=2, scheduled_kickoff="2026-08-22T15:00:00+00:00", status="NS", provider="mock", provider_fixture_id="501"),
    ])
    service = FixtureService(custom_provider=mock)
    fixtures, meta = service.get_todays_matches(date_str="2026-08-22", provider_name="mock")

    assert len(fixtures) == 1
    assert fixtures[0].fixture_id == 501


def test_physical_gate_checks_in_prediction(prediction_service):
    """Verify all 5 physical gate checks are present and Boolean."""
    res = prediction_service.predict_manual_matchup("Arsenal", "Chelsea", "Premier League")
    assert res is not None

    checks = res.v4_6_gate_checks
    assert len(checks) == 5
    assert "draw_prob_ge_0_26" in checks
    assert "winner_margin_le_0_10" in checks
    assert "v4_winner_conf_le_0_45" in checks
    assert "abs_elo_diff_le_100" in checks
    assert "tot_expected_goals_le_2_50" in checks
    for k, v in checks.items():
        assert isinstance(v, bool)


def test_production_asset_integrity():
    """Verify all 20 protected repository baseline assets remain 100% bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}: expected {exp}, got {act}"
