"""Phase 35 — Outcome Reconciliation & 5-League Matchday Discovery Test Suite.

Verifies:
1. All five target league mappings (Premier League, La Liga, Bundesliga, Serie A, Ligue 1).
2. Selected Chennai date filtering (UTC -> IST boundary conversion).
3. Robust score extraction (handling integer 0 goals without falsy bugs).
4. Actual Result (H/D/A) calculation for completed FT matches.
5. FT fixtures without score return 'Result unavailable' safely.
6. Live fixtures display status/score but are NOT evaluated as final outcomes.
7. Future NS fixtures display '--' actual results and are evaluated only after FT.
8. Pre-kickoff buffer preserves completed FT fixtures while gating future NS fixtures.
9. Fixture deduplication by unique fixture ID.
10. Multi-model evaluation (V4, V4.6, Hist H) correctly flags CORRECT / WRONG on FT matches.
11. 5-League breakdown metrics accurately track league counts.
12. All 20 protected repository baseline assets remain 100% bit-identical.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.fixture_service import DashboardFixture, FixtureService, TARGET_LEAGUES
from dashboard.prediction_service import PredictionService
from dashboard.time_utils import format_kickoff_ist, to_chennai_date
from data.providers.football_fixture_provider import MockTestFixtureProvider, UpcomingFixture

# Pinned MD5 hashes of 20 protected assets
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


def test_five_league_mapping_and_independent_breakdown():
    """Verify all 5 target leagues are present in TARGET_LEAGUES and tracked independently."""
    expected = {
        423: "Premier League",
        419: "La Liga",
        477: "Bundesliga",
        499: "Serie A",
        200: "Ligue 1",
    }
    assert TARGET_LEAGUES == expected

    mock_prov = MockTestFixtureProvider()
    for cid, lname in TARGET_LEAGUES.items():
        mock_prov.add_fixture(UpcomingFixture(
            fixture_id=1000 + cid,
            league_id=cid,
            league_name=lname,
            home_team=f"Home_{cid}",
            away_team=f"Away_{cid}",
            home_team_id=1,
            away_team_id=2,
            scheduled_kickoff="2026-08-22T14:00:00+00:00",
            status="NS",
            provider="mock",
            provider_fixture_id=str(1000 + cid),
        ))

    fs = FixtureService(custom_provider=mock_prov)
    fixtures, meta = fs.get_todays_matches(date_str="2026-08-22", custom_provider=mock_prov)

    assert len(fixtures) == 5
    for lname in expected.values():
        assert meta["league_breakdown"][lname] == 1


def test_zero_goals_score_extraction_no_falsy_bug():
    """Verify clean extraction of integer 0 scores (e.g. 3-0, 0-0, 0-2) without falsy bugs."""
    mock_prov = MockTestFixtureProvider()
    # 3-0 Home Win (away has 0 goals)
    mock_prov.add_fixture(UpcomingFixture(
        fixture_id=201,
        league_id=423,
        league_name="Premier League",
        home_team="Arsenal",
        away_team="Coventry City",
        home_team_id=9825,
        away_team_id=10125,
        scheduled_kickoff="2026-08-21T19:00:00+00:00",  # 12:30 AM IST Aug 22
        status="FT",
        provider="mock",
        provider_fixture_id="201",
        home_goals=3,
        away_goals=0,
    ))
    # 0-0 Draw
    mock_prov.add_fixture(UpcomingFixture(
        fixture_id=202,
        league_id=419,
        league_name="La Liga",
        home_team="Sevilla",
        away_team="Real Betis",
        home_team_id=9825,
        away_team_id=10125,
        scheduled_kickoff="2026-08-22T15:00:00+00:00",  # 08:30 PM IST Aug 22
        status="FT",
        provider="mock",
        provider_fixture_id="202",
        home_goals=0,
        away_goals=0,
    ))

    fs = FixtureService(custom_provider=mock_prov)
    fixtures, _ = fs.get_todays_matches(date_str="2026-08-22", custom_provider=mock_prov)

    f201 = next(f for f in fixtures if f.fixture_id == 201)
    f202 = next(f for f in fixtures if f.fixture_id == 202)

    assert f201.home_goals == 3
    assert f201.away_goals == 0
    assert f201.actual_outcome == "H"

    assert f202.home_goals == 0
    assert f202.away_goals == 0
    assert f202.actual_outcome == "D"


def test_completed_matches_not_removed_by_prekickoff_buffer():
    """Verify FT completed matches are NEVER excluded by pre-kickoff buffer."""
    ps = PredictionService()

    # Completed match with past kickoff timestamp
    past_kickoff = "2026-08-21T18:45:00+00:00"
    fix_ft = DashboardFixture(
        fixture_id=301,
        competition_id=200,
        competition_name="Ligue 1",
        season_name="2026/2027",
        home_team="Olympique Marseille",
        away_team="Strasbourg",
        home_id=9825,
        away_id=10125,
        scheduled_kickoff=past_kickoff,
        status="FT",
        provider="OddAlerts API",
        home_goals=4,
        away_goals=0,
        actual_outcome="H",
    )

    # Prediction evaluated at current time well after kickoff
    pred = ps.predict_dashboard_fixture(
        fix_ft,
        pre_kickoff_buffer_minutes=15,
        current_time_iso="2026-08-22T12:00:00+00:00",
    )

    assert pred.prediction_allowed is True
    assert pred.actual_outcome == "H"
    assert pred.v4_decision is not None
    assert pred.v4_correct is not None


def test_future_ns_matches_respect_prekickoff_buffer():
    """Verify future NS matches respect the pre-kickoff buffer and do NOT leak future results."""
    ps = PredictionService()

    # Match kicking off at 14:00 UTC
    kickoff = "2026-08-22T14:00:00+00:00"
    fix_ns = DashboardFixture(
        fixture_id=302,
        competition_id=423,
        competition_name="Premier League",
        season_name="2026/2027",
        home_team="Everton",
        away_team="Crystal Palace",
        home_id=10125,
        away_id=9825,
        scheduled_kickoff=kickoff,
        status="NS",
        provider="OddAlerts API",
        home_goals=None,
        away_goals=None,
        actual_outcome=None,
    )

    # 1. 30 min before kickoff (13:30 UTC) -> Allowed, evaluation is None
    pred_early = ps.predict_dashboard_fixture(
        fix_ns,
        pre_kickoff_buffer_minutes=15,
        current_time_iso="2026-08-22T13:30:00+00:00",
    )
    assert pred_early.prediction_allowed is True
    assert pred_early.actual_outcome is None
    assert pred_early.v4_correct is None

    # 2. 5 min before kickoff (13:55 UTC) -> Blocked by 15-min buffer
    pred_late = ps.predict_dashboard_fixture(
        fix_ns,
        pre_kickoff_buffer_minutes=15,
        current_time_iso="2026-08-22T13:55:00+00:00",
    )
    assert pred_late.prediction_allowed is False
    assert "kickoff passed before prediction lock" in pred_late.status_message


def test_live_matches_not_evaluated_as_final():
    """Verify live matches display live status/score but are NOT evaluated as final outcomes."""
    mock_prov = MockTestFixtureProvider()
    mock_prov.add_fixture(UpcomingFixture(
        fixture_id=401,
        league_id=423,
        league_name="Premier League",
        home_team="Chelsea",
        away_team="Tottenham Hotspur",
        home_team_id=10125,
        away_team_id=9825,
        scheduled_kickoff="2026-08-22T14:00:00+00:00",
        status="2H",
        provider="mock",
        provider_fixture_id="401",
        home_goals=1,
        away_goals=1,
    ))

    fs = FixtureService(custom_provider=mock_prov)
    fixtures, _ = fs.get_todays_matches(date_str="2026-08-22", custom_provider=mock_prov)
    fix_live = fixtures[0]

    # Live matches must NOT have final actual_outcome
    assert fix_live.status == "2H"
    assert fix_live.actual_outcome is None


def test_ft_without_score_fails_safely():
    """Verify FT fixture with missing scores returns actual_outcome None safely."""
    mock_prov = MockTestFixtureProvider()
    mock_prov.add_fixture(UpcomingFixture(
        fixture_id=501,
        league_id=423,
        league_name="Premier League",
        home_team="Everton",
        away_team="Crystal Palace",
        home_team_id=10125,
        away_team_id=9825,
        scheduled_kickoff="2026-08-22T14:00:00+00:00",
        status="FT",
        provider="mock",
        provider_fixture_id="501",
        home_goals=None,
        away_goals=None,
    ))

    fs = FixtureService(custom_provider=mock_prov)
    fixtures, _ = fs.get_todays_matches(date_str="2026-08-22", custom_provider=mock_prov)
    assert fixtures[0].actual_outcome is None


def test_fixture_deduplication():
    """Verify duplicate fixture records from provider are deduplicated by fixture_id."""
    mock_prov = MockTestFixtureProvider()
    for _ in range(3):
        mock_prov.add_fixture(UpcomingFixture(
            fixture_id=601,
            league_id=423,
            league_name="Premier League",
            home_team="Everton",
            away_team="Crystal Palace",
            home_team_id=10125,
            away_team_id=9825,
            scheduled_kickoff="2026-08-22T14:00:00+00:00",
            status="NS",
            provider="mock",
            provider_fixture_id="601",
        ))

    fs = FixtureService(custom_provider=mock_prov)
    fixtures, _ = fs.get_todays_matches(date_str="2026-08-22", custom_provider=mock_prov)
    assert len(fixtures) == 1


def test_model_evaluation_correctness():
    """Verify V4, V4.6, and Hist H predictions are accurately checked against reconciled outcomes."""
    ps = PredictionService()

    # Match with known Home Win outcome
    fix = DashboardFixture(
        fixture_id=701,
        competition_id=423,
        competition_name="Premier League",
        season_name="2026/2027",
        home_team="Arsenal",
        away_team="Coventry City",
        home_id=9825,
        away_id=10125,
        scheduled_kickoff="2026-08-21T19:00:00+00:00",
        status="FT",
        provider="OddAlerts API",
        home_goals=3,
        away_goals=0,
        actual_outcome="H",
    )

    pred = ps.predict_dashboard_fixture(fix, pre_kickoff_buffer_minutes=15)
    assert pred.prediction_allowed is True
    assert pred.actual_outcome == "H"
    assert pred.v4_correct == (pred.v4_decision == "H")
    assert pred.v4_6_correct == (pred.v4_6_decision == "H")
    assert pred.hist_h_correct == (pred.hist_h_decision == "H")


def test_production_asset_integrity():
    """Verify all 20 protected repository baseline assets remain 100% bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}: expected {exp}, got {act}"
