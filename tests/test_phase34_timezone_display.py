"""Phase 34 — Timezone and 12-Hour Display Test Suite.

Verifies:
1. Strict timezone conversion from UTC to Asia/Kolkata (IST / Chennai).
2. Exact 12-hour format outputs:
   - 11:30 UTC -> 05:00 PM IST
   - 14:00 UTC -> 07:30 PM IST
   - 16:30 UTC -> 10:00 PM IST
   - 19:30 UTC -> 01:00 AM IST (next calendar day)
3. Local date rollover (e.g., 2026-08-22 19:30 UTC rolls over to 2026-08-23 Chennai date).
4. Immutability of underlying UTC timestamp in fixture objects and database.
5. Causal pre-kickoff validation and buffer timing remain strictly UTC-based.
6. Matchday discovery filtering correctly applies Chennai local date boundaries.
7. Verification of all 20 protected repository baseline assets.
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
from dashboard.time_utils import (
    IST_ZONE,
    format_kickoff_datetime_ist,
    format_kickoff_ist,
    is_kickoff_on_chennai_date,
    parse_to_utc_datetime,
    to_chennai_date,
    to_ist_datetime,
)
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


def test_required_time_conversions():
    """Verify exact required 12-hour IST conversions from prompt."""
    test_cases = [
        ("2026-08-22T11:30:00Z", "05:00 PM IST", "05:00 PM"),
        ("2026-08-22T14:00:00Z", "07:30 PM IST", "07:30 PM"),
        ("2026-08-22T16:30:00Z", "10:00 PM IST", "10:00 PM"),
        ("2026-08-22T19:30:00Z", "01:00 AM IST", "01:00 AM"),
    ]

    for utc_ts, exp_with_suffix, exp_no_suffix in test_cases:
        assert format_kickoff_ist(utc_ts, include_suffix=True) == exp_with_suffix
        assert format_kickoff_ist(utc_ts, include_suffix=False) == exp_no_suffix


def test_date_rollover_behavior():
    """Verify 19:30 UTC rolls over to the next calendar day in Chennai (IST)."""
    # 2026-08-22 19:30 UTC -> 2026-08-23 01:00 IST
    utc_ts = "2026-08-22T19:30:00Z"
    assert to_chennai_date(utc_ts) == "2026-08-23"
    assert is_kickoff_on_chennai_date(utc_ts, "2026-08-23") is True
    assert is_kickoff_on_chennai_date(utc_ts, "2026-08-22") is False

    # 2026-08-22 18:29:59 UTC -> 2026-08-22 23:59:59 IST (Same day in Chennai)
    utc_same_day = "2026-08-22T18:29:59Z"
    assert to_chennai_date(utc_same_day) == "2026-08-22"
    assert is_kickoff_on_chennai_date(utc_same_day, "2026-08-22") is True


def test_zoneinfo_awareness():
    """Verify ZoneInfo('Asia/Kolkata') is strictly used without offset arithmetic."""
    dt_ist = to_ist_datetime("2026-08-22T11:30:00Z")
    assert getattr(dt_ist.tzinfo, "key", str(dt_ist.tzinfo)) == "Asia/Kolkata"
    assert dt_ist.tzname() in ("IST", "Asia/Kolkata")
    assert dt_ist.hour == 17
    assert dt_ist.minute == 0


def test_kickoff_datetime_full_format():
    """Verify format_kickoff_datetime_ist produces clean human-readable date + time."""
    utc_ts = "2026-08-22T11:30:00Z"
    formatted = format_kickoff_datetime_ist(utc_ts)
    assert formatted == "22 Aug 2026, 05:00 PM IST"

    rollover_ts = "2026-08-22T19:30:00Z"
    formatted_rollover = format_kickoff_datetime_ist(rollover_ts)
    assert formatted_rollover == "23 Aug 2026, 01:00 AM IST"


def test_raw_utc_timestamp_immutability():
    """Verify raw UTC timestamp remains unchanged internally and in fixture objects."""
    raw_utc = "2026-08-22T14:00:00+00:00"
    fix = DashboardFixture(
        fixture_id=12345,
        competition_id=423,
        competition_name="Premier League",
        season_name="2026/2027",
        home_team="Everton",
        away_team="Crystal Palace",
        home_id=10125,
        away_id=9825,
        scheduled_kickoff=raw_utc,
        status="NS",
        provider="OddAlerts API",
    )

    # Calling display helpers should NOT mutate fix.scheduled_kickoff
    disp_time = format_kickoff_ist(fix.scheduled_kickoff)
    assert disp_time == "07:30 PM IST"
    assert fix.scheduled_kickoff == raw_utc


def test_prediction_timing_cutoff_remains_utc():
    """Verify pre-kickoff causality check correctly evaluates using UTC timestamps."""
    svc = PredictionService()

    # Match at 14:00 UTC (19:30 IST)
    fix = DashboardFixture(
        fixture_id=8888,
        competition_id=423,
        competition_name="Premier League",
        season_name="2026/2027",
        home_team="Everton",
        away_team="Crystal Palace",
        home_id=10125,
        away_id=9825,
        scheduled_kickoff="2026-08-22T14:00:00+00:00",
        status="NS",
        provider="OddAlerts API",
    )

    # 1. Prediction 30 minutes before kickoff (13:30 UTC / 19:00 IST) -> Allowed
    pred_ok = svc.predict_dashboard_fixture(
        fix,
        pre_kickoff_buffer_minutes=15,
        current_time_iso="2026-08-22T13:30:00+00:00",
    )
    assert pred_ok.prediction_allowed is True

    # 2. Prediction 10 minutes before kickoff (13:50 UTC / 19:20 IST) -> Blocked by 15-min buffer
    pred_late = svc.predict_dashboard_fixture(
        fix,
        pre_kickoff_buffer_minutes=15,
        current_time_iso="2026-08-22T13:50:00+00:00",
    )
    assert pred_late.prediction_allowed is False
    assert "kickoff passed before prediction lock" in pred_late.status_message


def test_fixture_service_filtering_by_chennai_date():
    """Verify FixtureService filters matches based on local Chennai date."""
    mock_prov = MockTestFixtureProvider()
    # Match 1: 2026-08-22 14:00 UTC -> 2026-08-22 19:30 IST (Aug 22 in Chennai)
    mock_prov.add_fixture(UpcomingFixture(
        fixture_id=101,
        league_id=423,
        league_name="Premier League",
        home_team="Everton",
        away_team="Crystal Palace",
        home_team_id=10125,
        away_team_id=9825,
        scheduled_kickoff="2026-08-22T14:00:00+00:00",
        status="NS",
        provider="mock",
        provider_fixture_id="101",
    ))
    # Match 2: 2026-08-22 19:30 UTC -> 2026-08-23 01:00 IST (Aug 23 in Chennai)
    mock_prov.add_fixture(UpcomingFixture(
        fixture_id=102,
        league_id=423,
        league_name="Premier League",
        home_team="Arsenal",
        away_team="Chelsea",
        home_team_id=9825,
        away_team_id=10125,
        scheduled_kickoff="2026-08-22T19:30:00+00:00",
        status="NS",
        provider="mock",
        provider_fixture_id="102",
    ))

    fs = FixtureService(custom_provider=mock_prov)

    # Query for Chennai date 2026-08-22
    f_aug22, _ = fs.get_todays_matches(date_str="2026-08-22", custom_provider=mock_prov)
    f_ids_22 = [f.fixture_id for f in f_aug22]
    assert 101 in f_ids_22

    # Query for Chennai date 2026-08-23
    f_aug23, _ = fs.get_todays_matches(date_str="2026-08-23", custom_provider=mock_prov)
    f_ids_23 = [f.fixture_id for f in f_aug23]
    assert 102 in f_ids_23


def test_production_asset_integrity():
    """Verify all 20 protected repository baseline assets remain 100% bit-identical."""
    for rel, exp in PINNED_20.items():
        act = hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        assert act == exp, f"Integrity failure on {rel}: expected {exp}, got {act}"
