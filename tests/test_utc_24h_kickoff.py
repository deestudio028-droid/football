"""Test suite for UTC 24-Hour Kickoff Display and Live API Fixture Fetching."""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.fixture_service import FixtureService
from dashboard.prediction_service import PredictionService
from dashboard.time_utils import (
    format_kickoff_datetime_utc,
    format_kickoff_utc,
    is_kickoff_on_utc_date,
    parse_to_utc_datetime,
    to_utc_date,
)

V4_PATH = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
V4_1_PATH = PROJECT_ROOT / "data/models/v4_1_prospective_candidate_2025_26.pkl"

EXP_V4_MD5 = "06841f0c03c8597b2b8cd8f8ab064864"
EXP_V4_1_MD5 = "145f918d933eb343c0f63ca342b10289"


def test_1_v4_0_md5_immutability():
    """Verify V4.0 production baseline MD5 hash is bit-identical."""
    assert V4_PATH.exists()
    act = hashlib.md5(V4_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_MD5, f"V4.0 MD5 mutated: expected {EXP_V4_MD5}, got {act}"


def test_2_v4_1_md5_immutability():
    """Verify V4.1 candidate MD5 hash is bit-identical."""
    assert V4_1_PATH.exists()
    act = hashlib.md5(V4_1_PATH.read_bytes()).hexdigest()
    assert act == EXP_V4_1_MD5, f"V4.1 MD5 mutated: expected {EXP_V4_1_MD5}, got {act}"


def test_3_utc_24h_formatting():
    """Verify format_kickoff_utc formats timestamps in 24-hour UTC format."""
    test_cases = [
        ("2026-08-27T18:30:00Z", "18:30 UTC", "18:30"),
        ("2026-08-27T19:00:00Z", "19:00 UTC", "19:00"),
        ("2026-08-27T00:05:00Z", "00:05 UTC", "00:05"),
        ("2026-08-27T12:00:00Z", "12:00 UTC", "12:00"),
    ]
    for ts, exp_suff, exp_no_suff in test_cases:
        assert format_kickoff_utc(ts, include_suffix=True) == exp_suff
        assert format_kickoff_utc(ts, include_suffix=False) == exp_no_suff


def test_4_utc_datetime_full_format():
    """Verify format_kickoff_datetime_utc formats full date and 24h UTC time."""
    assert format_kickoff_datetime_utc("2026-08-27T18:30:00Z") == "27 Aug 2026, 18:30 UTC"
    assert format_kickoff_datetime_utc("2026-08-27T19:00:00Z") == "27 Aug 2026, 19:00 UTC"


def test_5_to_utc_date():
    """Verify to_utc_date extracts correct UTC date string."""
    assert to_utc_date("2026-08-27T18:30:00Z") == "2026-08-27"
    assert to_utc_date("2026-08-27T23:59:59Z") == "2026-08-27"
    assert is_kickoff_on_utc_date("2026-08-27T18:30:00Z", "2026-08-27") is True


def test_6_api_fetch_matches_today():
    """Verify OddAlerts API live fetch discovers matches on 2026-08-27."""
    fs = FixtureService()
    fixtures, meta = fs.get_todays_matches(date_str="2026-08-27", provider_name="oddalerts")

    assert meta["provider_display"] == "OddAlerts API"
    assert meta["status"] == "SUCCESS"
    assert len(fixtures) >= 2

    match_teams = [(f.home_team, f.away_team) for f in fixtures]
    assert any("Celta" in h and "Osasuna" in a for h, a in match_teams)
    assert any("Barcelona" in h and "Athletic" in a for h, a in match_teams)

    # Verify all kickoffs format cleanly in 24h UTC
    for f in fixtures:
        formatted = format_kickoff_utc(f.scheduled_kickoff, include_suffix=False)
        assert len(formatted) == 5
        assert ":" in formatted
        hour, minute = [int(x) for x in formatted.split(":")]
        assert 0 <= hour <= 23
        assert 0 <= minute <= 59


def test_7_prediction_service_on_api_matches():
    """Verify PredictionService generates valid predictions on live API fixtures."""
    fs = FixtureService()
    ps = PredictionService()
    fixtures, _ = fs.get_todays_matches(date_str="2026-08-27", provider_name="oddalerts")

    for fix in fixtures:
        pred = ps.predict_dashboard_fixture(fix, pre_kickoff_buffer_minutes=15)
        assert pred.prediction_allowed is True
        assert pred.prediction_model == "V4.0 Production"
        assert pred.production_decision in ("H", "D", "A")
        assert np.isclose(pred.production_probs["H"] + pred.production_probs["D"] + pred.production_probs["A"], 1.0, atol=1e-2)


def test_8_dashboard_schema_and_utc_caption():
    """Verify dashboard match table preserves 15-column schema and declares UTC 24h format."""
    app_text = (PROJECT_ROOT / "src/dashboard/app.py").read_text(encoding="utf-8")
    assert "format_kickoff_utc(fix.scheduled_kickoff, include_suffix=False)" in app_text
    assert "st.caption(\"ℹ️ All kickoff times shown in UTC (24-hour format)\")" in app_text

    for col in [
        "\"Kickoff (UTC)\":",
        "\"Date (UTC)\":",
        "\"League\":",
        "\"Home Team\":",
        "\"Away Team\":",
        "\"Score\":",
        "\"P(H)\":",
        "\"P(D)\":",
        "\"P(A)\":",
        "\"Model Prediction\":",
        "\"Selected\":",
        "\"Eval\":",
        "\"Draw Risk\":",
        "\"Actual Game Result\":",
        "\"Goal Prediction\":",
        "\"Actual Goal Result\":",
    ]:
        assert col in app_text
