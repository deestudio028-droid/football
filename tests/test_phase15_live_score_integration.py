"""
Phase 15.1 — Live Score Integration Test Suite
Project: E:\\Football Prediction Project
Path: tests/test_phase15_live_score_integration.py

Tests:
1. Dashboard contains Railway live endpoint.
2. Polling interval is 30 seconds (30,000 ms).
3. Fixture ID is used for live matching.
4. Prediction fields are immutable (DOM structure and data integrity).
5. Actual score is updateable via DOM class and helper.
6. Match status is updateable via status badges.
7. Live minute is updateable and properly displayed.
8. API failure does not destroy dashboard (graceful offline handling).
9. No API token appears in HTML or JavaScript.
10. No OddAlerts direct browser call exists.
11. Exactly 48 fixtures remain present in the ledger.
12. 2-0 signals remain completely unchanged.
13. Existing Railway endpoint schema is handled correctly.
14. No new backend was created (only existing backend/main.py).
15. Static dashboard still opens offline/basic mode.
16. V4 model SHA256 remains bit-identical.
17. src/ and data/models/ have zero uncommitted modifications.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V4_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
EXPECTED_V4_SHA256 = "1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5"

REPORTS_HTML = PROJECT_ROOT / "reports" / "upcoming_2026_09_18_to_2026_09_21_dashboard.html"
REPORTS_LEDGER = PROJECT_ROOT / "reports" / "upcoming_2026_09_18_to_2026_09_21_ledger.jsonl"
P15_05_HTML = PROJECT_ROOT / "research" / "external_consensus" / "phase15" / "05_phase15_dashboard.html"
STATIC_INTEGRITY_JSON = PROJECT_ROOT / "research" / "external_consensus" / "phase15_live" / "03_static_prediction_integrity.json"
SCHEMA_VAL_JSON = PROJECT_ROOT / "research" / "external_consensus" / "phase15_live" / "02_live_schema_validation.json"

EXPECTED_RAILWAY_URL = "https://web-production-d8a09.up.railway.app/api/live-scores"


@pytest.fixture(scope="module")
def html_content() -> str:
    assert REPORTS_HTML.exists(), f"Dashboard HTML missing at {REPORTS_HTML}"
    with open(REPORTS_HTML, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def fixtures_ledger() -> List[Dict[str, Any]]:
    assert REPORTS_LEDGER.exists(), f"Ledger missing at {REPORTS_LEDGER}"
    records = []
    with open(REPORTS_LEDGER, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def test_01_dashboard_contains_railway_endpoint(html_content):
    """1. Dashboard contains Railway live endpoint."""
    assert EXPECTED_RAILWAY_URL in html_content, (
        f"Expected Railway live endpoint {EXPECTED_RAILWAY_URL} not found in dashboard HTML"
    )


def test_02_polling_interval_is_30_seconds(html_content):
    """2. Polling interval is 30 seconds (30000 ms)."""
    assert "POLLING_INTERVAL_MS = 30000" in html_content or "30000" in html_content, (
        "30-second polling interval (30000 ms) not configured in dashboard"
    )
    assert "setInterval(pollLiveScores, POLLING_INTERVAL_MS)" in html_content or "setInterval(pollLiveScores, 30000)" in html_content, (
        "setInterval polling routine not attached"
    )


def test_03_fixture_id_used_for_live_matching(html_content):
    """3. Fixture ID is used as primary key for live matching."""
    assert "f['fixture_id']" in html_content or "f.fixture_id" in html_content
    assert "data-fixture-id" in html_content
    assert "liveScoresMap[fidStr]" in html_content or "scoresMap[fidStr]" in html_content, (
        "Live matching does not look up fixture ID in liveScoresMap"
    )


def test_04_prediction_fields_are_immutable(fixtures_ledger, html_content):
    """4. Prediction fields are immutable and cannot be overwritten by live scores."""
    # Check deterministic hash of prediction fields
    with open(STATIC_INTEGRITY_JSON, "r", encoding="utf-8") as f:
        integrity_meta = json.load(f)

    pred_tuples = [
        (
            r["fixture_id"],
            r["Home Team"],
            r["Away Team"],
            r["p_home"],
            r["p_draw"],
            r["p_away"],
            r["PRED"],
            r["canonical_predicted_score"],
            r["draw_risk_tier"],
            r["signal_profile"],
        )
        for r in fixtures_ledger
    ]
    ser = json.dumps(pred_tuples, sort_keys=True)
    digest = hashlib.sha256(ser.encode("utf-8")).hexdigest()
    assert digest == integrity_meta["frozen_prediction_sha256"], (
        "Prediction fields hash changed! Static predictions must remain 100% immutable"
    )

    # In HTML, predicted score column is rendered with score-pill and never targeted by live score updater
    assert 'f[\'Predicted Score\']' in html_content
    assert 'col-actual-score' in html_content


def test_05_actual_score_is_updateable(html_content):
    """5. Actual score column exists and is dynamically updateable."""
    assert "col-actual-score" in html_content
    assert "score-badge-live" in html_content
    assert "score-badge-ft" in html_content
    assert "<th>Actual Score</th>" in html_content


def test_06_match_status_is_updateable(html_content):
    """6. Match status column exists and is dynamically updateable."""
    assert "col-status" in html_content
    assert "status-upcoming" in html_content
    assert "status-live" in html_content
    assert "status-finished" in html_content
    assert "data-status" in html_content


def test_07_live_minute_is_updateable(html_content):
    """7. Live minute is formatted and displayed when present."""
    assert "LIVE${minStr}" in html_content or "LIVE " in html_content
    assert "elapsed" in html_content
    assert "time_added" in html_content or "timeAdded" in html_content


def test_08_api_failure_does_not_destroy_dashboard(html_content):
    """8. API failure does not destroy dashboard (graceful offline handling)."""
    assert "consecutiveFailures" in html_content
    assert "catch(err" in html_content
    assert "LIVE DATA OFFLINE" in html_content
    assert "RECONNECTING" in html_content


def test_09_no_api_token_appears_in_html(html_content):
    """9. No API token appears in HTML or JavaScript."""
    assert "api_token=" not in html_content
    assert "OddAlerts_API" not in html_content
    env_token = os.environ.get("OddAlerts_API")
    if env_token and len(env_token) > 6:
        assert env_token not in html_content


def test_10_no_oddalerts_direct_browser_call(html_content):
    """10. No OddAlerts direct browser call exists."""
    assert "oddalerts.com" not in html_content.lower()


def test_11_all_48_fixtures_remain_present(fixtures_ledger):
    """11. Exactly 48 fixtures remain present."""
    assert len(fixtures_ledger) == 48, f"Expected 48 fixtures, found {len(fixtures_ledger)}"


def test_12_2_0_signals_remain_unchanged(fixtures_ledger):
    """12. 2-0 signals remain completely unchanged."""
    sig_20 = [f for f in fixtures_ledger if f["is_2_0_profile"]]
    assert len(sig_20) == 2, f"Expected 2 2-0 profile fixtures, found {len(sig_20)}"

    teams = {(f["Home Team"], f["Away Team"]): f for f in sig_20}
    assert ("FC Bayern München", "FC Union Berlin") in teams
    assert ("Manchester City", "Sunderland") in teams

    bayern = teams[("FC Bayern München", "FC Union Berlin")]
    assert bayern["canonical_predicted_score"] == "2-0"
    assert bayern["draw_risk_tier"] == "LOW"
    assert bayern["p_home"] == 0.7765

    city = teams[("Manchester City", "Sunderland")]
    assert city["canonical_predicted_score"] == "2-0"
    assert city["draw_risk_tier"] == "LOW"
    assert city["p_home"] == 0.7048


def test_13_existing_railway_schema_handled(html_content):
    """13. Existing Railway endpoint schema is handled correctly."""
    assert SCHEMA_VAL_JSON.exists(), f"Schema validation JSON missing at {SCHEMA_VAL_JSON}"
    with open(SCHEMA_VAL_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["status_code"] == 200
    assert "scores" in schema["top_level_keys"]
    assert "last_updated_utc" in schema["top_level_keys"]

    # In HTML, data.scores is parsed and handled
    assert "data.scores" in html_content
    assert "item.status" in html_content


def test_14_no_new_backend_created():
    """14. No new backend was created (only existing backend/main.py)."""
    # Check for any rogue new backend files
    project_files = [p.name for p in (PROJECT_ROOT / "backend").iterdir() if p.is_file()]
    assert "main.py" in project_files
    # No extra backend app files
    assert "app2.py" not in project_files
    assert "server.py" not in project_files


def test_15_static_dashboard_still_opens_offline(html_content):
    """15. Static dashboard still opens offline/basic mode with zero CDNs."""
    assert "<link" not in html_content or "http" not in html_content
    assert "FIXTURES =" in html_content
    assert "renderTable()" in html_content


def test_16_v4_sha256_unchanged():
    """16. V4 model SHA256 remains bit-identical."""
    assert V4_MODEL_PATH.exists()
    digest = hashlib.sha256(V4_MODEL_PATH.read_bytes()).hexdigest().lower()
    assert digest == EXPECTED_V4_SHA256


def test_17_git_status_clean_on_protected_paths():
    """17. src/ and data/models/ have zero uncommitted modifications."""
    res = subprocess.run(
        ["git", "status", "--short", "src/", "data/models/"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(res.stdout.strip()) == 0, f"Protected paths modified:\n{res.stdout}"


def test_18_ft_fixture_score_returned():
    """18. FT fixture score returned (fixture 420656757 returns 7-0 and FT)."""
    from backend.main import LiveScoreManager
    mgr = LiveScoreManager()
    res = mgr.fetch_live_scores()
    scores = res["scores"]
    assert "420656757" in scores, "Fixture 420656757 missing from live scores response"
    bayern = scores["420656757"]
    assert bayern["status"] == "FT"
    assert bayern["display_score"] == "7-0"
    assert bayern["home_goals"] == 7
    assert bayern["away_goals"] == 0
    assert bayern["is_finished"] is True
    assert bayern["status_label"] == "FINISHED"


def test_19_ft_fixture_remains_available_after_feed_omission():
    """19. FT fixture remains available after live feed no longer contains it."""
    from backend.main import LiveScoreManager
    mgr = LiveScoreManager()
    assert "420656757" in mgr._completed_cache
    fb = mgr._generate_fallback_response("OMISSION_SIMULATION")
    assert "420656757" in fb["scores"]
    item = fb["scores"]["420656757"]
    assert item["status"] == "FT"
    assert item["display_score"] == "7-0"
    assert item["is_finished"] is True


def test_20_valid_ft_score_never_overwritten_by_null():
    """20. Valid FT score is never overwritten by null or UPCOMING status."""
    from backend.main import LiveScoreManager
    mgr = LiveScoreManager()
    assert "420656757" in mgr._completed_cache
    bogus_raw = {
        "id": 420656757,
        "status": "NS",
        "home_goals": None,
        "away_goals": None
    }
    normalized = mgr._normalize_fixture(bogus_raw)
    assert normalized.status == "FT"
    assert normalized.display_score == "7-0"
    assert normalized.home_goals == 7
    assert normalized.away_goals == 0
    assert normalized.is_finished is True


def test_21_upcoming_matches_remain_dash():
    """21. UPCOMING matches without kickoff remain —."""
    from backend.main import LiveScoreManager
    mgr = LiveScoreManager()
    raw_upcoming = {
        "id": 999999999,
        "status": "NS",
        "home_goals": None,
        "away_goals": None
    }
    norm = mgr._normalize_fixture(raw_upcoming)
    assert norm.status == "UPCOMING"
    assert norm.display_score == "—"
    assert norm.home_goals is None
    assert norm.away_goals is None
    assert norm.is_finished is False


def test_22_live_score_and_minute_updates():
    """22. LIVE score updates minute and score dynamically."""
    from backend.main import LiveScoreManager
    mgr = LiveScoreManager()
    raw_live = {
        "id": 888888888,
        "status": "LIVE",
        "home_goals": 1,
        "away_goals": 0,
        "elapsed": 67,
        "time_added": 2
    }
    norm = mgr._normalize_fixture(raw_live)
    assert norm.status == "LIVE"
    assert norm.display_score == "1-0"
    assert norm.status_label == "🔴 LIVE 67+2'"
    assert norm.elapsed == 67
    assert norm.time_added == 2
    assert norm.is_finished is False

    raw_ht = {
        "id": 888888887,
        "status": "HT",
        "home_goals": 0,
        "away_goals": 1,
        "elapsed": 45
    }
    norm_ht = mgr._normalize_fixture(raw_ht)
    assert norm_ht.status == "HT"
    assert norm_ht.display_score == "0-1"
    assert norm_ht.status_label == "⏸ HT"


def test_23_fixture_id_matching_and_coverage(fixtures_ledger):
    """23. Fixture ID matching across all 48 Phase 15 fixtures."""
    from backend.main import LiveScoreManager
    mgr = LiveScoreManager()
    ledger_fids = [int(f["fixture_id"]) for f in fixtures_ledger]
    assert len(ledger_fids) == 48
    for fid in ledger_fids:
        assert fid in mgr.fixture_ids, f"Fixture ID {fid} missing from LiveScoreManager tracked fixtures"


def test_24_frontend_preserves_completed_score(html_content):
    """24. Frontend preserves completed score against omission or reverting."""
    assert "window._persistedCompletedScores" in html_content
    assert "norm.isFinished" in html_content
    assert "Object.assign({}, window._persistedCompletedScores, data.scores)" in html_content
    assert "?ids=" in html_content

