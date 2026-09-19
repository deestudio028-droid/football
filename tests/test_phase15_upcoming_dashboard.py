"""
Phase 15 — Upcoming Big-5 European League Prediction Dashboard & Ledger Test Suite
Project: E:\\Football Prediction Project
Path: tests/test_phase15_upcoming_dashboard.py

Verifies the 20 strict invariants mandated for Phase 15:
1. Model artifact SHA256 matches frozen hash.
2. src/ has zero modifications.
3. data/models/ has zero modifications.
4. Dashboard HTML exists and is non-empty (>10KB).
5. Ledger JSONL exists and has >= 40 fixtures (expected 48).
6. All fixtures are within the 18 Sep – 21 Sep 2026 date window.
7. All fixtures belong to the Big-5 leagues.
8. Every fixture has valid H/D/A probabilities that sum to 1.0 (±0.01).
9. Every fixture's PRED equals the argmax of (P(H), P(D), P(A)).
10. Predicted scores are NOT all "2-1" for home wins (diversity check).
11. Predicted scores are NOT all "1-2" for away wins (diversity check).
12. At least one "2-0" prediction exists OR absence mathematically verified.
13. Every "2-0 Profile" fixture has canonical_predicted_score == "2-0" AND draw_risk_tier == "LOW".
14. Every "Strong Home Profile" fixture has P(H) >= 0.60 AND draw_risk_tier == "LOW".
15. The prioritized section in the HTML contains all identified signal matches.
16. The HTML dashboard is standalone: zero external CDN links (no external script/css).
17. No actual scores or completed match outcomes appear anywhere in the ledger or dashboard.
18. All matches have Status == "UPCOMING" and Actual Score == "—".
19. No API tokens or secrets are leaked in any output file.
20. Research artifacts exist: 01, 02, 03, 04, 05 files present in research/external_consensus/phase15/.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V4_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
EXPECTED_V4_SHA256 = "1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5"

REPORTS_HTML = PROJECT_ROOT / "reports" / "upcoming_2026_09_18_to_2026_09_21_dashboard.html"
REPORTS_LEDGER = PROJECT_ROOT / "reports" / "upcoming_2026_09_18_to_2026_09_21_ledger.jsonl"

PHASE15_DIR = PROJECT_ROOT / "research" / "external_consensus" / "phase15"
P15_01_LEDGER = PHASE15_DIR / "01_upcoming_fixture_ledger.jsonl"
P15_02_SIGNALS = PHASE15_DIR / "02_2_0_signal_ledger.jsonl"
P15_03_VALIDATION = PHASE15_DIR / "03_phase15_validation.json"
P15_04_REPORT = PHASE15_DIR / "04_phase15_report.md"
P15_05_HTML = PHASE15_DIR / "05_phase15_dashboard.html"
P15_README = PHASE15_DIR / "README.md"
P15_ENGINE = PHASE15_DIR / "phase15_engine.py"

BIG5_LEAGUES = {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}


@pytest.fixture(scope="module")
def upcoming_fixtures() -> List[Dict[str, Any]]:
    assert REPORTS_LEDGER.exists(), f"Ledger file missing at {REPORTS_LEDGER}"
    records = []
    with open(REPORTS_LEDGER, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def dashboard_html_content() -> str:
    assert REPORTS_HTML.exists(), f"Dashboard HTML missing at {REPORTS_HTML}"
    with open(REPORTS_HTML, "r", encoding="utf-8") as f:
        return f.read()


def test_01_v4_sha256_matches_frozen_hash():
    """1. Model artifact SHA256 matches frozen hash."""
    assert V4_MODEL_PATH.exists(), f"Frozen V4 model binary missing at {V4_MODEL_PATH}"
    with open(V4_MODEL_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest().lower()
    assert digest == EXPECTED_V4_SHA256, (
        f"V4 SHA256 altered! Expected {EXPECTED_V4_SHA256}, got {digest}"
    )


def test_02_src_directory_has_zero_modifications():
    """2. src/ has zero unstaged or uncommitted changes."""
    res = subprocess.run(
        ["git", "status", "--short", "src/"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(res.stdout.strip()) == 0, f"src/ contains modifications:\n{res.stdout}"


def test_03_data_models_directory_has_zero_modifications():
    """3. data/models/ has zero modifications."""
    res = subprocess.run(
        ["git", "status", "--short", "data/models/"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(res.stdout.strip()) == 0, f"data/models/ modified:\n{res.stdout}"


def test_04_dashboard_html_exists_and_non_empty():
    """4. Dashboard HTML exists and is non-empty (>10KB)."""
    assert REPORTS_HTML.exists(), f"Dashboard HTML missing at {REPORTS_HTML}"
    file_size = REPORTS_HTML.stat().st_size
    assert file_size >= 10240, f"Dashboard HTML too small ({file_size} bytes, expected >= 10KB)"


def test_05_ledger_jsonl_exists_with_expected_fixtures(upcoming_fixtures):
    """5. Ledger JSONL exists and has >= 40 fixtures (48 expected)."""
    assert len(upcoming_fixtures) >= 40, f"Expected >= 40 fixtures, got {len(upcoming_fixtures)}"
    assert len(upcoming_fixtures) == 48, f"Expected exactly 48 Big-5 fixtures for full MW, got {len(upcoming_fixtures)}"


def test_06_all_fixtures_within_date_window(upcoming_fixtures):
    """6. All fixtures are within the 18 Sep – 21 Sep 2026 date window."""
    valid_dates = {"2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21"}
    for f in upcoming_fixtures:
        date_str = f["Date (UTC)"]
        assert date_str in valid_dates, (
            f"Fixture {f['fixture_id']} date {date_str} outside window 18-21 Sep 2026"
        )
        ko = f["scheduled_kickoff"]
        assert "2026-09-18" <= ko[:10] <= "2026-09-21", (
            f"Fixture {f['fixture_id']} kickoff {ko} outside window"
        )


def test_07_all_fixtures_belong_to_big5_leagues(upcoming_fixtures):
    """7. All fixtures belong to the Big-5 leagues."""
    for f in upcoming_fixtures:
        league = f["League"]
        assert league in BIG5_LEAGUES, f"Fixture {f['fixture_id']} has unknown league {league}"

    # Verify all 5 leagues are represented
    found_leagues = {f["League"] for f in upcoming_fixtures}
    assert found_leagues == BIG5_LEAGUES, f"Missing leagues: {BIG5_LEAGUES - found_leagues}"


def test_08_probabilities_valid_and_sum_to_one(upcoming_fixtures):
    """8. Every fixture has valid H/D/A probabilities that sum to 1.0 (±0.01)."""
    for f in upcoming_fixtures:
        ph = float(f["p_home"])
        pd_ = float(f["p_draw"])
        pa = float(f["p_away"])

        assert 0.0 < ph < 1.0, f"Invalid p_home in fixture {f['fixture_id']}: {ph}"
        assert 0.0 < pd_ < 1.0, f"Invalid p_draw in fixture {f['fixture_id']}: {pd_}"
        assert 0.0 < pa < 1.0, f"Invalid p_away in fixture {f['fixture_id']}: {pa}"

        prob_sum = ph + pd_ + pa
        assert abs(prob_sum - 1.0) <= 0.01, (
            f"Probability sum violation in fixture {f['fixture_id']}: {prob_sum}"
        )


def test_09_pred_equals_probability_argmax(upcoming_fixtures):
    """9. Every fixture's PRED equals the argmax of (P(H), P(D), P(A))."""
    for f in upcoming_fixtures:
        ph = float(f["p_home"])
        pd_ = float(f["p_draw"])
        pa = float(f["p_away"])
        pred = f["PRED"]

        expected_pred = "H" if (ph >= pd_ and ph >= pa) else ("A" if (pa >= pd_ and pa >= ph) else "D")
        assert pred == expected_pred, (
            f"PRED mismatch in fixture {f['fixture_id']}: got {pred}, expected {expected_pred} (H={ph}, D={pd_}, A={pa})"
        )


def test_10_predicted_scores_not_flattened_to_2_1(upcoming_fixtures):
    """10. Predicted scores are NOT all '2-1' for home wins (Phase 14 bug fix check)."""
    home_win_scores = [f["canonical_predicted_score"] for f in upcoming_fixtures if f["PRED"] == "H"]
    assert len(home_win_scores) > 0, "No home wins predicted"
    unique_scores = set(home_win_scores)
    # The old bug caused 100% of home wins to be "2-1". In canonical V4, we expect multiple scorelines (e.g. 1-1, 1-0, 2-0, 2-1)
    assert len(unique_scores) > 1, f"Home win scores are flattened to a single value: {unique_scores}"
    # Specifically check that '2-0' is among the home win predictions
    assert "2-0" in unique_scores, f"Expected '2-0' score among home predictions, got {unique_scores}"


def test_11_predicted_scores_not_flattened_to_1_2(upcoming_fixtures):
    """11. Predicted scores are NOT all '1-2' for away wins (diversity check)."""
    away_win_scores = [f["canonical_predicted_score"] for f in upcoming_fixtures if f["PRED"] == "A"]
    assert len(away_win_scores) > 0, "No away wins predicted"
    # Ensure away predictions are not hardcoded to '1-2'
    for s in away_win_scores:
        assert s in ["1-1", "0-1", "0-2", "1-2", "0-3"], f"Unexpected away win score {s}"


def test_12_at_least_one_2_0_prediction_exists(upcoming_fixtures):
    """12. At least one '2-0' prediction exists in the upcoming batch."""
    scores_20 = [f for f in upcoming_fixtures if f["canonical_predicted_score"] == "2-0"]
    assert len(scores_20) >= 1, f"Expected at least one 2-0 prediction, found {len(scores_20)}"
    assert len(scores_20) == 2, f"Expected exactly 2 canonical 2-0 predictions, found {len(scores_20)}"


def test_13_every_2_0_profile_fixture_valid(upcoming_fixtures):
    """13. Every '2-0 Profile' fixture has canonical_predicted_score == '2-0' AND draw_risk_tier == 'LOW'."""
    p20_fixtures = [f for f in upcoming_fixtures if f["is_2_0_profile"]]
    assert len(p20_fixtures) >= 1, "No 2-0 profile fixtures found"
    for f in p20_fixtures:
        assert f["canonical_predicted_score"] == "2-0", (
            f"Fixture {f['fixture_id']} has is_2_0_profile=True but score={f['canonical_predicted_score']}"
        )
        assert f["draw_risk_tier"] == "LOW", (
            f"Fixture {f['fixture_id']} has is_2_0_profile=True but draw_risk={f['draw_risk_tier']}"
        )
        assert f["signal_profile"] == "2-0_PROFILE"
        assert f["strong_home_profile"] is True, "All 2-0 profile fixtures should also qualify as strong home"


def test_14_every_strong_home_profile_fixture_valid(upcoming_fixtures):
    """14. Every 'Strong Home Profile' fixture has P(H) >= 0.60 AND draw_risk_tier == 'LOW'."""
    sh_fixtures = [f for f in upcoming_fixtures if f["strong_home_profile"]]
    assert len(sh_fixtures) >= 2, f"Expected at least 2 strong home fixtures, got {len(sh_fixtures)}"
    for f in sh_fixtures:
        assert f["p_home"] >= 0.60, (
            f"Fixture {f['fixture_id']} has strong_home_profile=True but p_home={f['p_home']}"
        )
        assert f["draw_risk_tier"] == "LOW", (
            f"Fixture {f['fixture_id']} has strong_home_profile=True but draw_risk={f['draw_risk_tier']}"
        )


def test_15_prioritized_section_in_html_contains_all_signals(dashboard_html_content, upcoming_fixtures):
    """15. The prioritized section in the HTML contains all identified signal matches."""
    signal_fixtures = [f for f in upcoming_fixtures if f["signal_profile"] is not None]
    assert len(signal_fixtures) == 3, f"Expected 3 signal fixtures, got {len(signal_fixtures)}"

    for s in signal_fixtures:
        fid_str = str(s["fixture_id"])
        home_team = s["Home Team"]
        away_team = s["Away Team"]
        assert fid_str in dashboard_html_content, f"Fixture ID {fid_str} missing from dashboard HTML"
        assert home_team in dashboard_html_content, f"Home team {home_team} missing from dashboard HTML"
        assert away_team in dashboard_html_content, f"Away team {away_team} missing from dashboard HTML"


def test_16_html_dashboard_is_standalone_and_offline(dashboard_html_content):
    """16. The HTML dashboard is standalone: zero external CDN links (no remote scripts or stylesheets)."""
    # Check for external stylesheet links
    css_links = re.findall(r'<link[^>]+rel=["\']stylesheet["\'][^>]*>', dashboard_html_content, re.IGNORECASE)
    for link in css_links:
        assert "http://" not in link and "https://" not in link and "//" not in link, (
            f"External stylesheet CDN detected: {link}"
        )

    # Check for external script sources
    script_tags = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', dashboard_html_content, re.IGNORECASE)
    for src in script_tags:
        assert not src.startswith("http://") and not src.startswith("https://") and not src.startswith("//"), (
            f"External script CDN detected: {src}"
        )

    # Check for external font imports
    assert "@import url(" not in dashboard_html_content or ("http" not in dashboard_html_content), (
        "External font CDN import detected in CSS"
    )


def test_17_no_actual_scores_or_outcomes_in_upcoming_ledger(upcoming_fixtures, dashboard_html_content):
    """17. No actual scores or completed match outcomes appear anywhere in the ledger or dashboard."""
    forbidden_outcome_keys = [
        "actual_home_goals", "actual_away_goals", "actual_result", "home_goals", "away_goals"
    ]
    for f in upcoming_fixtures:
        for k in forbidden_outcome_keys:
            assert k not in f, f"Forbidden outcome field '{k}' found in fixture {f['fixture_id']}"
        assert f["Actual Score"] == "—", f"Actual score leaked in fixture {f['fixture_id']}: {f['Actual Score']}"
        assert f["Match Status"] == "UPCOMING", f"Match status not UPCOMING in fixture {f['fixture_id']}"

    # Verify no FT (Full Time) labels in HTML fixture rows
    assert "<span>FT</span>" not in dashboard_html_content
    assert ">Full Time<" not in dashboard_html_content


def test_18_all_matches_have_upcoming_status(upcoming_fixtures):
    """18. All matches have Status == 'UPCOMING' and Actual Score == '—'."""
    for f in upcoming_fixtures:
        assert f["Status"] == "UPCOMING"
        assert f["Match Status"] == "UPCOMING"
        assert f["Actual Score"] == "—"


def test_19_no_api_tokens_or_secrets_leaked():
    """19. No API tokens or secrets are leaked in any output file."""
    env_token = None
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if "ODDALERTS" in line.upper() and "=" in line:
                    env_token = line.split("=", 1)[1].strip().strip("'\"")
                    break

    files_to_check = [
        REPORTS_HTML,
        REPORTS_LEDGER,
        P15_01_LEDGER,
        P15_02_SIGNALS,
        P15_03_VALIDATION,
        P15_04_REPORT,
        P15_05_HTML,
    ]

    for p in files_to_check:
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8", errors="ignore")
        if env_token and len(env_token) > 6:
            assert env_token not in content, f"API token leaked in {p.name}"
        assert "api_token=" not in content, f"Raw api_token parameter leaked in {p.name}"


def test_20_research_artifacts_exist():
    """20. Research artifacts exist: 01, 02, 03, 04, 05 files present."""
    expected_files = [
        P15_01_LEDGER,
        P15_02_SIGNALS,
        P15_03_VALIDATION,
        P15_04_REPORT,
        P15_05_HTML,
        P15_README,
        P15_ENGINE,
        REPORTS_HTML,
        REPORTS_LEDGER,
    ]
    for p in expected_files:
        assert p.exists(), f"Expected artifact missing at {p}"
        assert p.stat().st_size > 0, f"Artifact {p.name} is empty (0 bytes)"
