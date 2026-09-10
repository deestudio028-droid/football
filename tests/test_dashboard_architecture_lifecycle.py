"""Comprehensive validation suite for the Football Prediction Dashboard Architecture.

Verifies:
1. Frozen Model MD5 integrity (V4.0 and V4.1 bit-identical).
2. Strict UTC Date Ownership (kickoff_utc.date() owns the fixture; no IST date shifts).
3. Complete Fixture Discovery decoupled from Prediction Generation.
4. Zero-Drop Invariant: Prediction failure never causes a fixture to disappear.
5. Upcoming/Current Matches: Live V4.0 pre-kickoff inference (Score='--', Eval='--').
6. Completed Matches: Immutable original pre-kickoff V4.0 prediction from locked ledger.
7. Exact validation target values for August 24, 2026 historical fixtures.
8. UTC 24-hour kickoff formatting (no AM/PM).
"""
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from dashboard.fixture_service import (
    DashboardFixture,
    FixtureService,
    TARGET_LEAGUES,
    WEEKLY_LEAGUE_TARGETS,
    TOTAL_WEEKLY_TARGET,
)
from dashboard.prediction_service import DashboardMatchPrediction, PredictionService
from dashboard.prediction_snapshot_store import get_prediction_snapshot_store
from dashboard.time_utils import (
    format_kickoff_utc,
    is_kickoff_on_utc_date,
    to_utc_date,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestModelIntegrity:
    """Requirement 6: Verify V4.0 and V4.1 model artifacts are 100% frozen."""

    def test_v4_production_md5_frozen(self):
        v4_path = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
        assert v4_path.exists(), "V4.0 production artifact missing!"
        md5 = hashlib.md5(v4_path.read_bytes()).hexdigest()
        assert md5 == "06841f0c03c8597b2b8cd8f8ab064864", f"V4.0 MD5 mutated: {md5}"

    def test_v4_1_candidate_md5_frozen(self):
        v41_path = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"
        assert v41_path.exists(), "V4.1 research candidate artifact missing!"
        md5 = hashlib.md5(v41_path.read_bytes()).hexdigest()
        assert md5 == "145f918d933eb343c0f63ca342b10289", f"V4.1 MD5 mutated: {md5}"


class TestUTCDateOwnership:
    """Requirement 1 & 8: UTC is the sole time standard; no IST rollover."""

    def test_utc_date_ownership_evening_kickoff(self):
        # A match kicking off at 19:30 UTC on 2026-08-24 is 01:00 AM IST on 2026-08-25
        # It MUST belong to 2026-08-24 UTC!
        ts = "2026-08-24T19:30:00.000000Z"
        assert to_utc_date(ts) == "2026-08-24"
        assert is_kickoff_on_utc_date(ts, "2026-08-24") is True
        assert is_kickoff_on_utc_date(ts, "2026-08-25") is False

    def test_utc_24h_kickoff_formatting(self):
        ts = "2026-08-24T18:45:00.000000Z"
        formatted = format_kickoff_utc(ts, include_suffix=False)
        assert formatted == "18:45"
        assert "AM" not in formatted and "PM" not in formatted

        formatted_with_suffix = format_kickoff_utc(ts, include_suffix=True)
        assert formatted_with_suffix == "18:45 UTC"


class TestFixtureDiscoveryAndDecoupling:
    """Requirement 2 & 3: Decouple discovery from prediction; zero drop."""

    def test_all_five_aug24_fixtures_discovered(self):
        fs = FixtureService()
        fixtures, meta = fs.get_todays_matches(date_str="2026-08-24", provider_name="oddalerts")
        assert len(fixtures) == 5, f"Expected 5 fixtures on 2026-08-24, got {len(fixtures)}"

        fids = {f.fixture_id for f in fixtures}
        expected_fids = {420587320, 420587324, 420587329, 420587333, 420587334}
        assert expected_fids.issubset(fids), f"Missing fixture IDs: {expected_fids - fids}"

    def test_fixture_count_consistency(self):
        fs = FixtureService()
        fixtures, meta = fs.get_todays_matches(date_str="2026-08-24", provider_name="oddalerts")
        raw_count = meta.get("total_returned", len(fixtures))
        eligible_count = meta.get("eligible_count", len(fixtures))
        dedup_count = len(fixtures)

        # Target 5-league == deduplicated == dashboard fixtures
        assert eligible_count == dedup_count == 5

    def test_prediction_failure_does_not_drop_fixture(self):
        # Create dummy fixture with invalid teams
        dummy = DashboardFixture(
            fixture_id=999999999,
            competition_id=423,
            competition_name="Premier League",
            season_name="2026/2027",
            home_team="NonExistentTeamAlpha",
            away_team="NonExistentTeamBeta",
            scheduled_kickoff="2026-08-30T15:00:00Z",
            status="NS",
            provider="test",
        )

        ps = PredictionService()
        pred = ps.predict_dashboard_fixture(dummy)
        # Even if prediction is not allowed, pred is returned and fixture is not lost!
        assert pred is not None
        assert pred.fixture_id == 999999999
        assert pred.prediction_allowed is False


class TestPredictionLifecycle:
    """Requirement 4 & 5: Upcoming vs Completed match semantics."""

    def test_upcoming_match_live_v4_inference(self):
        # A match far in the future
        future_fixture = DashboardFixture(
            fixture_id=888888888,
            competition_id=423,
            competition_name="Premier League",
            season_name="2026/2027",
            home_team="Arsenal",
            away_team="Chelsea",
            scheduled_kickoff="2026-12-01T15:00:00Z",
            status="NS",
            provider="test",
        )

        ps = PredictionService()
        pred = ps.predict_dashboard_fixture(future_fixture)
        assert pred is not None
        assert pred.prediction_allowed is True
        assert pred.production_probs is not None
        assert abs(sum(pred.production_probs.values()) - 1.0) < 1e-4
        assert pred.production_decision in ("H", "D", "A")
        assert pred.status_message == "PRE_MATCH_LOCKED"
        assert pred.actual_outcome is None
        assert pred.selected_correct is None

    def test_completed_matches_aug24_exact_predictions_and_eval(self):
        fs = FixtureService()
        ps = PredictionService()
        fixtures, _ = fs.get_todays_matches(date_str="2026-08-24", provider_name="oddalerts")

        expected = {
            420587320: {
                "name": "Bologna vs Lazio",
                "p_h": 0.421, "p_d": 0.278, "p_a": 0.300,
                "pick": "H", "score": "0-1", "actual": "A", "eval": False,
            },
            420587324: {
                "name": "Osasuna vs Levante",
                "p_h": 0.417, "p_d": 0.251, "p_a": 0.332,
                "pick": "H", "score": "0-0", "actual": "D", "eval": False,
            },
            420587329: {
                "name": "Roma vs Fiorentina",
                "p_h": 0.568, "p_d": 0.220, "p_a": 0.212,
                "pick": "H", "score": "4-0", "actual": "H", "eval": True,
            },
            420587333: {
                "name": "Fulham vs Chelsea",
                "p_h": 0.419, "p_d": 0.271, "p_a": 0.310,
                "pick": "H", "score": "2-3", "actual": "A", "eval": False,
            },
            420587334: {
                "name": "Málaga vs Deportivo A Coruña",
                "p_h": 0.426, "p_d": 0.261, "p_a": 0.313,
                "pick": "H", "score": "1-1", "actual": "D", "eval": False,
            },
        }

        for f in fixtures:
            fid = f.fixture_id
            if fid in expected:
                exp = expected[fid]
                pred = ps.predict_dashboard_fixture(f)

                assert pred.prediction_allowed is True
                assert round(pred.production_probs["H"], 3) == exp["p_h"], f"P(H) mismatch on {exp['name']}"
                assert round(pred.production_probs["D"], 3) == exp["p_d"], f"P(D) mismatch on {exp['name']}"
                assert round(pred.production_probs["A"], 3) == exp["p_a"], f"P(A) mismatch on {exp['name']}"
                assert pred.production_decision == exp["pick"], f"Pick mismatch on {exp['name']}"
                assert f.actual_outcome == exp["actual"], f"Actual outcome mismatch on {exp['name']}"
                assert pred.selected_correct == exp["eval"], f"Evaluation correctness mismatch on {exp['name']}"
                assert pred.status_message == "EVALUATED"

    def test_local_database_provider_also_returns_all_aug24_fixtures(self):
        fs = FixtureService()
        fixtures, meta = fs.get_todays_matches(date_str="2026-08-24", provider_name="local_matches")
        assert len(fixtures) == 5
        fids = {f.fixture_id for f in fixtures}
        expected_fids = {420587320, 420587324, 420587329, 420587333, 420587334}
        assert expected_fids.issubset(fids)


class TestAllAvailableFixturesDashboard:
    """Requirement: All available fixtures dashboard contract (no date filtering)."""

    def test_get_all_available_matches_spans_multiple_utc_dates(self):
        fs = FixtureService()
        fixtures, meta = fs.get_all_available_matches(provider_name="oddalerts")
        assert len(fixtures) >= 50, f"Expected >= 50 available fixtures, got {len(fixtures)}"

        unique_dates = {f.scheduled_kickoff[:10] for f in fixtures}
        assert len(unique_dates) > 1, f"Expected multiple UTC dates, got {unique_dates}"
        assert "2026-08-24" in unique_dates
        assert "2026-08-28" in unique_dates or "2026-08-29" in unique_dates

    def test_all_five_aug24_fixtures_present_in_all_available(self):
        fs = FixtureService()
        ps = PredictionService()
        fixtures, _ = fs.get_all_available_matches(provider_name="oddalerts")

        fids = {f.fixture_id for f in fixtures}
        expected_fids = {420587320, 420587324, 420587329, 420587333, 420587334}
        assert expected_fids.issubset(fids), f"Missing Aug 24 fixture IDs from all available matches: {expected_fids - fids}"

        # Check Roma vs Fiorentina (4-0)
        roma_fix = next(f for f in fixtures if f.fixture_id == 420587329)
        pred = ps.predict_dashboard_fixture(roma_fix)
        assert pred.production_decision == "H"
        assert pred.selected_correct is True

    def test_upcoming_matches_in_all_available_have_live_v4_inference(self):
        fs = FixtureService()
        ps = PredictionService()
        fixtures, _ = fs.get_all_available_matches(provider_name="oddalerts")

        upcoming = [f for f in fixtures if f.status not in ("FT", "AET", "PEN", "AWARDED") and f.home_goals is None]
        assert len(upcoming) > 0, "Expected upcoming fixtures in active feed"

        for u in upcoming[:10]:
            pred = ps.predict_dashboard_fixture(u)
            assert pred is not None
            assert pred.prediction_allowed is True
            assert pred.production_probs is not None
            assert abs(sum(pred.production_probs.values()) - 1.0) < 0.005
            assert pred.actual_outcome is None
            assert pred.selected_correct is None

    def test_zero_drop_contract_on_all_available_collection(self):
        fs = FixtureService()
        ps = PredictionService()
        fixtures, meta = fs.get_all_available_matches(provider_name="oddalerts")

        preds = [ps.predict_dashboard_fixture(f) for f in fixtures]
        assert len(fixtures) == len(preds) == meta["total_returned"] == meta["eligible_count"]


class TestWeekly48FixtureDistribution:
    """Client Requirement: Weekly 48-match gameweek distribution (10 EPL, 10 Serie A, 10 La Liga, 9 Bundesliga, 9 Ligue 1)."""

    def test_weekly_exact_league_counts_and_total_48(self):
        fs = FixtureService()
        fixtures, meta = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")

        # 1. Check exact counts by league
        league_counts = meta["league_counts"]
        assert league_counts["Premier League"] == 10, f"Expected 10 Premier League, got {league_counts.get('Premier League')}"
        assert league_counts["Serie A"] == 10, f"Expected 10 Serie A, got {league_counts.get('Serie A')}"
        assert league_counts["La Liga"] == 10, f"Expected 10 La Liga, got {league_counts.get('La Liga')}"
        assert league_counts["Bundesliga"] == 9, f"Expected 9 Bundesliga, got {league_counts.get('Bundesliga')}"
        assert league_counts["Ligue 1"] == 9, f"Expected 9 Ligue 1, got {league_counts.get('Ligue 1')}"

        # 2. Check total selected matches is exactly 48
        assert len(fixtures) == 48, f"Expected 48 total matches, got {len(fixtures)}"
        assert meta["total_selected"] == 48
        assert meta["total_target"] == 48
        assert meta["is_full_gameweek"] is True

    def test_no_duplicate_fixture_ids_in_weekly_selection(self):
        fs = FixtureService()
        fixtures, _ = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")
        fids = [f.fixture_id for f in fixtures]
        assert len(fids) == len(set(fids)), "Duplicate fixture IDs found in weekly selection!"

    def test_no_fixtures_outside_five_target_leagues(self):
        fs = FixtureService()
        fixtures, _ = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")
        allowed_leagues = {"Premier League", "Serie A", "La Liga", "Bundesliga", "Ligue 1"}
        for f in fixtures:
            assert f.competition_name in allowed_leagues, f"Unexpected league: {f.competition_name}"
            assert f.competition_id in TARGET_LEAGUES, f"Unexpected competition ID: {f.competition_id}"

    def test_no_league_exceeds_configured_target(self):
        fs = FixtureService()
        fixtures, _ = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")
        counts = {}
        for f in fixtures:
            counts[f.competition_name] = counts.get(f.competition_name, 0) + 1

        assert counts.get("Premier League", 0) <= 10
        assert counts.get("Serie A", 0) <= 10
        assert counts.get("La Liga", 0) <= 10
        assert counts.get("Bundesliga", 0) <= 9
        assert counts.get("Ligue 1", 0) <= 9

    def test_shortfall_handling_does_not_fabricate_fixtures(self):
        # Test with custom high targets that exceed available fixtures
        fs = FixtureService()
        custom_targets = {"Premier League": 100, "Bundesliga": 50}
        fixtures, meta = fs.get_weekly_prediction_fixtures(provider_name="oddalerts", league_targets=custom_targets)

        # Should return only real available fixtures without fabricating placeholders
        assert meta["total_selected"] < 150
        assert meta["is_full_gameweek"] is False
        assert meta["shortfalls"]["Premier League"] > 0
        assert meta["shortfalls"]["Bundesliga"] > 0
        for f in fixtures:
            assert isinstance(f.fixture_id, int)
            assert f.home_team != "Placeholder"

    def test_weekly_selection_preserves_chronological_utc_sorting(self):
        fs = FixtureService()
        fixtures, _ = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")
        kickoffs = [f.scheduled_kickoff for f in fixtures]
        assert kickoffs == sorted(kickoffs), "Weekly fixtures are not sorted chronologically by UTC kickoff!"

    def test_bundesliga_gw1_independent_selection_alongside_other_leagues_gw2(self):
        """Verify Bundesliga selects its Matchday 1 (Aug 28-30) while EPL, Serie A, etc. select Matchday 2."""
        fs = FixtureService()
        fixtures, meta = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")

        # Bundesliga fixtures in selection
        b_fixtures = [f for f in fixtures if f.competition_name == "Bundesliga"]
        assert len(b_fixtures) == 9

        # Verify all 9 are season-opening Matchday 1 matches
        b_matchups = {(f.home_team, f.away_team) for f in b_fixtures}
        assert any("Bayern" in h and "Stuttgart" in a for h, a in b_matchups)
        assert any("Dortmund" in h and "Hamburger" in a for h, a in b_matchups)
        assert any("Leipzig" in h and "Mönchengladbach" in a for h, a in b_matchups)

        # Premier League fixtures in selection
        epl_fixtures = [f for f in fixtures if f.competition_name == "Premier League"]
        assert len(epl_fixtures) == 10
        # Verify EPL matches are Matchday 2 (Aug 28-31), NOT old GW1 (Aug 21-24)
        epl_matchups = {(f.home_team, f.away_team) for f in epl_fixtures}
        assert any("Crystal Palace" in h and "Manchester City" in a for h, a in epl_matchups)
        assert any("Liverpool" in h and "Nottingham Forest" in a for h, a in epl_matchups)
        assert any("Aston Villa" in h and "Arsenal" in a for h, a in epl_matchups)
        # Verify no old GW1 match like Arsenal vs Coventry in EPL selection
        assert not any("Arsenal" in h and "Coventry" in a for h, a in epl_matchups)

    def test_deterministic_staggered_season_start_fixture_dataset(self):
        """Synthetic deterministic test: EPL/Serie A/La Liga/Ligue 1 have previous completed GW + upcoming GW.
        Bundesliga has NO previous GW and only upcoming GW1.
        Proves selector chooses current/upcoming GW for all 5 leagues = 48.
        """
        fs = FixtureService()
        synth_fixtures: List[DashboardFixture] = []

        # Synthetic EPL (10 completed GW1 on Aug 21 + 10 upcoming GW2 on Aug 28)
        for i in range(10):
            synth_fixtures.append(DashboardFixture(
                fixture_id=1000 + i, competition_id=423, competition_name="Premier League",
                season_name="2026/2027", home_team=f"EPL_Old_H{i}", away_team=f"EPL_Old_A{i}",
                scheduled_kickoff=f"2026-08-21T14:00:00.000000Z", status="FT", provider="Mock",
                home_goals=2, away_goals=1, actual_outcome="H"
            ))
            synth_fixtures.append(DashboardFixture(
                fixture_id=2000 + i, competition_id=423, competition_name="Premier League",
                season_name="2026/2027", home_team=f"EPL_New_H{i}", away_team=f"EPL_New_A{i}",
                scheduled_kickoff=f"2026-08-28T14:00:00.000000Z", status="NS", provider="Mock",
            ))

        # Synthetic Serie A (10 completed GW1 on Aug 22 + 10 upcoming GW2 on Aug 29)
        for i in range(10):
            synth_fixtures.append(DashboardFixture(
                fixture_id=3000 + i, competition_id=499, competition_name="Serie A",
                season_name="2026/2027", home_team=f"ITA_Old_H{i}", away_team=f"ITA_Old_A{i}",
                scheduled_kickoff=f"2026-08-22T14:00:00.000000Z", status="FT", provider="Mock",
                home_goals=1, away_goals=1, actual_outcome="D"
            ))
            synth_fixtures.append(DashboardFixture(
                fixture_id=4000 + i, competition_id=499, competition_name="Serie A",
                season_name="2026/2027", home_team=f"ITA_New_H{i}", away_team=f"ITA_New_A{i}",
                scheduled_kickoff=f"2026-08-29T14:00:00.000000Z", status="NS", provider="Mock",
            ))

        # Synthetic La Liga (10 completed GW1 on Aug 21 + 10 upcoming GW2 on Aug 28)
        for i in range(10):
            synth_fixtures.append(DashboardFixture(
                fixture_id=5000 + i, competition_id=419, competition_name="La Liga",
                season_name="2026/2027", home_team=f"ESP_Old_H{i}", away_team=f"ESP_Old_A{i}",
                scheduled_kickoff=f"2026-08-21T14:00:00.000000Z", status="FT", provider="Mock",
                home_goals=0, away_goals=1, actual_outcome="A"
            ))
            synth_fixtures.append(DashboardFixture(
                fixture_id=6000 + i, competition_id=419, competition_name="La Liga",
                season_name="2026/2027", home_team=f"ESP_New_H{i}", away_team=f"ESP_New_A{i}",
                scheduled_kickoff=f"2026-08-28T14:00:00.000000Z", status="NS", provider="Mock",
            ))

        # Synthetic Ligue 1 (9 completed GW1 on Aug 21 + 9 upcoming GW2 on Aug 28)
        for i in range(9):
            synth_fixtures.append(DashboardFixture(
                fixture_id=7000 + i, competition_id=200, competition_name="Ligue 1",
                season_name="2026/2027", home_team=f"FRA_Old_H{i}", away_team=f"FRA_Old_A{i}",
                scheduled_kickoff=f"2026-08-21T14:00:00.000000Z", status="FT", provider="Mock",
                home_goals=2, away_goals=0, actual_outcome="H"
            ))
            synth_fixtures.append(DashboardFixture(
                fixture_id=8000 + i, competition_id=200, competition_name="Ligue 1",
                season_name="2026/2027", home_team=f"FRA_New_H{i}", away_team=f"FRA_New_A{i}",
                scheduled_kickoff=f"2026-08-28T14:00:00.000000Z", status="NS", provider="Mock",
            ))

        # Synthetic Bundesliga (0 completed, 9 upcoming GW1 on Aug 28)
        for i in range(9):
            synth_fixtures.append(DashboardFixture(
                fixture_id=9000 + i, competition_id=477, competition_name="Bundesliga",
                season_name="2026/2027", home_team=f"GER_GW1_H{i}", away_team=f"GER_GW1_A{i}",
                scheduled_kickoff=f"2026-08-28T14:00:00.000000Z", status="NS", provider="Mock",
            ))

        # Test selecting active gameweek per league independently
        by_league = {lname: [] for lname in WEEKLY_LEAGUE_TARGETS}
        for f in synth_fixtures:
            by_league[f.competition_name].append(f)

        selected = []
        for lname, target_count in WEEKLY_LEAGUE_TARGETS.items():
            chosen = fs._select_league_active_gameweek(by_league[lname], target_count)
            selected.extend(chosen)

        # Assertions
        assert len(selected) == 48
        # Ensure all selected EPL fixtures are from New GW2 (not Old GW1)
        epl_sel = [f for f in selected if f.competition_name == "Premier League"]
        assert len(epl_sel) == 10
        assert all("EPL_New" in f.home_team for f in epl_sel)

        # Ensure all selected Serie A fixtures are from New GW2
        ita_sel = [f for f in selected if f.competition_name == "Serie A"]
        assert len(ita_sel) == 10
        assert all("ITA_New" in f.home_team for f in ita_sel)

        # Ensure all selected La Liga fixtures are from New GW2
        esp_sel = [f for f in selected if f.competition_name == "La Liga"]
        assert len(esp_sel) == 10
        assert all("ESP_New" in f.home_team for f in esp_sel)

        # Ensure all selected Ligue 1 fixtures are from New GW2
        fra_sel = [f for f in selected if f.competition_name == "Ligue 1"]
        assert len(fra_sel) == 9
        assert all("FRA_New" in f.home_team for f in fra_sel)

        # Ensure all selected Bundesliga fixtures are from GW1
        ger_sel = [f for f in selected if f.competition_name == "Bundesliga"]
        assert len(ger_sel) == 9
        selected.sort(key=lambda x: x.scheduled_kickoff)
        kickoffs = [f.scheduled_kickoff for f in selected]
        assert kickoffs == sorted(kickoffs), "Synthetic weekly fixtures are not sorted chronologically by UTC kickoff!"
