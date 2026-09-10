"""Regression and validation suite for Previous Week results (29 Aug 2026 -> 1 Sep 2026).

Verifies requirements A through L:
A. 29 Aug fixtures are included (22 fixtures)
B. 30 Aug fixtures are included (15 fixtures)
C. 31 Aug fixtures are included (5 fixtures)
D. 1 Sep fixtures are included (0 fixtures, verified no fake fixtures)
E. No fixtures before 29 Aug are included in Previous Week
F. No fixtures after 1 Sep are included in Previous Week
G. Historical predictions are loaded from stored snapshots, NOT regenerated
H. Actual scores appear only for completed matches
I. Current Week remains isolated (48 fixtures, zero overlap with Previous Week)
J. Upcoming Week remains isolated (zero overlap with Previous Week or Current Week)
K. Date filter correctly isolates each UTC date
L. Existing previous-week accuracy calculations remain unchanged
"""
import hashlib
import json
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard.fixture_service import FixtureService
from dashboard.prediction_snapshot_store import get_prediction_snapshot_store


@pytest.fixture(scope="module")
def fixture_service():
    return FixtureService()


@pytest.fixture(scope="module")
def prev_week_records(fixture_service):
    return fixture_service.get_previous_week_performance_records(
        start_date="2026-08-29",
        end_date="2026-09-01",
        provider_name="oddalerts",
    )


class TestPreviousWeekAug29Sep01:
    """Test suite validating requirements A through L for 29 Aug -> 1 Sep 2026."""

    def test_A_29_aug_fixtures_included(self, prev_week_records):
        recs_29 = [r for r in prev_week_records if r["Date (UTC)"] == "2026-08-29"]
        assert len(recs_29) == 22, f"Expected exactly 22 fixtures on 2026-08-29, got {len(recs_29)}"

    def test_B_30_aug_fixtures_included(self, prev_week_records):
        recs_30 = [r for r in prev_week_records if r["Date (UTC)"] == "2026-08-30"]
        assert len(recs_30) == 15, f"Expected exactly 15 fixtures on 2026-08-30, got {len(recs_30)}"

    def test_C_31_aug_fixtures_included(self, prev_week_records):
        recs_31 = [r for r in prev_week_records if r["Date (UTC)"] == "2026-08-31"]
        assert len(recs_31) == 5, f"Expected exactly 5 fixtures on 2026-08-31, got {len(recs_31)}"

    def test_D_01_sep_fixtures_handled(self, prev_week_records):
        recs_01 = [r for r in prev_week_records if r["Date (UTC)"] == "2026-09-01"]
        assert len(recs_01) == 0, f"Expected 0 fixtures scheduled on 2026-09-01, got {len(recs_01)}"

    def test_E_no_fixtures_before_29_aug(self, prev_week_records):
        early = [r for r in prev_week_records if r["Date (UTC)"] < "2026-08-29"]
        assert len(early) == 0, f"Found unexpected fixtures before 2026-08-29: {[r['fixture_id'] for r in early]}"

    def test_F_no_fixtures_after_01_sep(self, prev_week_records):
        late = [r for r in prev_week_records if r["Date (UTC)"] > "2026-09-01"]
        assert len(late) == 0, f"Found unexpected fixtures after 2026-09-01: {[r['fixture_id'] for r in late]}"

    def test_G_historical_predictions_loaded_from_stored_snapshots(self, prev_week_records):
        store = get_prediction_snapshot_store()
        assert len(prev_week_records) == 42, f"Expected 42 total completed records, got {len(prev_week_records)}"
        for r in prev_week_records:
            fid = r["fixture_id"]
            snap = store.get_snapshot(fid)
            assert snap is not None, f"Fixture {fid} missing immutable snapshot!"
            assert snap.is_locked, f"Snapshot for fixture {fid} is not locked!"
            assert abs(snap.p_home - r["p_home"]) < 1e-4, f"Mismatch in p_home for {fid}"
            assert abs(snap.p_draw - r["p_draw"]) < 1e-4, f"Mismatch in p_draw for {fid}"
            assert abs(snap.p_away - r["p_away"]) < 1e-4, f"Mismatch in p_away for {fid}"
            assert snap.model_decision == r["predicted_outcome"], f"Mismatch in model_decision for {fid}"

    def test_H_actual_scores_appear_only_for_completed_matches(self, prev_week_records):
        for r in prev_week_records:
            assert r["actual_score"] != "", f"Missing actual score for fixture {r['fixture_id']}"
            assert "-" in r["actual_score"], f"Invalid score format for fixture {r['fixture_id']}"
            assert r["Actual Outcome"] in ("H", "D", "A"), f"Invalid outcome for fixture {r['fixture_id']}"
            assert r["actual_home_goals"] is not None
            assert r["actual_away_goals"] is not None

    def test_I_current_week_remains_isolated(self, fixture_service, prev_week_records):
        current_fixtures, meta = fixture_service.get_weekly_prediction_fixtures(offset_weeks=0)
        assert len(current_fixtures) == 48, f"Expected 48 current fixtures, got {len(current_fixtures)}"
        prev_fids = {r["fixture_id"] for r in prev_week_records}
        curr_fids = {f.fixture_id for f in current_fixtures}
        overlap = prev_fids.intersection(curr_fids)
        assert len(overlap) == 0, f"Found overlap between Previous Week and Current Week: {overlap}"

    def test_J_upcoming_week_remains_isolated(self, fixture_service, prev_week_records):
        current_fixtures, _ = fixture_service.get_weekly_prediction_fixtures(offset_weeks=0)
        upcoming_fixtures, _ = fixture_service.get_weekly_prediction_fixtures(offset_weeks=1)
        prev_fids = {r["fixture_id"] for r in prev_week_records}
        curr_fids = {f.fixture_id for f in current_fixtures}
        up_fids = {f.fixture_id for f in upcoming_fixtures}
        overlap_prev = prev_fids.intersection(up_fids)
        assert len(overlap_prev) == 0, f"Found overlap between Previous Week and Upcoming Week: {overlap_prev}"
        overlap_curr = curr_fids.intersection(up_fids)
        assert len(overlap_curr) == 0, f"Found overlap between Current Week and Upcoming Week: {overlap_curr}"

    def test_K_date_filter_correctly_isolates_each_utc_date(self, prev_week_records):
        dates = ["2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01"]
        expected_counts = {"2026-08-29": 22, "2026-08-30": 15, "2026-08-31": 5, "2026-09-01": 0}
        for d in dates:
            filtered = [r for r in prev_week_records if r["Date (UTC)"] == d]
            assert len(filtered) == expected_counts[d], f"Date {d} expected {expected_counts[d]} matches, got {len(filtered)}"

    def test_L_accuracy_calculations_match_methodology(self, prev_week_records):
        tot = len(prev_week_records)
        assert tot == 42
        corr = sum(1 for r in prev_week_records if r["prediction_correct"])
        exact = sum(1 for r in prev_week_records if r["exact_score_correct"])
        acc_1x2 = corr / tot * 100.0
        exact_acc = exact / tot * 100.0

        assert corr == 25, f"Expected 25 correct predictions, got {corr}"
        assert tot - corr == 17, f"Expected 17 wrong predictions, got {tot - corr}"
        assert abs(acc_1x2 - 59.52) < 0.05, f"Expected ~59.52% accuracy, got {acc_1x2:.2f}%"
        assert exact == 3, f"Expected 3 exact score hits, got {exact}"
        assert abs(exact_acc - 7.14) < 0.05, f"Expected ~7.14% exact score accuracy, got {exact_acc:.2f}%"

        brier_avg = sum(r["brier_score"] for r in prev_week_records) / tot
        assert abs(brier_avg - 0.5619) < 0.01, f"Expected ~0.5619 Brier score, got {brier_avg:.4f}"

    def test_M_frozen_model_hashes(self):
        v4_path = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
        v41_path = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"

        h_v4 = hashlib.md5(v4_path.read_bytes()).hexdigest()
        h_v41 = hashlib.md5(v41_path.read_bytes()).hexdigest()

        assert h_v4 == "06841f0c03c8597b2b8cd8f8ab064864", "V4.0 Production model hash modified!"
        assert h_v41 == "145f918d933eb343c0f63ca342b10289", "V4.1 Candidate model hash modified!"

    def test_N_draw_risk_stratification_preserved_not_all_low(self, prev_week_records):
        tiers = [r.get("draw_risk_tier") for r in prev_week_records]
        low_count = tiers.count("LOW")
        med_count = tiers.count("MEDIUM")
        high_count = tiers.count("HIGH")

        assert low_count == 23, f"Expected 23 LOW draw risk fixtures, got {low_count}"
        assert med_count == 13, f"Expected 13 MEDIUM draw risk fixtures, got {med_count}"
        assert high_count == 6, f"Expected 6 HIGH draw risk fixtures, got {high_count}"

        # Specific matches must preserve their original stored HIGH risk classification
        rec_map = {r["fixture_id"]: r for r in prev_week_records}
        assert rec_map[420592585]["draw_risk_tier"] == "HIGH"  # FC Union Berlin vs Eintracht Frankfurt
        assert rec_map[420592584]["draw_risk_tier"] == "HIGH"  # FC Koln vs TSG Hoffenheim
        assert rec_map[420592998]["draw_risk_tier"] == "HIGH"  # Brest vs Toulouse
        assert rec_map[420593006]["draw_risk_tier"] == "HIGH"  # Sevilla vs Atletico Madrid
        assert rec_map[420593587]["draw_risk_tier"] == "HIGH"  # Chelsea vs Brighton
        assert rec_map[420593832]["draw_risk_tier"] == "HIGH"  # Celta de Vigo vs Athletic Club

    def test_O_no_duplicate_fixtures_in_previous_week(self, prev_week_records):
        fids = [r["fixture_id"] for r in prev_week_records]
        assert len(fids) == len(set(fids)), f"Duplicate fixture IDs found in Previous Week: {len(fids) - len(set(fids))}"

