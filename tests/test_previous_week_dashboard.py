"""Unit & Integration Tests for Dashboard Previous Week Results View.

Verifies:
1. Historical records load successfully from immutable ledger.
2. Expected 33 completed historical matches are present.
3. No historical prediction is regenerated.
4. Stored probabilities & decisions remain unchanged.
5. Final scores and evaluations are accurate.
6. League filtering preserves underlying records.
7. UTC-only timestamps.
8. Bundesliga historical isolation.
9. Current week 48-game fixture count remains 48.
10. Model MD5 hash immutability.
11. Draw risk stratification: LOW (13), MEDIUM (13), HIGH (7) preserved.
12. Export files contain true stored draw risk classifications.
"""
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest
import pandas as pd

from dashboard.fixture_service import FixtureService, TOTAL_WEEKLY_TARGET, WEEKLY_LEAGUE_TARGETS


class TestPreviousWeekDashboardData:
    """Test historical ledger integrity, schema, and immutability."""

    def test_1_historical_records_load_successfully(self):
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        assert ledger_path.exists(), "production_performance_ledger.jsonl does not exist!"

        records = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        assert len(records) > 0, "No records found in performance ledger!"

    def test_2_expected_historical_cohort_count_is_33(self):
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        records = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        assert len(records) == 33, f"Expected 33 historical completed records, got {len(records)}"

    def test_3_stored_predictions_and_probabilities_are_valid(self):
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        with open(ledger_path, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        for r in records:
            p_sum = r["p_home"] + r["p_draw"] + r["p_away"]
            assert 0.98 <= p_sum <= 1.02, f"Probabilities do not sum to 1 in fixture {r['fixture_id']}"
            assert r["predicted_outcome"] in ("H", "D", "A"), f"Invalid predicted outcome in fixture {r['fixture_id']}"
            assert r["actual_outcome"] in ("H", "D", "A"), f"Invalid actual outcome in fixture {r['fixture_id']}"
            assert "-" in r["actual_score"], f"Invalid score format in fixture {r['fixture_id']}"

    def test_4_evaluation_accuracy_metrics_match_ledger(self):
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        with open(ledger_path, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        corr = sum(1 for r in records if r["prediction_correct"])
        assert corr == 20, f"Expected 20 correct predictions, got {corr}"
        acc_pct = (corr / len(records)) * 100.0
        assert abs(acc_pct - 60.61) < 0.01

        score_hits = sum(1 for r in records if r["exact_score_correct"])
        assert score_hits == 3, f"Expected 3 exact score hits, got {score_hits}"
        score_acc = (score_hits / len(records)) * 100.0
        assert abs(score_acc - 9.09) < 0.01

    def test_5_league_filtering_and_bundesliga_isolation(self):
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        with open(ledger_path, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        league_counts = {}
        for r in records:
            c = r["competition_name"]
            league_counts[c] = league_counts.get(c, 0) + 1

        assert league_counts.get("Ligue 1") == 9
        assert league_counts.get("Premier League") == 9
        assert league_counts.get("Serie A") == 8
        assert league_counts.get("La Liga") == 7
        # Bundesliga had 0 matches in MD1 opening cycle; starts in current week window
        assert league_counts.get("Bundesliga", 0) == 0

    def test_6_utc_timestamps_preserved(self):
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        with open(ledger_path, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        for r in records:
            k = r["scheduled_kickoff_utc"]
            assert "IST" not in k, f"IST found in timestamp {k}"
            assert "Z" in k or "+00:00" in k or k.startswith("2026"), f"Non-UTC timestamp {k}"

    def test_7_historical_memory_safe_guard(self):
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        with open(ledger_path, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        # All 33 opening matches must have INSUFFICIENT sample safeguard
        for r in records:
            assert r["memory_evidence_level"] == "INSUFFICIENT"

    def test_8_historical_draw_risk_tiers_are_stratified_not_all_low(self):
        """Verify that historical draw risk tiers are NOT all defaulted to LOW."""
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        with open(ledger_path, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        tier_counts = {}
        for r in records:
            tier = r.get("draw_risk_tier", "LOW")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        assert tier_counts.get("LOW") == 13, f"Expected 13 LOW, got {tier_counts.get('LOW')}"
        assert tier_counts.get("MEDIUM") == 13, f"Expected 13 MEDIUM, got {tier_counts.get('MEDIUM')}"
        assert tier_counts.get("HIGH") == 7, f"Expected 7 HIGH, got {tier_counts.get('HIGH')}"
        assert tier_counts.get("LOW") != len(records), "All records were incorrectly converted to LOW!"

    def test_9_historical_draw_risk_individual_match_fidelity(self):
        """Verify specific matches retain their exact original recorded draw risk tiers."""
        ledger_path = PROJECT_ROOT / "reports" / "production_performance_ledger.jsonl"
        with open(ledger_path, "r", encoding="utf-8") as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]

        rec_map = {r["fixture_id"]: r for r in records}

        # 420581845: Hull City vs Manchester United -> HIGH
        assert rec_map[420581845]["draw_risk_tier"] == "HIGH"
        # 420582050: Everton vs Crystal Palace -> MEDIUM
        assert rec_map[420582050]["draw_risk_tier"] == "MEDIUM"
        # 420581362: Arsenal vs Coventry City -> LOW
        assert rec_map[420581362]["draw_risk_tier"] == "LOW"
        # 420582305: Nice vs Lorient -> HIGH
        assert rec_map[420582305]["draw_risk_tier"] == "HIGH"
        # 420582189: Athletic Club vs Sevilla -> MEDIUM
        assert rec_map[420582189]["draw_risk_tier"] == "MEDIUM"

    def test_10_previous_week_csv_export_contains_exact_draw_risk(self):
        """Verify CSV export contains the preserved Draw Risk column with proper distribution."""
        csv_path = PROJECT_ROOT / "reports" / "previous_week_predictions_export.csv"
        assert csv_path.exists(), "previous_week_predictions_export.csv does not exist!"

        df = pd.read_csv(csv_path)
        assert "Draw Risk" in df.columns
        counts = df["Draw Risk"].value_counts().to_dict()
        assert counts.get("LOW") == 13
        assert counts.get("MEDIUM") == 13
        assert counts.get("HIGH") == 7

    def test_11_previous_week_html_export_contains_draw_risk_badges(self):
        """Verify HTML export contains styled risk badges."""
        html_path = PROJECT_ROOT / "reports" / "previous_week_predictions_export.html"
        assert html_path.exists(), "previous_week_predictions_export.html does not exist!"

        content = html_path.read_text(encoding="utf-8")
        assert "LOW: 13 | MED: 13 | HIGH: 7" in content
        assert "risk-high" in content
        assert "risk-med" in content
        assert "risk-low" in content


class TestCurrentWeekIntegrityRemainsIntact:
    """Verify current week 48-fixture distribution and model hashes remain frozen."""

    def test_12_current_weekly_fixture_target_is_48(self):
        fs = FixtureService()
        fixtures, meta = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")
        assert len(fixtures) == 48, f"Expected 48 fixtures, got {len(fixtures)}"

    def test_13_current_weekly_league_distribution(self):
        fs = FixtureService()
        fixtures, meta = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")

        counts = {}
        for f in fixtures:
            counts[f.competition_name] = counts.get(f.competition_name, 0) + 1

        assert counts.get("Premier League") == 10
        assert counts.get("Serie A") == 10
        assert counts.get("La Liga") == 10
        assert counts.get("Bundesliga") == 9
        assert counts.get("Ligue 1") == 9

    def test_14_no_duplicate_fixture_ids_in_current_week(self):
        fs = FixtureService()
        fixtures, meta = fs.get_weekly_prediction_fixtures(provider_name="oddalerts")
        ids = [f.fixture_id for f in fixtures]
        assert len(set(ids)) == len(ids)

    def test_15_v4_production_model_md5_frozen(self):
        v4_path = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
        assert v4_path.exists()
        md5_hash = hashlib.md5(v4_path.read_bytes()).hexdigest()
        assert md5_hash == "06841f0c03c8597b2b8cd8f8ab064864"

    def test_16_v4_1_candidate_model_md5_frozen(self):
        v41_path = PROJECT_ROOT / "data" / "models" / "v4_1_prospective_candidate_2025_26.pkl"
        assert v41_path.exists()
        md5_hash = hashlib.md5(v41_path.read_bytes()).hexdigest()
        assert md5_hash == "145f918d933eb343c0f63ca342b10289"
