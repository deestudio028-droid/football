"""
tests/test_phase15_final_audit.py
==================================
20+ tests validating the Phase 15 Final Results Audit outputs.

Run with:
  py -3.13 -m pytest -p no:pytest_ethereum tests/test_phase15_final_audit.py -v
"""

import json
import math
import os
import pytest

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_DIR = os.path.join(ROOT, "research", "external_consensus", "phase15_final_audit")
LEDGER_PATH = os.path.join(ROOT, "reports", "upcoming_2026_09_18_to_2026_09_21_ledger.jsonl")
RAW_RESULTS_PATH = os.path.join(ROOT, "data", "phase15_oddalerts_raw.json")

FROZEN_SHA256 = "1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5"


# ─── Fixtures (pytest fixtures) ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def final_ledger():
    path = os.path.join(AUDIT_DIR, "01_phase15_final_results_ledger.jsonl")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def metrics():
    path = os.path.join(AUDIT_DIR, "02_phase15_metrics.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def risk_analysis():
    path = os.path.join(AUDIT_DIR, "03_phase15_risk_analysis.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def signal_analysis():
    path = os.path.join(AUDIT_DIR, "04_phase15_signal_analysis.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def league_analysis():
    path = os.path.join(AUDIT_DIR, "05_phase15_league_analysis.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def forensics():
    path = os.path.join(AUDIT_DIR, "06_phase15_forensics.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def prediction_ledger():
    records = []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def raw_results():
    with open(RAW_RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── Test 1: Output files exist ───────────────────────────────────────────────

def test_01_all_output_files_exist():
    """All 9 required output files must exist."""
    expected_files = [
        "01_phase15_final_results_ledger.jsonl",
        "02_phase15_metrics.json",
        "03_phase15_risk_analysis.json",
        "04_phase15_signal_analysis.json",
        "05_phase15_league_analysis.json",
        "06_phase15_forensics.json",
        "07_phase15_final_report.md",
        "08_phase15_final_dashboard.html",
        "README.md",
    ]
    for fname in expected_files:
        path = os.path.join(AUDIT_DIR, fname)
        assert os.path.exists(path), f"Missing output file: {fname}"


# ─── Test 2: Final ledger has 48 records ──────────────────────────────────────

def test_02_final_ledger_has_48_records(final_ledger):
    """Final results ledger must contain exactly 48 records."""
    assert len(final_ledger) == 48, f"Expected 48 records, got {len(final_ledger)}"


# ─── Test 3: V4 SHA256 is frozen on every record ──────────────────────────────

def test_03_v4_sha256_frozen_on_all_records(final_ledger):
    """Every record must carry the exact frozen V4 SHA256."""
    for r in final_ledger:
        assert r["model_v4_sha256"] == FROZEN_SHA256, (
            f"SHA256 mismatch on fixture {r['fixture_id']}: {r['model_v4_sha256']}"
        )


# ─── Test 4: All 48 fixtures are FT ──────────────────────────────────────────

def test_04_all_48_fixtures_classified_ft(final_ledger):
    """All 48 Phase 15 fixtures must be classified as 'ft' (completed)."""
    non_ft = [r for r in final_ledger if r["classification"] != "ft"]
    assert non_ft == [], f"Non-FT fixtures found: {[(r['fixture_id'], r['classification']) for r in non_ft]}"


# ─── Test 5: Metrics fixture counts match ────────────────────────────────────

def test_05_metrics_fixture_counts(metrics):
    """Metrics must report 48 total and 48 completed fixtures."""
    assert metrics["meta"]["total_fixtures"] == 48
    assert metrics["meta"]["fixtures_completed"] == 48
    assert metrics["meta"]["fixtures_upcoming"] == 0
    assert metrics["meta"]["fixtures_postponed"] == 0


# ─── Test 6: 1X2 accuracy is in [0, 1] and denominator is 48 ─────────────────

def test_06_1x2_accuracy_valid_range(metrics):
    """1X2 accuracy must be a float in [0, 1] with denominator 48."""
    A = metrics["A_1x2_accuracy"]
    assert A["total"] == 48
    assert 0.0 <= A["accuracy"] <= 1.0
    assert 0 <= A["correct"] <= 48
    assert A["correct"] == round(A["accuracy"] * 48)


# ─── Test 7: Exact score accuracy is in [0, 1] ───────────────────────────────

def test_07_exact_score_accuracy_valid(metrics):
    """Exact score accuracy must be a valid fraction in [0, 1]."""
    B = metrics["B_exact_score"]
    assert B["total"] == 48
    assert 0.0 <= B["accuracy"] <= 1.0
    assert 0 <= B["correct"] <= 48


# ─── Test 8: Brier score is in (0, 2] ────────────────────────────────────────

def test_08_brier_score_valid_range(metrics):
    """Mean Brier score must be a positive number ≤ 2.0."""
    brier = metrics["C_brier_score"]["mean"]
    assert 0.0 < brier <= 2.0, f"Brier score out of range: {brier}"


# ─── Test 9: Log loss is positive ────────────────────────────────────────────

def test_09_log_loss_positive(metrics):
    """Mean log loss must be positive."""
    ll = metrics["D_log_loss"]["mean"]
    assert ll > 0.0, f"Log loss must be positive, got {ll}"


# ─── Test 10: Brier score per record is computed correctly ────────────────────

def test_10_brier_score_computation(final_ledger):
    """Spot-check Brier score computation for Bayern vs Union Berlin (7-0, H win)."""
    r = next(rec for rec in final_ledger if rec["fixture_id"] == "420656757")
    assert r["actual_outcome"] == "H"
    p_h = r["p_home"]  # 0.7765
    p_d = r["p_draw"]  # 0.1548
    p_a = r["p_away"]  # 0.0687
    expected_brier = (p_h - 1) ** 2 + (p_d - 0) ** 2 + (p_a - 0) ** 2
    assert abs(r["brier"] - expected_brier) < 1e-4, (
        f"Brier mismatch: got {r['brier']}, expected {expected_brier:.6f}"
    )


# ─── Test 11: Bayern vs Union Berlin actual score is 7-0 ─────────────────────

def test_11_bayern_union_berlin_7_0(final_ledger):
    """Bayern vs Union Berlin must have actual score 7-0, FT, outcome H."""
    r = next(rec for rec in final_ledger if rec["fixture_id"] == "420656757")
    assert r["actual_score"] == "7-0"
    assert r["actual_home_goals"] == 7
    assert r["actual_away_goals"] == 0
    assert r["actual_outcome"] == "H"
    assert r["actual_status"] == "FT"
    assert r["classification"] == "ft"


# ─── Test 12: 2-0 profile fixtures are correctly identified ──────────────────

def test_12_2_0_profile_fixtures(signal_analysis):
    """Exactly 2 fixtures must be flagged as 2-0 profile (Bayern and Man City)."""
    details = signal_analysis["F_2_0_signal"]["details"]
    fids = {str(d["fixture_id"]) for d in details}
    assert "420656757" in fids, "Bayern vs Union Berlin should be 2-0 profile"
    assert "420657842" in fids, "Man City vs Sunderland should be 2-0 profile"
    assert len(details) == 2, f"Expected 2 2-0 profile fixtures, got {len(details)}"


# ─── Test 13: Probabilities are preserved from immutable ledger ──────────────

def test_13_probabilities_preserved(final_ledger, prediction_ledger):
    """All prediction probabilities must exactly match the immutable ledger."""
    pred_by_fid = {str(r["fixture_id"]): r for r in prediction_ledger}
    for r in final_ledger:
        fid = r["fixture_id"]
        pred = pred_by_fid.get(fid)
        assert pred is not None, f"Fixture {fid} not in prediction ledger"
        assert abs(r["p_home"] - pred["p_home"]) < 1e-9
        assert abs(r["p_draw"] - pred["p_draw"]) < 1e-9
        assert abs(r["p_away"] - pred["p_away"]) < 1e-9


# ─── Test 14: Canonical predicted score preserved ────────────────────────────

def test_14_canonical_predicted_score_preserved(final_ledger, prediction_ledger):
    """canonical_predicted_score must be unchanged from the immutable ledger."""
    pred_by_fid = {str(r["fixture_id"]): r for r in prediction_ledger}
    for r in final_ledger:
        fid = r["fixture_id"]
        pred = pred_by_fid.get(fid)
        assert r["canonical_predicted_score"] == pred["canonical_predicted_score"], (
            f"Fixture {fid}: canonical score mismatch — "
            f"got {r['canonical_predicted_score']}, expected {pred['canonical_predicted_score']}"
        )


# ─── Test 15: Outcome consistency — correct field matches outcome comparison ──

def test_15_outcome_correct_consistency(final_ledger):
    """outcome_correct must equal (actual_outcome == pred_outcome) for every FT record."""
    for r in final_ledger:
        if r["classification"] == "ft":
            expected = r["actual_outcome"] == r["pred_outcome"]
            assert r["outcome_correct"] == expected, (
                f"Fixture {r['fixture_id']}: outcome_correct={r['outcome_correct']} "
                f"but actual={r['actual_outcome']}, pred={r['pred_outcome']}"
            )


# ─── Test 16: p_home + p_draw + p_away ≈ 1.0 ────────────────────────────────

def test_16_probabilities_sum_to_one(final_ledger):
    """All three probabilities must sum to ≈ 1.0 (within 0.005)."""
    for r in final_ledger:
        total = r["p_home"] + r["p_draw"] + r["p_away"]
        assert abs(total - 1.0) < 0.005, (
            f"Fixture {r['fixture_id']}: probabilities sum to {total:.6f}"
        )


# ─── Test 17: League breakdown covers all 5 Big-5 leagues ────────────────────

def test_17_league_breakdown_all_5_leagues(league_analysis):
    """League breakdown must contain entries for all 5 Big-5 leagues."""
    K = league_analysis["K_league_breakdown"]
    expected = {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
    # Check each expected league is represented
    for lg in expected:
        assert lg in K, f"League '{lg}' missing from league breakdown"


# ─── Test 18: Draw risk tiers are complete ────────────────────────────────────

def test_18_draw_risk_tiers_complete(risk_analysis):
    """Draw risk analysis must have entries for LOW, MEDIUM, and HIGH tiers."""
    E = risk_analysis["E_draw_risk_by_tier"]
    for tier in ("LOW", "MEDIUM", "HIGH"):
        assert tier in E, f"Tier '{tier}' missing from draw risk analysis"
        assert "count" in E[tier]
        assert "actual_draws" in E[tier]
        assert "draw_rate" in E[tier]
        assert E[tier]["count"] >= 0
        assert 0.0 <= E[tier]["draw_rate"] <= 1.0


# ─── Test 19: Forensics biggest misses are sorted by confidence descending ────

def test_19_biggest_misses_sorted_by_confidence(forensics):
    """Biggest misses must be sorted by max(p_home, p_away) descending."""
    L = forensics["L_biggest_misses"]
    if len(L) < 2:
        pytest.skip("Fewer than 2 misses to check order")
    for i in range(len(L) - 1):
        conf_i = max(L[i]["p_home"], L[i]["p_away"])
        conf_j = max(L[i + 1]["p_home"], L[i + 1]["p_away"])
        assert conf_i >= conf_j, (
            f"Misses not sorted: index {i} confidence {conf_i:.4f} < index {i+1} confidence {conf_j:.4f}"
        )


# ─── Test 20: Log loss computation spot check ─────────────────────────────────

def test_20_log_loss_computation(final_ledger):
    """Spot-check log loss for Espanyol vs Elche (1-3, A win)."""
    r = next(rec for rec in final_ledger if rec["fixture_id"] == "420654561")
    assert r["actual_outcome"] == "A"
    p_away = r["p_away"]
    expected_ll = -math.log(max(min(p_away, 1 - 1e-9), 1e-9))
    assert abs(r["log_loss"] - expected_ll) < 1e-4, (
        f"Log loss mismatch: got {r['log_loss']:.6f}, expected {expected_ll:.6f}"
    )


# ─── Test 21: Atletico vs Real Madrid result ──────────────────────────────────

def test_21_atletico_real_madrid_result(final_ledger):
    """Atletico de Madrid vs Real Madrid must show actual score 2-1."""
    r = next(rec for rec in final_ledger if rec["fixture_id"] == "420654564")
    assert r["actual_score"] == "2-1"
    assert r["actual_outcome"] == "H"
    assert r["actual_status"] == "FT"


# ─── Test 22: Brentford vs Chelsea result ────────────────────────────────────

def test_22_brentford_chelsea_3_0(final_ledger):
    """Brentford vs Chelsea must show actual score 3-0 (FT)."""
    r = next(rec for rec in final_ledger if rec["fixture_id"] == "420656764")
    assert r["actual_score"] == "3-0"
    assert r["actual_home_goals"] == 3
    assert r["actual_away_goals"] == 0
    assert r["actual_outcome"] == "H"


# ─── Test 23: Draw risk tier count must sum to 48 ────────────────────────────

def test_23_draw_risk_tier_count_sums_to_48(risk_analysis, final_ledger):
    """Total fixtures across all draw risk tiers must equal 48."""
    E = risk_analysis["E_draw_risk_by_tier"]
    total = sum(E[t]["count"] for t in ("LOW", "MEDIUM", "HIGH"))
    # Count from final ledger
    ft_count = len([r for r in final_ledger if r["classification"] == "ft"])
    assert total == ft_count, f"Risk tier counts sum to {total}, but FT fixtures = {ft_count}"


# ─── Test 24: All actual results from OddAlerts are FT ───────────────────────

def test_24_all_raw_results_are_ft(raw_results):
    """All raw OddAlerts results for Phase 15 fixtures must be FT status."""
    non_ft = [r for r in raw_results if r.get("status") != "FT"]
    assert non_ft == [], f"Non-FT results found: {[(r['id'], r['status']) for r in non_ft]}"


# ─── Test 25: Dashboard HTML file is non-empty and well-formed ───────────────

def test_25_dashboard_html_exists_and_valid():
    """Dashboard HTML must exist and contain key structural elements."""
    path = os.path.join(AUDIT_DIR, "08_phase15_final_dashboard.html")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 5000, "Dashboard HTML too small"
    assert "Phase 15 Final Results Audit" in content
    assert FROZEN_SHA256 in content
    assert "<table" in content
    assert "1X2 Accuracy" in content or "Accuracy" in content


# ─── Test 26: Markdown report contains all required sections ──────────────────

def test_26_markdown_report_has_required_sections():
    """Final markdown report must contain all key section headers."""
    path = os.path.join(AUDIT_DIR, "07_phase15_final_report.md")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    required_sections = [
        "Executive Summary",
        "1X2 Accuracy",
        "Exact Score",
        "Brier Score",
        "Log Loss",
        "Draw Risk",
        "2-0 Signal",
        "Strong Home Signal",
        "League Breakdown",
        "Biggest Misses",
        "Model Change Decision",
        FROZEN_SHA256,
    ]
    for sec in required_sections:
        assert sec in content, f"Missing section/content in report: '{sec}'"


# ─── Test 27: No predictions were modified (pred_outcome matches ledger) ──────

def test_27_pred_outcome_unchanged_from_ledger(final_ledger, prediction_ledger):
    """PRED outcome must not be modified from the immutable ledger for any fixture."""
    pred_by_fid = {str(r["fixture_id"]): r for r in prediction_ledger}
    for r in final_ledger:
        fid = r["fixture_id"]
        pred = pred_by_fid.get(fid)
        assert r["pred_outcome"] == pred["PRED"], (
            f"Fixture {fid}: pred_outcome was modified! "
            f"Got {r['pred_outcome']}, expected {pred['PRED']}"
        )


# ─── Test 28: Strong home profile count is correct ───────────────────────────

def test_28_strong_home_profile_fixtures(signal_analysis, final_ledger):
    """Strong home profile count in signal analysis must match ledger count."""
    sh_from_ledger = [r for r in final_ledger if r.get("strong_home_profile")]
    sh_from_signal = signal_analysis["G_strong_home_signal"]["fixtures"]
    assert sh_from_signal == len(sh_from_ledger), (
        f"Strong home count mismatch: signal={sh_from_signal}, ledger={len(sh_from_ledger)}"
    )


# ─── Test 29: 2-0 profiles signal shows outcome for both flagged fixtures ──────

def test_29_2_0_profile_details_are_complete(signal_analysis):
    """Both 2-0 profile records must have outcome_ok and exact_ok fields."""
    for d in signal_analysis["F_2_0_signal"]["details"]:
        assert "outcome_ok" in d
        assert "exact_ok" in d
        assert "actual" in d
        assert "lambda_home" in d
        assert "lambda_away" in d


# ─── Test 30: Frozen SHA256 in metrics meta ───────────────────────────────────

def test_30_frozen_sha256_in_metrics_meta(metrics):
    """Metrics meta must embed the exact frozen V4 SHA256."""
    assert metrics["meta"]["frozen_v4_sha256"] == FROZEN_SHA256
    assert metrics["meta"]["sportmonks_excluded"] is True
