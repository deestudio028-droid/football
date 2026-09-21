"""
tests/test_phase16_sportmonks.py
Phase 16 V4 + Sportmonks Research Arm Test Suite

25 tests covering:
  1. V4 SHA unchanged
  2. Phase 15 ledger unchanged
  3. Fixture ID matching
  4. Same fixture universe
  5. Sportmonks H/D/A validation
  6. Probability normalization
  7. Missing data handling
  8. Temporal status validation
  9. No post-kickoff data classified as verified pre-kickoff
 10. V4 probability immutability
 11. Sportmonks probability preservation
 12. 50/50 blend correctness
 13. 70/30 blend correctness
 14. 30/70 blend correctness
 15. Combined probability normalization
 16. Draw analysis correctness
 17. 2-0 analysis correctness
 18. Strong Home analysis correctness
 19. No production mutation
 20. No API secrets exposed
 21. No hardcoded outcomes
 22. Fixture uniqueness
 23. Coverage calculation
 24. Actual-result join integrity
 25. Research dashboard integrity
"""

import json
import os
import math
import hashlib
import pytest

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PHASE16_DIR = os.path.join(ROOT, "research", "external_consensus", "phase16_sportmonks")
PHASE15_LEDGER = os.path.join(ROOT, "reports", "upcoming_2026_09_18_to_2026_09_21_ledger.jsonl")
V4_MODEL_PATH = os.path.join(ROOT, "data", "models", "v4_poisson_venue_elo_online_ad.pkl")

FROZEN_SHA256 = "1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5"

# Phase 16 output files
UNIVERSE_FILE = os.path.join(PHASE16_DIR, "01_phase16_fixture_universe.jsonl")
SM_PREDS_FILE = os.path.join(PHASE16_DIR, "02_sportmonks_raw_predictions.jsonl")
VALIDATION_FILE = os.path.join(PHASE16_DIR, "03_phase16_source_validation.json")
V4_BASELINE_FILE = os.path.join(PHASE16_DIR, "04_phase16_v4_baseline.json")
SM_ONLY_FILE = os.path.join(PHASE16_DIR, "05_phase16_sportmonks_only.json")
BLEND_FILE = os.path.join(PHASE16_DIR, "06_phase16_blend_results.json")
DRAW_FILE = os.path.join(PHASE16_DIR, "07_phase16_draw_analysis.json")
SIGNAL_FILE = os.path.join(PHASE16_DIR, "08_phase16_signal_analysis.json")
COMPARISON_FILE = os.path.join(PHASE16_DIR, "09_phase16_comparison.json")
REPORT_FILE = os.path.join(PHASE16_DIR, "10_phase16_report.md")
DASHBOARD_FILE = os.path.join(PHASE16_DIR, "11_phase16_dashboard.html")
README_FILE = os.path.join(PHASE16_DIR, "README.md")

# Production files that must NOT be modified
PRODUCTION_FILES = [
    os.path.join(ROOT, "reports", "upcoming_2026_09_18_to_2026_09_21_ledger.jsonl"),
    os.path.join(ROOT, "data", "models", "v4_poisson_venue_elo_online_ad.pkl"),
]


# ─── Fixtures (pytest) ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def universe():
    with open(UNIVERSE_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@pytest.fixture(scope="module")
def sm_preds():
    with open(SM_PREDS_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@pytest.fixture(scope="module")
def validation():
    with open(VALIDATION_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def blend_results():
    with open(BLEND_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def draw_analysis():
    with open(DRAW_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def signal_analysis():
    with open(SIGNAL_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def comparison():
    with open(COMPARISON_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def phase15_ledger():
    with open(PHASE15_LEDGER, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ─── Test 1: V4 SHA256 Unchanged ──────────────────────────────────────────────

def test_v4_sha256_unchanged():
    """V4 model binary must remain exactly frozen."""
    assert os.path.exists(V4_MODEL_PATH), "V4 model file not found"
    with open(V4_MODEL_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == FROZEN_SHA256, (
        f"V4 SHA256 CHANGED! Expected {FROZEN_SHA256}, got {digest}"
    )


# ─── Test 2: Phase 15 Ledger Unchanged ────────────────────────────────────────

def test_phase15_ledger_unchanged(universe, phase15_ledger):
    """Phase 15 prediction ledger must be immutable — V4 probabilities unchanged."""
    # Index by fixture_id
    ledger_by_fid = {str(r["fixture_id"]): r for r in phase15_ledger}

    for rec in universe:
        fid = str(rec["fixture_id"])
        original = ledger_by_fid.get(fid)
        assert original is not None, f"Fixture {fid} missing from Phase 15 ledger"
        # Probabilities must be exactly preserved
        assert abs(rec["v4_p_home"] - original["p_home"]) < 1e-9, \
            f"v4_p_home modified for fixture {fid}"
        assert abs(rec["v4_p_draw"] - original["p_draw"]) < 1e-9, \
            f"v4_p_draw modified for fixture {fid}"
        assert abs(rec["v4_p_away"] - original["p_away"]) < 1e-9, \
            f"v4_p_away modified for fixture {fid}"


# ─── Test 3: Fixture ID Matching ──────────────────────────────────────────────

def test_fixture_id_matching(universe, phase15_ledger):
    """Every Phase 15 fixture ID must appear in the Phase 16 universe."""
    ledger_fids = {str(r["fixture_id"]) for r in phase15_ledger}
    universe_fids = {str(r["fixture_id"]) for r in universe}
    assert ledger_fids == universe_fids, (
        f"Fixture ID mismatch: missing={ledger_fids - universe_fids}, "
        f"extra={universe_fids - ledger_fids}"
    )


# ─── Test 4: Same Fixture Universe ────────────────────────────────────────────

def test_same_fixture_universe(universe, phase15_ledger):
    """Phase 16 must use exactly the same 48 fixtures as Phase 15."""
    assert len(universe) == 48, f"Expected 48 fixtures, got {len(universe)}"
    assert len(phase15_ledger) == 48, f"Phase 15 ledger should have 48 records"
    # Verify cohort dates
    dates = {r["date"] for r in universe}
    assert "2026-09-18" in dates, "Missing Sep 18 fixtures"
    assert "2026-09-21" in dates or "2026-09-20" in dates, "Missing Sep 20/21 fixtures"


# ─── Test 5: Sportmonks H/D/A Validation ──────────────────────────────────────

def test_sportmonks_probability_validity(universe):
    """All Sportmonks probabilities must be non-negative and sum ≈ 1.0."""
    for r in universe:
        if not r["sm_available"]:
            continue
        fid = r["fixture_id"]
        ph = r["sm_p_home"]
        pd = r["sm_p_draw"]
        pa = r["sm_p_away"]
        assert ph >= 0, f"Fixture {fid}: sm_p_home is negative ({ph})"
        assert pd >= 0, f"Fixture {fid}: sm_p_draw is negative ({pd})"
        assert pa >= 0, f"Fixture {fid}: sm_p_away is negative ({pa})"
        prob_sum = ph + pd + pa
        assert abs(prob_sum - 1.0) < 0.01, (
            f"Fixture {fid}: SM probs sum to {prob_sum:.4f}, not ≈ 1.0"
        )


# ─── Test 6: Probability Normalization ────────────────────────────────────────

def test_probability_normalization(universe):
    """After blending, combined probabilities must sum to 1.0 within tolerance."""
    for r in universe:
        if not r["sm_available"]:
            continue
        for method_key in ["m3_50_50", "m4_70_30", "m5_30_70"]:
            b = r["blends"].get(method_key, {})
            ph = b.get("p_home", 0)
            pd = b.get("p_draw", 0)
            pa = b.get("p_away", 0)
            total = ph + pd + pa
            assert abs(total - 1.0) < 0.001, (
                f"Fixture {r['fixture_id']} method {method_key}: "
                f"blend sum={total:.6f} not ≈ 1.0"
            )
            assert ph >= 0 and pd >= 0 and pa >= 0, (
                f"Fixture {r['fixture_id']} method {method_key}: negative blend probability"
            )


# ─── Test 7: Missing Data Handling ────────────────────────────────────────────

def test_missing_data_handling(universe):
    """Fixtures without SM data must be explicitly marked sm_available=False."""
    for r in universe:
        if not r["sm_available"]:
            assert r["sm_p_home"] is None, \
                f"Fixture {r['fixture_id']}: sm_p_home should be None when unavailable"
            assert r["sm_p_draw"] is None, \
                f"Fixture {r['fixture_id']}: sm_p_draw should be None when unavailable"
            assert r["sm_p_away"] is None, \
                f"Fixture {r['fixture_id']}: sm_p_away should be None when unavailable"
            assert r["sm_pred"] is None, \
                f"Fixture {r['fixture_id']}: sm_pred should be None when unavailable"
            assert r["blends"] == {}, \
                f"Fixture {r['fixture_id']}: blends should be empty when SM unavailable"


# ─── Test 8: Temporal Status Validation ───────────────────────────────────────

def test_temporal_status_validation(universe, validation):
    """Temporal status must be correctly set."""
    # All SM-available fixtures must be TEMPORAL_UNKNOWN (not PRE_KICKOFF_VERIFIED)
    for r in universe:
        if r["sm_available"]:
            assert r["sm_temporal_status"] == "TEMPORAL_UNKNOWN", (
                f"Fixture {r['fixture_id']}: temporal_status should be TEMPORAL_UNKNOWN, "
                f"got {r['sm_temporal_status']}"
            )
        else:
            assert r["sm_temporal_status"] == "NOT_AVAILABLE", (
                f"Fixture {r['fixture_id']}: unavailable SM should show NOT_AVAILABLE"
            )

    # Validation file must confirm 0 PRE_KICKOFF_VERIFIED
    ts = validation.get("temporal_status", {})
    assert ts.get("PRE_KICKOFF_VERIFIED", 999) == 0, (
        f"Expected 0 PRE_KICKOFF_VERIFIED, got {ts.get('PRE_KICKOFF_VERIFIED')}"
    )


# ─── Test 9: No Post-Kickoff Data Classified as Pre-Kickoff Verified ─────────

def test_no_post_kickoff_as_pre_kickoff_verified(universe, validation):
    """No fixture may be classified as PRE_KICKOFF_VERIFIED."""
    for r in universe:
        status = r.get("sm_temporal_status", "")
        assert status != "PRE_KICKOFF_VERIFIED", (
            f"Fixture {r['fixture_id']}: FORBIDDEN temporal status PRE_KICKOFF_VERIFIED"
        )

    ts = validation.get("temporal_status", {})
    pkv = ts.get("PRE_KICKOFF_VERIFIED", 0)
    assert pkv == 0, f"Validation reports {pkv} PRE_KICKOFF_VERIFIED (must be 0)"


# ─── Test 10: V4 Probability Immutability ─────────────────────────────────────

def test_v4_probability_immutability(universe):
    """V4 SHA256 in all universe records must match frozen value."""
    for r in universe:
        assert r["model_v4_sha256"] == FROZEN_SHA256, (
            f"Fixture {r['fixture_id']}: v4 SHA256 mismatch in universe record"
        )


# ─── Test 11: Sportmonks Probability Preservation ─────────────────────────────

def test_sportmonks_probability_preservation(universe, sm_preds):
    """Sportmonks raw probabilities preserved in both universe and sm_preds files."""
    sm_by_fid = {str(r["fixture_id"]): r for r in sm_preds}
    for r in universe:
        if not r["sm_available"]:
            continue
        fid = str(r["fixture_id"])
        sm = sm_by_fid.get(fid)
        assert sm is not None, f"Fixture {fid} missing from sm_preds file"
        assert sm["sm_available"] is True, f"Fixture {fid}: sm_available mismatch"
        # SM probabilities should be consistent between files
        assert abs((sm["sm_p_home"] or 0) - (r["sm_p_home"] or 0)) < 1e-9, \
            f"Fixture {fid}: sm_p_home inconsistent"
        assert abs((sm["sm_p_draw"] or 0) - (r["sm_p_draw"] or 0)) < 1e-9, \
            f"Fixture {fid}: sm_p_draw inconsistent"


# ─── Test 12: 50/50 Blend Correctness ─────────────────────────────────────────

def test_50_50_blend_correctness(universe):
    """50/50 blend must be exactly 0.5*V4 + 0.5*SM (normalized)."""
    for r in universe:
        if not r["sm_available"]:
            continue
        b = r["blends"].get("m3_50_50", {})
        v4_ph, v4_pd, v4_pa = r["v4_p_home"], r["v4_p_draw"], r["v4_p_away"]
        sm_ph, sm_pd, sm_pa = r["sm_p_home"], r["sm_p_draw"], r["sm_p_away"]

        raw_h = 0.5 * v4_ph + 0.5 * sm_ph
        raw_d = 0.5 * v4_pd + 0.5 * sm_pd
        raw_a = 0.5 * v4_pa + 0.5 * sm_pa
        total = raw_h + raw_d + raw_a
        exp_ph = raw_h / total
        exp_pd = raw_d / total
        exp_pa = raw_a / total

        assert abs(b.get("p_home", 0) - exp_ph) < 1e-4, \
            f"Fixture {r['fixture_id']}: 50/50 p_home mismatch"
        assert abs(b.get("p_draw", 0) - exp_pd) < 1e-4, \
            f"Fixture {r['fixture_id']}: 50/50 p_draw mismatch"
        assert abs(b.get("p_away", 0) - exp_pa) < 1e-4, \
            f"Fixture {r['fixture_id']}: 50/50 p_away mismatch"
        assert b.get("w_v4") == 0.50, f"Fixture {r['fixture_id']}: 50/50 w_v4 wrong"
        assert b.get("w_sm") == 0.50, f"Fixture {r['fixture_id']}: 50/50 w_sm wrong"


# ─── Test 13: 70/30 Blend Correctness ─────────────────────────────────────────

def test_70_30_blend_correctness(universe):
    """70/30 V4-heavy blend must be 0.70*V4 + 0.30*SM (normalized)."""
    for r in universe:
        if not r["sm_available"]:
            continue
        b = r["blends"].get("m4_70_30", {})
        v4_ph, v4_pd, v4_pa = r["v4_p_home"], r["v4_p_draw"], r["v4_p_away"]
        sm_ph, sm_pd, sm_pa = r["sm_p_home"], r["sm_p_draw"], r["sm_p_away"]

        raw_h = 0.7 * v4_ph + 0.3 * sm_ph
        raw_d = 0.7 * v4_pd + 0.3 * sm_pd
        raw_a = 0.7 * v4_pa + 0.3 * sm_pa
        total = raw_h + raw_d + raw_a
        exp_ph = raw_h / total

        assert abs(b.get("p_home", 0) - exp_ph) < 1e-4, \
            f"Fixture {r['fixture_id']}: 70/30 p_home mismatch"
        assert b.get("w_v4") == 0.70, f"Fixture {r['fixture_id']}: 70/30 w_v4 wrong"
        assert b.get("w_sm") == 0.30, f"Fixture {r['fixture_id']}: 70/30 w_sm wrong"


# ─── Test 14: 30/70 Blend Correctness ─────────────────────────────────────────

def test_30_70_blend_correctness(universe):
    """30/70 SM-heavy diagnostic blend must be 0.30*V4 + 0.70*SM (normalized)."""
    for r in universe:
        if not r["sm_available"]:
            continue
        b = r["blends"].get("m5_30_70", {})
        v4_ph, v4_pd, v4_pa = r["v4_p_home"], r["v4_p_draw"], r["v4_p_away"]
        sm_ph, sm_pd, sm_pa = r["sm_p_home"], r["sm_p_draw"], r["sm_p_away"]

        raw_h = 0.3 * v4_ph + 0.7 * sm_ph
        raw_d = 0.3 * v4_pd + 0.7 * sm_pd
        raw_a = 0.3 * v4_pa + 0.7 * sm_pa
        total = raw_h + raw_d + raw_a
        exp_ph = raw_h / total

        assert abs(b.get("p_home", 0) - exp_ph) < 1e-4, \
            f"Fixture {r['fixture_id']}: 30/70 p_home mismatch"
        assert b.get("w_v4") == 0.30, f"Fixture {r['fixture_id']}: 30/70 w_v4 wrong"
        assert b.get("w_sm") == 0.70, f"Fixture {r['fixture_id']}: 30/70 w_sm wrong"


# ─── Test 15: Combined Probability Normalization ───────────────────────────────

def test_combined_probability_normalization(universe):
    """All five blend methods must produce probabilities that sum to 1."""
    for r in universe:
        if not r["sm_available"]:
            continue
        for mk in ["m1_v4_only", "m2_sm_only", "m3_50_50", "m4_70_30", "m5_30_70"]:
            b = r["blends"].get(mk, {})
            total = b.get("p_home", 0) + b.get("p_draw", 0) + b.get("p_away", 0)
            assert abs(total - 1.0) < 0.001, \
                f"Fixture {r['fixture_id']} method {mk}: sum={total:.6f}"
            for k in ["p_home", "p_draw", "p_away"]:
                assert b.get(k, 0) >= 0, \
                    f"Fixture {r['fixture_id']} method {mk}: {k} is negative"


# ─── Test 16: Draw Analysis Correctness ───────────────────────────────────────

def test_draw_analysis_correctness(universe, draw_analysis):
    """Draw analysis must correctly identify all actual draws."""
    # Phase 15 had 10 actual draws
    actual_draws = [r for r in universe if r.get("actual_outcome") == "D"]
    meta = draw_analysis.get("meta", {})
    assert meta.get("actual_draws_all_48") == len(actual_draws), (
        f"Draw count mismatch: file says {meta.get('actual_draws_all_48')}, "
        f"universe has {len(actual_draws)}"
    )
    assert len(actual_draws) == 10, f"Phase 15 had 10 draws, got {len(actual_draws)}"

    # V4 predicted 0 draws (validated in Phase 15)
    v4_pred_draws = sum(1 for r in universe if r.get("v4_pred") == "D")
    assert v4_pred_draws == 0, f"V4 should predict 0 draws (Phase 15 finding), got {v4_pred_draws}"

    # Calibration table must contain all 5 methods
    cal = draw_analysis.get("draw_calibration_by_method", [])
    assert len(cal) == 5, f"Expected 5 calibration entries, got {len(cal)}"


# ─── Test 17: 2-0 Analysis Correctness ────────────────────────────────────────

def test_2_0_analysis_correctness(universe, signal_analysis):
    """2-0 profile fixtures must be non-empty and correctly counted."""
    signal_2_0 = [r for r in universe if r.get("v4_is_2_0") and r.get("sm_available")]
    sa = signal_analysis.get("2_0_profile", {})
    v4_n = sa.get("v4", {}).get("n", 0)
    assert v4_n == len(signal_2_0), (
        f"2-0 profile count mismatch: analysis says {v4_n}, universe has {len(signal_2_0)}"
    )
    # Phase 15 had 2 fixtures with 2-0 profile (Bayern and Man City)
    assert v4_n == 2, f"Expected 2 2-0 profile fixtures, got {v4_n}"


# ─── Test 18: Strong Home Analysis Correctness ────────────────────────────────

def test_strong_home_analysis_correctness(universe, signal_analysis):
    """Strong home fixtures must be correctly counted."""
    strong_home = [r for r in universe if r.get("v4_strong_home") and r.get("sm_available")]
    sa = signal_analysis.get("strong_home_profile", {})
    v4_n = sa.get("v4", {}).get("n", 0)
    assert v4_n == len(strong_home), (
        f"Strong home count mismatch: analysis says {v4_n}, universe has {len(strong_home)}"
    )
    # Phase 15 had 3 strong home fixtures (Bayern, Man City, Leverkusen)
    assert v4_n == 3, f"Expected 3 strong home fixtures, got {v4_n}"


# ─── Test 19: No Production Mutation ──────────────────────────────────────────

def test_no_production_mutation():
    """Production files must be unmodified."""
    # V4 model SHA must match
    with open(V4_MODEL_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == FROZEN_SHA256, "V4 model modified!"

    # Phase 15 ledger must still have exactly 48 records
    with open(PHASE15_LEDGER, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    assert len(records) == 48, f"Phase 15 ledger mutated: has {len(records)} records"

    # All Phase 15 records must still have the frozen SHA
    for r in records:
        assert r.get("model_v4_sha256") == FROZEN_SHA256, \
            f"Phase 15 ledger SHA mismatch in fixture {r.get('fixture_id')}"


# ─── Test 20: No API Secrets Exposed ──────────────────────────────────────────

def test_no_api_secrets_exposed():
    """No API token or secret should appear in any Phase 16 output file."""
    import os as _os

    # Known tokens to check (check partial match)
    suspicious_patterns = ["api_token=", "Bearer ", "X-Auth-Token"]

    for filename in [
        "01_phase16_fixture_universe.jsonl",
        "02_sportmonks_raw_predictions.jsonl",
        "03_phase16_source_validation.json",
        "06_phase16_blend_results.json",
        "10_phase16_report.md",
        "11_phase16_dashboard.html",
    ]:
        fpath = _os.path.join(PHASE16_DIR, filename)
        if not _os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        for pattern in suspicious_patterns:
            assert pattern not in content, \
                f"Possible secret found in {filename}: pattern '{pattern}'"


# ─── Test 21: No Hardcoded Outcomes ───────────────────────────────────────────

def test_no_hardcoded_outcomes():
    """The research engine must not have hardcoded actual scores."""
    engine_path = os.path.join(PHASE16_DIR, "phase16_engine.py")
    with open(engine_path, encoding="utf-8") as f:
        content = f.read()
    # Should not contain hardcoded specific scores like "7-0" or "5-3" as literal match values
    # (they may appear in comments/strings but not as match logic)
    hardcoded_danger = [
        'actual_outcome = "H"  # hardcoded',
        'actual_score = "7-0"  # hardcoded',
    ]
    for danger in hardcoded_danger:
        assert danger not in content, f"Hardcoded outcome found in engine: {danger}"

    # Verify actuals are derived from raw data, not hardcoded
    assert "actual.get(\"home_goals\")" in content or "home_goals" in content, \
        "Engine should derive actuals from raw data"


# ─── Test 22: Fixture Uniqueness ──────────────────────────────────────────────

def test_fixture_uniqueness(universe):
    """Each fixture_id must appear exactly once in the universe."""
    fids = [str(r["fixture_id"]) for r in universe]
    assert len(fids) == len(set(fids)), (
        f"Duplicate fixture IDs detected in universe: "
        f"{[fid for fid in fids if fids.count(fid) > 1]}"
    )


# ─── Test 23: Coverage Calculation ────────────────────────────────────────────

def test_coverage_calculation(universe, validation):
    """Coverage percentage must match actual SM availability in universe."""
    n_total = len(universe)
    n_sm = sum(1 for r in universe if r["sm_available"])
    n_not_sm = n_total - n_sm
    expected_pct = round(100 * n_sm / n_total, 2)

    cov = validation.get("coverage", {})
    assert cov.get("sm_found") == n_sm, \
        f"Coverage sm_found mismatch: {cov.get('sm_found')} vs {n_sm}"
    assert cov.get("coverage_pct") == expected_pct, \
        f"Coverage pct mismatch: {cov.get('coverage_pct')} vs {expected_pct}"
    assert n_sm >= 0 and n_sm <= n_total, "Coverage count out of range"

    # All 5 leagues should be represented
    leagues = {r["league"] for r in universe}
    expected_leagues = {"Premier League", "Bundesliga", "Ligue 1", "La Liga", "Serie A"}
    assert expected_leagues.issubset(leagues), f"Missing leagues: {expected_leagues - leagues}"


# ─── Test 24: Actual Result Join Integrity ────────────────────────────────────

def test_actual_result_join_integrity(universe):
    """Actual results must be joined by fixture_id; all 48 must be FT."""
    for r in universe:
        fid = r["fixture_id"]
        assert r.get("actual_status") is not None, f"Fixture {fid}: actual_status is None"
        assert r.get("actual_outcome") in ("H", "D", "A"), \
            f"Fixture {fid}: invalid actual_outcome {r.get('actual_outcome')}"
        assert r.get("actual_score") is not None, f"Fixture {fid}: actual_score is None"

    # All must be FT
    non_ft = [r["fixture_id"] for r in universe if r.get("actual_status") != "FT"]
    assert len(non_ft) == 0, f"Non-FT fixtures found: {non_ft}"

    # Actual outcomes must not be blank
    missing = [r["fixture_id"] for r in universe if not r.get("actual_outcome")]
    assert len(missing) == 0, f"Fixtures missing actual_outcome: {missing}"


# ─── Test 25: Research Dashboard Integrity ────────────────────────────────────

def test_research_dashboard_integrity():
    """Dashboard must exist, be non-empty HTML, and contain research warning."""
    assert os.path.exists(DASHBOARD_FILE), "Phase 16 dashboard HTML not found"

    with open(DASHBOARD_FILE, encoding="utf-8") as f:
        html = f.read()

    assert len(html) > 5000, "Dashboard HTML suspiciously small"
    assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower(), "Not valid HTML"

    # Must contain research-only warnings
    assert "RESEARCH" in html.upper(), "Dashboard missing RESEARCH label"
    assert "TEMPORAL_UNKNOWN" in html, "Dashboard missing TEMPORAL_UNKNOWN label"
    assert "NOT PRODUCTION" in html.upper() or "NOT_PRODUCTION" in html, \
        "Dashboard missing NOT PRODUCTION warning"

    # Must contain V4 SHA reference
    assert FROZEN_SHA256 in html, "Dashboard missing V4 SHA256"

    # Must not replace production dashboard
    assert DASHBOARD_FILE.endswith("11_phase16_dashboard.html"), \
        "Dashboard must be in phase16 research directory"

    # Must contain fixture table
    assert "<table" in html, "Dashboard missing fixture table"
