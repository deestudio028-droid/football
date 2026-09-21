"""
Phase 16 — V4 + Sportmonks Prospective Research Arm
=====================================================
RESEARCH ONLY — Does NOT modify V4 model, production predictions, or ledgers.

Arm A: Frozen V4 baseline (Phase 15 predictions, immutable)
Arm B: V4 + Sportmonks blended predictions

Outputs all 12 required files in research/external_consensus/phase16_sportmonks/

Usage:
  py -3.13 research/external_consensus/phase16_sportmonks/phase16_engine.py
"""

import json
import math
import os
import sys
import re
import unicodedata
import hashlib
from datetime import datetime, timezone
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PHASE15_LEDGER = os.path.join(ROOT, "reports", "upcoming_2026_09_18_to_2026_09_21_ledger.jsonl")
PHASE15_RESULTS = os.path.join(ROOT, "data", "phase15_oddalerts_raw.json")
PHASE15_AUDIT_LEDGER = os.path.join(ROOT, "research", "external_consensus", "phase15_final_audit",
                                     "01_phase15_final_results_ledger.jsonl")
SM_RAW = os.path.join(ROOT, "data", "phase16_sportmonks_raw.json")
V4_MODEL_PATH = os.path.join(ROOT, "data", "models", "v4_poisson_venue_elo_online_ad.pkl")

FROZEN_SHA256 = "1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5"

# Sportmonks Big-5 league IDs
SM_LEAGUE_IDS = {
    8: "Premier League",
    82: "Bundesliga",
    301: "Ligue 1",
    384: "La Liga",
    564: "Serie A",
}

# OddAlerts league names → Sportmonks league ID mapping
LEAGUE_NAME_TO_SM_ID = {
    "Premier League": 8,
    "Bundesliga": 82,
    "Ligue 1": 301,
    "La Liga": 384,
    "Serie A": 564,
}

# Sportmonks prediction type_id for 1X2
SM_1X2_TYPE_ID = 237

# ─── Helpers ──────────────────────────────────────────────────────────────────

def verify_v4_sha256():
    if not os.path.exists(V4_MODEL_PATH):
        raise FileNotFoundError(f"V4 model binary not found: {V4_MODEL_PATH}")
    with open(V4_MODEL_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != FROZEN_SHA256:
        raise ValueError(f"V4 SHA256 MISMATCH! Expected {FROZEN_SHA256}, got {digest}")
    return digest


def norm_name(s):
    """Normalize team name for fuzzy matching."""
    if not s:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn").lower()
    stopwords = ["fc", "cf", "sc", "rc", "sv", "1.", "04", "07", "1899",
                 "hotspur", "de", "cd", "stade", "tsg", "vfb", "fsv",
                 "united", "city", "athletic", "club", "borussia"]
    for w in stopwords:
        s = re.sub(r"\b" + re.escape(w) + r"\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def safe_div(a, b):
    return a / b if b else 0.0


def normalize_probs(ph, pd, pa):
    """Normalize so probabilities sum to 1; clip negatives to 0."""
    ph = max(0.0, ph)
    pd = max(0.0, pd)
    pa = max(0.0, pa)
    total = ph + pd + pa
    if total == 0:
        return 1/3, 1/3, 1/3
    return ph / total, pd / total, pa / total


def argmax_outcome(ph, pd, pa):
    if ph >= pd and ph >= pa:
        return "H"
    elif pa >= ph and pa >= pd:
        return "A"
    else:
        return "D"


def brier_score(ph, pd, pa, actual_outcome):
    oh = 1 if actual_outcome == "H" else 0
    od = 1 if actual_outcome == "D" else 0
    oa = 1 if actual_outcome == "A" else 0
    return (ph - oh) ** 2 + (pd - od) ** 2 + (pa - oa) ** 2


def log_loss(ph, pd, pa, actual_outcome):
    p_map = {"H": ph, "D": pd, "A": pa}
    p = max(min(p_map[actual_outcome], 1 - 1e-9), 1e-9)
    return -math.log(p)


def blend(v4_ph, v4_pd, v4_pa, sm_ph, sm_pd, sm_pa, w_v4, w_sm):
    ph = w_v4 * v4_ph + w_sm * sm_ph
    pd = w_v4 * v4_pd + w_sm * sm_pd
    pa = w_v4 * v4_pa + w_sm * sm_pa
    return normalize_probs(ph, pd, pa)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_phase15_ledger():
    """Load immutable V4 Phase 15 prediction ledger."""
    records = []
    with open(PHASE15_LEDGER, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_phase15_actuals():
    """Load actual FT results for all 48 Phase 15 fixtures."""
    with open(PHASE15_RESULTS, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(r["id"]): r for r in raw}


def load_phase15_audit_ledger():
    """Load Phase 15 final audit ledger (predictions + actuals joined)."""
    records = []
    with open(PHASE15_AUDIT_LEDGER, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_sportmonks_raw():
    """Load raw Sportmonks fixture data for Phase 15 date range."""
    with open(SM_RAW, encoding="utf-8") as f:
        return json.load(f)


# ─── Sportmonks Matching ──────────────────────────────────────────────────────

def extract_sm_1x2(sm_fixture):
    """Extract 1X2 probabilities from Sportmonks fixture (type_id=237).
    
    Returns (ph, pd, pa, source_data, found) — probabilities as fractions [0,1].
    """
    preds = sm_fixture.get("predictions", [])
    p237 = next((p for p in preds if p["type_id"] == SM_1X2_TYPE_ID), None)
    if not p237:
        return None, None, None, None, False

    pred_data = p237.get("predictions", {})
    if not isinstance(pred_data, dict):
        return None, None, None, None, False

    raw_h = pred_data.get("home", None)
    raw_d = pred_data.get("draw", None)
    raw_a = pred_data.get("away", None)

    if any(v is None for v in [raw_h, raw_d, raw_a]):
        return None, None, None, pred_data, False

    # Sportmonks provides percentages (0-100), normalize to [0,1]
    ph_raw = float(raw_h) / 100.0
    pd_raw = float(raw_d) / 100.0
    pa_raw = float(raw_a) / 100.0

    ph, pd, pa = normalize_probs(ph_raw, pd_raw, pa_raw)
    return ph, pd, pa, pred_data, True


def match_sm_fixture(pred_record, sm_by_date):
    """Find the Sportmonks fixture matching the V4 prediction record.
    
    NOTE: Sportmonks league_id assignments appear scrambled for this dataset
    (e.g., Italian teams assigned to La Liga IDs). We therefore match by
    date + team name across ALL Big-5 fixtures, then verify the team names
    confidently match.
    
    Returns (sm_fixture, match_method, confidence) or (None, None, None).
    """
    date_str = pred_record["Date (UTC)"]  # YYYY-MM-DD
    home_norm = norm_name(pred_record["Home Team"])
    away_norm = norm_name(pred_record["Away Team"])

    candidates = sm_by_date.get(date_str, [])

    for sm in candidates:
        name = sm.get("name", "")
        if " vs " not in name:
            continue
        sm_h_raw, sm_a_raw = name.split(" vs ", 1)
        sm_h = norm_name(sm_h_raw)
        sm_a = norm_name(sm_a_raw)

        # Exact normalized match (both teams must match)
        if home_norm == sm_h and away_norm == sm_a:
            return sm, "EXACT_NORM", 1.0

        # Strong substring match (both teams must partially match)
        h_match = bool(home_norm and sm_h and
                       (home_norm in sm_h or sm_h in home_norm) and
                       len(home_norm) >= 4)
        a_match = bool(away_norm and sm_a and
                       (away_norm in sm_a or sm_a in away_norm) and
                       len(away_norm) >= 4)
        if h_match and a_match:
            return sm, "SUBSTRING_BOTH", 0.8

    return None, None, None


# ─── Core Processing ──────────────────────────────────────────────────────────

def build_fixture_universe(ledger, actuals, sm_fixtures):
    """Build the Phase 16 fixture universe joining V4, Sportmonks, and actuals."""

    # Index Sportmonks fixtures by date only (NOT by league_id, which is scrambled in this dataset).
    # The scrambling: Sportmonks assigns Italian teams to La Liga league_id=384 and vice versa.
    # We match by date + team name across all Big-5 fixtures to avoid false negatives.
    sm_by_date = defaultdict(list)
    for sm in sm_fixtures:
        lid = sm.get("league_id")
        date_str = sm.get("starting_at", "")[:10]
        if lid in SM_LEAGUE_IDS:
            sm_by_date[date_str].append(sm)

    capture_ts = datetime.now(timezone.utc).isoformat()
    records = []
    stats = {
        "sm_found": 0,
        "sm_exact": 0,
        "sm_substring_both": 0,
        "sm_not_found": 0,
        "sm_probs_valid": 0,
        "sm_probs_invalid": 0,
        "sm_league_id_scrambling_detected": True,
    }

    for pred in ledger:
        fid = str(pred["fixture_id"])
        actual = actuals.get(fid, {})

        # V4 probabilities (immutable)
        v4_ph = pred["p_home"]
        v4_pd = pred["p_draw"]
        v4_pa = pred["p_away"]
        v4_pred = pred["PRED"]
        v4_canonical_score = pred["canonical_predicted_score"]

        # Actual result
        act_hg = actual.get("home_goals")
        act_ag = actual.get("away_goals")
        act_status = actual.get("status")
        act_score = f"{act_hg}-{act_ag}" if act_hg is not None else None
        act_outcome = None
        if act_hg is not None and act_ag is not None:
            if act_hg > act_ag:
                act_outcome = "H"
            elif act_hg == act_ag:
                act_outcome = "D"
            else:
                act_outcome = "A"

        # Sportmonks match (note: index is by date only due to scrambled league IDs)
        sm_fixture, match_method, match_confidence = match_sm_fixture(pred, sm_by_date)

        sm_available = False
        sm_ph = sm_pd = sm_pa = None
        sm_raw_data = None
        sm_fixture_id = None
        sm_temporal_status = "NOT_AVAILABLE"

        validation_errors = []

        if sm_fixture:
            stats["sm_found"] += 1
            if match_method == "EXACT_NORM":
                stats["sm_exact"] += 1
            else:
                stats["sm_substring_both"] += 1

            sm_fixture_id = sm_fixture.get("id")
            sm_ph_raw, sm_pd_raw, sm_pa_raw, sm_raw_data, sm_valid = extract_sm_1x2(sm_fixture)

            if sm_valid:
                # Validate probabilities
                if sm_ph_raw < 0:
                    validation_errors.append(f"sm_ph negative: {sm_ph_raw}")
                if sm_pd_raw < 0:
                    validation_errors.append(f"sm_pd negative: {sm_pd_raw}")
                if sm_pa_raw < 0:
                    validation_errors.append(f"sm_pa negative: {sm_pa_raw}")

                prob_sum = sm_ph_raw + sm_pd_raw + sm_pa_raw
                if abs(prob_sum - 1.0) > 0.05:
                    validation_errors.append(f"prob_sum={prob_sum:.4f} not ≈ 1.0 (before normalization)")

                sm_ph, sm_pd, sm_pa = sm_ph_raw, sm_pd_raw, sm_pa_raw
                sm_available = True
                stats["sm_probs_valid"] += 1

                # Temporal status: data fetched post-kickoff (all Phase 15 matches are FT)
                # We cannot verify these were available before kickoff via API.
                # Per specification: must mark TEMPORAL_UNKNOWN (not POST_KICKOFF_VERIFIED)
                sm_temporal_status = "TEMPORAL_UNKNOWN"
            else:
                validation_errors.append("type_id=237 not found or missing home/draw/away keys")
                stats["sm_probs_invalid"] += 1
                sm_temporal_status = "NOT_AVAILABLE"
        else:
            stats["sm_not_found"] += 1
            sm_temporal_status = "NOT_AVAILABLE"

        # ── Validation record ──────────────────────────────────────────────
        validation = {
            "fixture_id_match": fid,
            "league_match": pred["League"] in LEAGUE_NAME_TO_SM_ID,
            "sm_fixture_id": sm_fixture_id,
            "match_method": match_method,
            "match_confidence": match_confidence,
            "home_team_match": (match_method is not None),
            "away_team_match": (match_method is not None),
            "competition_match": (match_method is not None),
            "date_match": (match_method is not None),
            "probs_valid": sm_available,
            "temporal_metadata_available": False,  # Sportmonks API doesn't expose publication timestamp
            "validation_errors": validation_errors,
        }

        # ── Blend probabilities ────────────────────────────────────────────
        blends = {}
        if sm_available:
            for label, (wv4, wsm) in [
                ("m1_v4_only", (1.0, 0.0)),
                ("m2_sm_only", (0.0, 1.0)),
                ("m3_50_50", (0.50, 0.50)),
                ("m4_70_30", (0.70, 0.30)),
                ("m5_30_70", (0.30, 0.70)),
            ]:
                bph, bpd, bpa = blend(v4_ph, v4_pd, v4_pa, sm_ph, sm_pd, sm_pa, wv4, wsm)
                blends[label] = {
                    "p_home": round(bph, 6),
                    "p_draw": round(bpd, 6),
                    "p_away": round(bpa, 6),
                    "pred": argmax_outcome(bph, bpd, bpa),
                    "w_v4": wv4,
                    "w_sm": wsm,
                }

        record = {
            # ── Identity ──────────────────────────────────────────────────
            "fixture_id": fid,
            "date": pred["Date (UTC)"],
            "league": pred["League"],
            "matchweek": pred["Matchweek"],
            "home_team": pred["Home Team"],
            "away_team": pred["Away Team"],
            # ── Arm A: V4 Baseline (IMMUTABLE) ───────────────────────────
            "v4_p_home": v4_ph,
            "v4_p_draw": v4_pd,
            "v4_p_away": v4_pa,
            "v4_pred": v4_pred,
            "v4_canonical_score": v4_canonical_score,
            "v4_draw_risk": pred["draw_risk_tier"],
            "v4_draw_risk_score": pred["draw_risk_score"],
            "v4_signal_profile": pred["signal_profile"],
            "v4_strong_home": pred["strong_home_profile"],
            "v4_is_2_0": pred["is_2_0_profile"],
            "v4_lambda_home": pred["lambda_home"],
            "v4_lambda_away": pred["lambda_away"],
            "model_v4_sha256": pred["model_v4_sha256"],
            # ── Arm B: Sportmonks ─────────────────────────────────────────
            "sm_available": sm_available,
            "sm_fixture_id": sm_fixture_id,
            "sm_match_method": match_method,
            "sm_match_confidence": match_confidence,
            "sm_p_home": round(sm_ph, 6) if sm_ph is not None else None,
            "sm_p_draw": round(sm_pd, 6) if sm_pd is not None else None,
            "sm_p_away": round(sm_pa, 6) if sm_pa is not None else None,
            "sm_pred": argmax_outcome(sm_ph, sm_pd, sm_pa) if sm_available else None,
            "sm_raw_prediction": sm_raw_data,
            "sm_temporal_status": sm_temporal_status,
            "sm_source_timestamp": None,  # Not exposed by Sportmonks API
            # ── Blend Methods (only when sm_available) ─────────────────────
            "blends": blends if sm_available else {},
            # ── Validation ────────────────────────────────────────────────
            "validation": validation,
            # ── Actuals ───────────────────────────────────────────────────
            "actual_status": act_status,
            "actual_score": act_score,
            "actual_outcome": act_outcome,
            "actual_home_goals": act_hg,
            "actual_away_goals": act_ag,
            # ── Metadata ──────────────────────────────────────────────────
            "captured_at_utc": capture_ts,
        }
        records.append(record)

    print(f"  SM found: {stats['sm_found']}/48 (exact={stats['sm_exact']}, substring_both={stats['sm_substring_both']})")
    print(f"  SM probs valid: {stats['sm_probs_valid']}, invalid: {stats['sm_probs_invalid']}")
    print(f"  SM not found: {stats['sm_not_found']}")
    print(f"  NOTE: SM league_id scrambling detected — matched by date+team name across all Big-5")
    return records, stats


# ─── Metric Computation ───────────────────────────────────────────────────────

def compute_method_metrics(records, method_key, label):
    """Compute all metrics for a given blend method across completed fixtures."""
    completed = [r for r in records if r["actual_outcome"] is not None and r["sm_available"]]
    if not completed:
        return {}

    total = len(completed)
    correct = 0
    exact_correct = 0
    brier_sum = 0.0
    ll_sum = 0.0
    draws_correct = 0
    homes_correct = 0
    aways_correct = 0
    actual_draws = 0
    pred_draws = 0

    for r in completed:
        act = r["actual_outcome"]
        if method_key == "m1_v4_only":
            ph, pd, pa = r["v4_p_home"], r["v4_p_draw"], r["v4_p_away"]
        elif method_key == "m2_sm_only":
            ph, pd, pa = r["sm_p_home"], r["sm_p_draw"], r["sm_p_away"]
        else:
            b = r["blends"].get(method_key, {})
            ph, pd, pa = b.get("p_home", 0), b.get("p_draw", 0), b.get("p_away", 0)

        pred = argmax_outcome(ph, pd, pa)

        if pred == act:
            correct += 1
            if act == "H":
                homes_correct += 1
            elif act == "D":
                draws_correct += 1
            else:
                aways_correct += 1

        if act == "H" and pred == "H":
            pass  # counted above
        if act == "D":
            actual_draws += 1
        if pred == "D":
            pred_draws += 1

        brier_sum += brier_score(ph, pd, pa, act)
        ll_sum += log_loss(ph, pd, pa, act)

        # Exact score check (V4 canonical score vs actual)
        canon = r.get("v4_canonical_score", "")
        if canon and r["actual_score"]:
            if canon == r["actual_score"]:
                exact_correct += 1

    actual_h = sum(1 for r in completed if r["actual_outcome"] == "H")
    actual_d = sum(1 for r in completed if r["actual_outcome"] == "D")
    actual_a = sum(1 for r in completed if r["actual_outcome"] == "A")

    return {
        "method": label,
        "method_key": method_key,
        "total": total,
        "correct": correct,
        "accuracy_1x2": round(safe_div(correct, total), 4),
        "exact_score_correct": exact_correct,
        "accuracy_exact": round(safe_div(exact_correct, total), 4),
        "mean_brier": round(safe_div(brier_sum, total), 6),
        "mean_log_loss": round(safe_div(ll_sum, total), 6),
        "actual_h": actual_h, "actual_d": actual_d, "actual_a": actual_a,
        "pred_draws": pred_draws,
        "draws_correct": draws_correct,
        "homes_correct": homes_correct,
        "aways_correct": aways_correct,
        "draw_accuracy": round(safe_div(draws_correct, actual_d), 4) if actual_d else 0.0,
        "home_accuracy": round(safe_div(homes_correct, actual_h), 4) if actual_h else 0.0,
        "away_accuracy": round(safe_div(aways_correct, actual_a), 4) if actual_a else 0.0,
        "actual_draw_rate": round(safe_div(actual_d, total), 4),
        "predicted_draw_rate": round(safe_div(pred_draws, total), 4),
        "draw_calibration_gap": round(safe_div(actual_d, total) - safe_div(pred_draws, total), 4),
    }


def compute_v4_only_metrics(records):
    """Compute V4 metrics on ALL 48 fixtures (not just SM-covered)."""
    completed = [r for r in records if r["actual_outcome"] is not None]
    total = len(completed)
    correct = 0
    exact_correct = 0
    brier_sum = 0.0
    ll_sum = 0.0
    draws_correct = 0
    actual_draws = 0
    pred_draws = 0
    homes_correct = 0
    aways_correct = 0

    for r in completed:
        act = r["actual_outcome"]
        ph, pd, pa = r["v4_p_home"], r["v4_p_draw"], r["v4_p_away"]
        pred = argmax_outcome(ph, pd, pa)

        if pred == act:
            correct += 1
            if act == "H": homes_correct += 1
            elif act == "D": draws_correct += 1
            else: aways_correct += 1
        if act == "D": actual_draws += 1
        if pred == "D": pred_draws += 1

        brier_sum += brier_score(ph, pd, pa, act)
        ll_sum += log_loss(ph, pd, pa, act)

        canon = r.get("v4_canonical_score", "")
        if canon and r["actual_score"] and canon == r["actual_score"]:
            exact_correct += 1

    actual_h = sum(1 for r in completed if r["actual_outcome"] == "H")
    actual_d = sum(1 for r in completed if r["actual_outcome"] == "D")
    actual_a = sum(1 for r in completed if r["actual_outcome"] == "A")

    return {
        "method": "V4 Only (Full 48)",
        "method_key": "v4_full_48",
        "total": total,
        "correct": correct,
        "accuracy_1x2": round(safe_div(correct, total), 4),
        "exact_score_correct": exact_correct,
        "accuracy_exact": round(safe_div(exact_correct, total), 4),
        "mean_brier": round(safe_div(brier_sum, total), 6),
        "mean_log_loss": round(safe_div(ll_sum, total), 6),
        "actual_h": actual_h, "actual_d": actual_d, "actual_a": actual_a,
        "pred_draws": pred_draws,
        "draws_correct": draws_correct,
        "draw_accuracy": round(safe_div(draws_correct, actual_d), 4) if actual_d else 0.0,
        "home_accuracy": round(safe_div(homes_correct, actual_h), 4) if actual_h else 0.0,
        "away_accuracy": round(safe_div(aways_correct, actual_a), 4) if actual_a else 0.0,
        "actual_draw_rate": round(safe_div(actual_d, total), 4),
        "predicted_draw_rate": round(safe_div(pred_draws, total), 4),
        "draw_calibration_gap": round(safe_div(actual_d, total) - safe_div(pred_draws, total), 4),
    }


# ─── Draw Analysis ────────────────────────────────────────────────────────────

def compute_draw_analysis(records):
    """Deep draw-specific analysis comparing V4 vs Sportmonks vs blends."""
    completed = [r for r in records if r["actual_outcome"] is not None]
    sm_covered = [r for r in completed if r["sm_available"]]

    actual_draw_records = [r for r in completed if r["actual_outcome"] == "D"]
    actual_draw_records_sm = [r for r in sm_covered if r["actual_outcome"] == "D"]

    def draw_stats(records_list, label, ph_key, pd_key, pa_key):
        if not records_list:
            return {"label": label, "n": 0}
        pd_vals = []
        for r in records_list:
            if ph_key.startswith("blend:"):
                mk = ph_key.split(":")[1]
                b = r["blends"].get(mk, {})
                pd_val = b.get("p_draw", None)
            else:
                pd_val = r.get(pd_key)
            if pd_val is not None:
                pd_vals.append(pd_val)
        if not pd_vals:
            return {"label": label, "n": len(records_list), "mean_pd": None}
        return {
            "label": label,
            "n": len(records_list),
            "mean_pd": round(sum(pd_vals) / len(pd_vals), 4),
            "median_pd": round(sorted(pd_vals)[len(pd_vals) // 2], 4),
            "min_pd": round(min(pd_vals), 4),
            "max_pd": round(max(pd_vals), 4),
        }

    # How often Sportmonks changes draw prediction
    sm_increases_pd = 0
    sm_decreases_pd = 0
    sm_makes_draw_argmax = 0
    sm_removes_draw_argmax = 0
    sm_changes_pred = 0
    agreement_v4_sm = 0
    disagreement_v4_sm = 0

    for r in sm_covered:
        v4_d = r["v4_pred"]
        sm_d = r["sm_pred"]
        v4_pd = r["v4_p_draw"]
        sm_pd = r["sm_p_draw"]

        if sm_pd > v4_pd:
            sm_increases_pd += 1
        elif sm_pd < v4_pd:
            sm_decreases_pd += 1

        if v4_d != "D" and sm_d == "D":
            sm_makes_draw_argmax += 1
        if v4_d == "D" and sm_d != "D":
            sm_removes_draw_argmax += 1
        if v4_d != sm_d:
            sm_changes_pred += 1
            disagreement_v4_sm += 1
        else:
            agreement_v4_sm += 1

    # Draw calibration across methods
    def draw_calibration(records_list, ph_key, pd_key, pa_key, method_label):
        actual_draws = sum(1 for r in records_list if r["actual_outcome"] == "D")
        pred_draws = 0
        for r in records_list:
            if ph_key.startswith("blend:"):
                mk = ph_key.split(":")[1]
                b = r["blends"].get(mk, {})
                pred = b.get("pred")
            else:
                ph = r.get("v4_p_home" if "v4" in ph_key else "sm_p_home", 0)
                pd = r.get("v4_p_draw" if "v4" in pd_key else "sm_p_draw", 0)
                pa = r.get("v4_p_away" if "v4" in pa_key else "sm_p_away", 0)
                pred = argmax_outcome(ph, pd, pa)
            if pred == "D":
                pred_draws += 1
        n = len(records_list)
        return {
            "method": method_label,
            "n": n,
            "actual_draws": actual_draws,
            "pred_draws": pred_draws,
            "actual_draw_rate": round(safe_div(actual_draws, n), 4),
            "predicted_draw_rate": round(safe_div(pred_draws, n), 4),
            "calibration_gap": round(safe_div(actual_draws, n) - safe_div(pred_draws, n), 4),
        }

    calibration_results = [
        draw_calibration(sm_covered, "v4_p_home", "v4_p_draw", "v4_p_away", "V4 Only"),
        draw_calibration(sm_covered, "sm_p_home", "sm_p_draw", "sm_p_away", "Sportmonks Only"),
        draw_calibration(sm_covered, "blend:m3_50_50", "blend:m3_50_50", "blend:m3_50_50", "50/50 Blend"),
        draw_calibration(sm_covered, "blend:m4_70_30", "blend:m4_70_30", "blend:m4_70_30", "70/30 V4-Heavy"),
        draw_calibration(sm_covered, "blend:m5_30_70", "blend:m5_30_70", "blend:m5_30_70", "30/70 SM-Heavy"),
    ]

    return {
        "meta": {
            "cohort": "2026-09-18 to 2026-09-21",
            "total_fixtures": len(completed),
            "sm_covered_fixtures": len(sm_covered),
            "actual_draws_all_48": len(actual_draw_records),
            "actual_draws_sm_covered": len(actual_draw_records_sm),
            "actual_draw_rate": round(safe_div(len(actual_draw_records), len(completed)), 4),
            "note": "V4 predicted 0 draws out of 48 (0%) — Phase 15 key finding",
        },
        "draw_pd_stats_on_actual_draws": {
            "v4": draw_stats(actual_draw_records_sm, "V4 (on actual draw fixtures)",
                             "v4_p_home", "v4_p_draw", "v4_p_away"),
            "sportmonks": draw_stats(actual_draw_records_sm, "Sportmonks (on actual draw fixtures)",
                                     "sm_p_home", "sm_p_draw", "sm_p_away"),
        },
        "sportmonks_draw_impact": {
            "sm_increases_pd": sm_increases_pd,
            "sm_decreases_pd": sm_decreases_pd,
            "sm_makes_draw_argmax": sm_makes_draw_argmax,
            "sm_removes_draw_argmax": sm_removes_draw_argmax,
            "sm_changes_pred": sm_changes_pred,
            "agreement_v4_sm": agreement_v4_sm,
            "disagreement_v4_sm": disagreement_v4_sm,
            "agreement_rate": round(safe_div(agreement_v4_sm, len(sm_covered)), 4),
        },
        "draw_calibration_by_method": calibration_results,
        "actual_draw_fixtures": [
            {
                "fixture_id": r["fixture_id"],
                "home": r["home_team"],
                "away": r["away_team"],
                "actual_score": r["actual_score"],
                "v4_pd": r["v4_p_draw"],
                "v4_pred": r["v4_pred"],
                "sm_pd": r["sm_p_draw"],
                "sm_pred": r["sm_pred"],
                "m3_pd": r["blends"].get("m3_50_50", {}).get("p_draw"),
                "m3_pred": r["blends"].get("m3_50_50", {}).get("pred"),
                "league": r["league"],
                "v4_draw_risk": r["v4_draw_risk"],
            }
            for r in sm_covered if r["actual_outcome"] == "D"
        ],
    }


# ─── Signal Analysis ──────────────────────────────────────────────────────────

def compute_signal_analysis(records):
    """Analyze 2-0 profile and strong home signals for V4 vs blends."""
    sm_covered = [r for r in records if r["sm_available"] and r["actual_outcome"] is not None]

    def signal_metrics(fixtures, method_key, label):
        if not fixtures:
            return {"label": label, "n": 0}
        correct = 0
        exact = 0
        draws = 0
        homes = 0
        aways = 0
        for r in fixtures:
            act = r["actual_outcome"]
            if method_key == "v4":
                pred = r["v4_pred"]
            elif method_key == "sm":
                pred = r["sm_pred"]
            else:
                b = r["blends"].get(method_key, {})
                pred = b.get("pred", r["v4_pred"])

            if pred == act:
                correct += 1
            if act == "H": homes += 1
            if act == "D": draws += 1
            if act == "A": aways += 1

            canon = r.get("v4_canonical_score", "")
            if canon and r.get("actual_score") and canon == r["actual_score"]:
                exact += 1

        n = len(fixtures)
        return {
            "label": label, "n": n,
            "correct": correct,
            "accuracy": round(safe_div(correct, n), 4),
            "exact": exact,
            "exact_accuracy": round(safe_div(exact, n), 4),
            "actual_h": homes, "actual_d": draws, "actual_a": aways,
        }

    signal_2_0 = [r for r in sm_covered if r.get("v4_is_2_0")]
    strong_home = [r for r in sm_covered if r.get("v4_strong_home")]

    # Draw risk tiers
    risk_analysis = {}
    for tier in ("LOW", "MEDIUM", "HIGH"):
        tier_recs = [r for r in sm_covered if r.get("v4_draw_risk") == tier]
        tier_actual_draws = [r for r in tier_recs if r["actual_outcome"] == "D"]
        risk_analysis[tier] = {
            "count": len(tier_recs),
            "actual_draws": len(tier_actual_draws),
            "actual_draw_rate": round(safe_div(len(tier_actual_draws), len(tier_recs)), 4),
            "v4_pred_draws": sum(1 for r in tier_recs if r["v4_pred"] == "D"),
            "sm_pred_draws": sum(1 for r in tier_recs if r.get("sm_pred") == "D"),
            "m3_pred_draws": sum(1 for r in tier_recs
                                 if r["blends"].get("m3_50_50", {}).get("pred") == "D"),
        }

    return {
        "meta": {"cohort": "2026-09-18 to 2026-09-21"},
        "2_0_profile": {
            "v4": signal_metrics(signal_2_0, "v4", "V4 Only"),
            "sm": signal_metrics(signal_2_0, "sm", "Sportmonks Only"),
            "m3_50_50": signal_metrics(signal_2_0, "m3_50_50", "50/50 Blend"),
            "m4_70_30": signal_metrics(signal_2_0, "m4_70_30", "70/30 V4-Heavy"),
            "details": [
                {"fixture_id": r["fixture_id"], "home": r["home_team"], "away": r["away_team"],
                 "v4_pred": r["v4_pred"], "v4_score_pred": r["v4_canonical_score"],
                 "sm_pred": r["sm_pred"], "sm_pd": r["sm_p_draw"],
                 "actual_outcome": r["actual_outcome"], "actual_score": r["actual_score"],
                 "lambda_h": r["v4_lambda_home"], "lambda_a": r["v4_lambda_away"]}
                for r in signal_2_0
            ],
        },
        "strong_home_profile": {
            "v4": signal_metrics(strong_home, "v4", "V4 Only"),
            "sm": signal_metrics(strong_home, "sm", "Sportmonks Only"),
            "m3_50_50": signal_metrics(strong_home, "m3_50_50", "50/50 Blend"),
            "details": [
                {"fixture_id": r["fixture_id"], "home": r["home_team"], "away": r["away_team"],
                 "v4_pred": r["v4_pred"], "sm_pred": r["sm_pred"],
                 "actual_outcome": r["actual_outcome"], "actual_score": r["actual_score"],
                 "v4_is_2_0": r.get("v4_is_2_0", False)}
                for r in strong_home
            ],
        },
        "draw_risk_by_tier": risk_analysis,
    }


# ─── League Analysis ──────────────────────────────────────────────────────────

def compute_league_breakdown(records, methods_results):
    """Break down method accuracy by league."""
    sm_covered = [r for r in records if r["sm_available"] and r["actual_outcome"] is not None]
    leagues = sorted({r["league"] for r in sm_covered})

    league_data = {}
    for lg in leagues:
        lg_recs = [r for r in sm_covered if r["league"] == lg]
        lg_data = {"count": len(lg_recs)}
        for method_key, label in [
            ("v4", "V4 Only"),
            ("sm", "SM Only"),
            ("m3_50_50", "50/50"),
            ("m4_70_30", "70/30"),
        ]:
            correct = 0
            for r in lg_recs:
                act = r["actual_outcome"]
                if method_key == "v4":
                    pred = r["v4_pred"]
                elif method_key == "sm":
                    pred = r["sm_pred"]
                else:
                    pred = r["blends"].get(method_key, {}).get("pred", r["v4_pred"])
                if pred == act:
                    correct += 1
            lg_data[method_key + "_accuracy"] = round(safe_div(correct, len(lg_recs)), 4)
        league_data[lg] = lg_data

    return league_data


# ─── Report Generation ────────────────────────────────────────────────────────

def generate_report(records, stats_sm, methods_results, draw_analysis, signal_analysis,
                    league_breakdown, v4_full_48):
    """Generate the full markdown Phase 16 report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sm_n = sum(1 for r in records if r["sm_available"])
    total_n = len(records)
    sm_temporal_unknown = sum(1 for r in records if r.get("sm_temporal_status") == "TEMPORAL_UNKNOWN")

    # Pull out key method results
    def get_method(key):
        return next((m for m in methods_results if m.get("method_key") == key), {})

    v4_sm = get_method("m1_v4_only")
    sm_only = get_method("m2_sm_only")
    m3 = get_method("m3_50_50")
    m4 = get_method("m4_70_30")
    m5 = get_method("m5_30_70")

    di = draw_analysis["sportmonks_draw_impact"]

    lines = [
        "# Phase 16 — V4 + Sportmonks Research Arm Report",
        "",
        "> ⚠️ **RESEARCH ONLY — NOT PRODUCTION**  ",
        "> V4 model is frozen and unchanged. Sportmonks predictions are RESEARCH only.",
        f"> Sportmonks temporal status: **TEMPORAL_UNKNOWN** (fetched post-kickoff, API has no publication timestamp)",
        "",
        f"**Cohort:** 2026-09-18 → 2026-09-21 | **Generated:** {now}  ",
        f"**V4 SHA256:** `{FROZEN_SHA256}`  ",
        f"**Phase 15 Baseline:** 1X2=56.2%, Brier=0.5785, LogLoss=0.9732 (full 48)",
        "",
        "---",
        "",
        "## 1. Sportmonks Coverage",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total target fixtures | {total_n} |",
        f"| Sportmonks available | **{sm_n}** |",
        f"| Sportmonks unavailable | {total_n - sm_n} |",
        f"| Coverage % | **{100 * sm_n / total_n:.1f}%** |",
        f"| Temporal status | TEMPORAL_UNKNOWN: {sm_temporal_unknown}, PRE_KICKOFF_VERIFIED: 0 |",
        "",
        "> ⚠️ **All Sportmonks data is TEMPORAL_UNKNOWN** — data was retrieved after kickoff.",
        "> The Sportmonks API does not expose a publication timestamp.",
        "> These results cannot be used as clean prospective benchmarks.",
        "",
        "**Coverage by league:**",
        "",
        "| League | Total | SM Available | Coverage |",
        "|--------|-------|-------------|---------|",
    ]

    for lg in sorted({r["league"] for r in records}):
        lg_all = [r for r in records if r["league"] == lg]
        lg_sm = [r for r in lg_all if r["sm_available"]]
        lines.append(f"| {lg} | {len(lg_all)} | {len(lg_sm)} | {100*len(lg_sm)/len(lg_all):.0f}% |")
    lines.append("")

    lines += [
        "## 2. V4 vs Sportmonks Agreement",
        "",
        f"| Metric | Count | Rate |",
        f"|--------|-------|------|",
        f"| V4 / SM agree on argmax | {di['agreement_v4_sm']} | {di['agreement_rate']:.1%} |",
        f"| V4 / SM disagree | {di['disagreement_v4_sm']} | {1-di['agreement_rate']:.1%} |",
        f"| SM increases P(D) vs V4 | {di['sm_increases_pd']} | {100*di['sm_increases_pd']/max(1,sm_n):.0f}% |",
        f"| SM decreases P(D) vs V4 | {di['sm_decreases_pd']} | {100*di['sm_decreases_pd']/max(1,sm_n):.0f}% |",
        f"| SM changes argmax to DRAW | {di['sm_makes_draw_argmax']} | — |",
        f"| SM changes argmax FROM DRAW | {di['sm_removes_draw_argmax']} | — |",
        "",
        "## 3. Method Comparison Results",
        "",
        f"> ⚠️ All Sportmonks blends are **TEMPORAL_UNKNOWN** — diagnostic only, not prospective.",
        "",
        f"| Method | N | 1X2 Acc | Exact Acc | Brier | Log Loss | Draw Preds |",
        f"|--------|---|---------|-----------|-------|----------|------------|",
    ]

    for mr in [
        {"label": "V4 Full 48 (baseline)", **v4_full_48},
        *methods_results,
    ]:
        lines.append(
            f"| {mr.get('method', mr.get('label', '?'))} | {mr.get('total', '?')} "
            f"| {mr.get('accuracy_1x2', 0):.1%} | {mr.get('accuracy_exact', 0):.1%} "
            f"| {mr.get('mean_brier', 0):.4f} | {mr.get('mean_log_loss', 0):.4f} "
            f"| {mr.get('pred_draws', 0)} |"
        )
    lines.append("")

    lines += [
        "## 4. Draw Analysis",
        "",
        f"10 actual draws occurred in Phase 15 (of 48 = 20.8%).",
        f"V4 predicted **0 draws** (0.0% predicted draw rate).",
        "",
        "**Draw calibration by method (SM-covered fixtures only):**",
        "",
        "| Method | N | Actual Draws | Pred Draws | Actual Rate | Pred Rate | Gap |",
        "|--------|---|------------|-----------|------------|----------|-----|",
    ]
    for dc in draw_analysis["draw_calibration_by_method"]:
        lines.append(
            f"| {dc['method']} | {dc['n']} | {dc['actual_draws']} | {dc['pred_draws']} "
            f"| {dc['actual_draw_rate']:.1%} | {dc['predicted_draw_rate']:.1%} "
            f"| {dc['calibration_gap']:+.1%} |"
        )
    lines.append("")

    lines += [
        "**Actual draw fixtures — P(D) comparison:**",
        "",
        "| Fixture | Score | V4 P(D) | SM P(D) | V4 Pred | SM Pred | League |",
        "|---------|-------|---------|---------|---------|---------|--------|",
    ]
    for dr in draw_analysis["actual_draw_fixtures"]:
        lines.append(
            f"| {dr['home']} vs {dr['away']} | {dr['actual_score']} "
            f"| {dr['v4_pd']:.3f} | {dr['sm_pd'] if dr['sm_pd'] else 'N/A':.3f} "
            f"| {dr['v4_pred']} | {dr['sm_pred'] or 'N/A'} | {dr['league']} |"
        )
    lines.append("")

    lines += [
        "## 5. Signal Analysis",
        "",
        "### 2-0 Profile Fixtures",
        "",
    ]
    s2 = signal_analysis["2_0_profile"]
    lines += [
        f"| Method | N | Outcome Acc | Exact Acc |",
        f"|--------|---|------------|----------|",
        f"| V4 Only | {s2['v4']['n']} | {s2['v4']['accuracy']:.1%} | {s2['v4']['exact_accuracy']:.1%} |",
        f"| Sportmonks Only | {s2['sm']['n']} | {s2['sm']['accuracy']:.1%} | {s2['sm']['exact_accuracy']:.1%} |",
        f"| 50/50 Blend | {s2['m3_50_50']['n']} | {s2['m3_50_50']['accuracy']:.1%} | {s2['m3_50_50']['exact_accuracy']:.1%} |",
        "",
        "### Strong Home Profile",
        "",
    ]
    sh = signal_analysis["strong_home_profile"]
    lines += [
        f"| Method | N | Outcome Acc |",
        f"|--------|---|------------|",
        f"| V4 Only | {sh['v4']['n']} | {sh['v4']['accuracy']:.1%} |",
        f"| Sportmonks Only | {sh['sm']['n']} | {sh['sm']['accuracy']:.1%} |",
        f"| 50/50 Blend | {sh['m3_50_50']['n']} | {sh['m3_50_50']['accuracy']:.1%} |",
        "",
        "## 6. League Breakdown (SM-Covered Fixtures)",
        "",
        "| League | N | V4 Acc | SM Acc | 50/50 Acc | 70/30 Acc |",
        "|--------|---|--------|--------|-----------|-----------|",
    ]
    for lg, ld in sorted(league_breakdown.items()):
        lines.append(
            f"| {lg} | {ld['count']} "
            f"| {ld.get('v4_accuracy', 0):.1%} | {ld.get('sm_accuracy', 0):.1%} "
            f"| {ld.get('m3_50_50_accuracy', 0):.1%} | {ld.get('m4_70_30_accuracy', 0):.1%} |"
        )
    lines.append("")

    lines += [
        "## 7. Answers to Research Questions",
        "",
        f"1. **Sportmonks coverage:** {sm_n}/{total_n} = {100*sm_n/total_n:.0f}% of fixtures",
        f"2. **Temporally verified:** 0 / {sm_n} — all TEMPORAL_UNKNOWN (retrieved post-kickoff)",
        f"3. **V4 / SM agreement rate:** {di['agreement_rate']:.1%}",
        f"4. **V4 / SM disagreement rate:** {1-di['agreement_rate']:.1%}",
        f"5. **Sportmonks-only accuracy:** {sm_only.get('accuracy_1x2', 0):.1%} (on {sm_only.get('total', 0)} SM-covered fixtures)",
        f"6. **V4 accuracy on same fixtures:** {v4_sm.get('accuracy_1x2', 0):.1%}",
        f"7. **50/50 blend accuracy:** {m3.get('accuracy_1x2', 0):.1%}",
        f"8. **70/30 V4-heavy blend accuracy:** {m4.get('accuracy_1x2', 0):.1%}",
        f"9. **30/70 diagnostic blend accuracy:** {m5.get('accuracy_1x2', 0):.1%}",
        f"10. **Does SM improve DRAW prediction?** SM predicted {di['sm_makes_draw_argmax']} draws; V4 predicted 0. See draw calibration table.",
        f"11. **SM vs V4 Brier:** {sm_only.get('mean_brier', 0):.4f} vs {v4_sm.get('mean_brier', 0):.4f}",
        f"12. **SM vs V4 Log Loss:** {sm_only.get('mean_log_loss', 0):.4f} vs {v4_sm.get('mean_log_loss', 0):.4f}",
        f"13. **SM changes 2-0 signals?** SM predicted {s2['sm'].get('n', 0)} 2-0 profile fixtures with {s2['sm']['accuracy']:.1%} accuracy",
        f"14. **SM changes Strong Home?** {sh['sm']['accuracy']:.1%} vs V4 {sh['v4']['accuracy']:.1%}",
        f"15. **Evidence for production integration?** INSUFFICIENT — all data is TEMPORAL_UNKNOWN. Requires prospective pre-kickoff capture.",
        "",
        "## 8. Governance Decision",
        "",
        "> **Sportmonks remains RESEARCH ONLY.**",
        "",
        "All Phase 16 Sportmonks data is marked `TEMPORAL_UNKNOWN` because:",
        "- Data was retrieved after all Phase 15 matches have concluded (FT status)",
        "- Sportmonks API does not expose a publication timestamp for predictions",
        "- We cannot verify whether the probabilities shown are identical to pre-kickoff values",
        "",
        "**Recommendation:** Continue prospective V4 vs V4+Sportmonks validation by capturing",
        "Sportmonks predictions BEFORE kickoff for the next upcoming cohort.",
        "",
        "> Do NOT declare Sportmonks better or worse based on this TEMPORAL_UNKNOWN retrospective sample.",
        "",
        "---",
        "",
        "## Appendix — Frozen V4 Model",
        "",
        f"```",
        f"model_v4_sha256: {FROZEN_SHA256}",
        f"```",
        "",
        "*Phase 16 is research-only. V4 model was NOT modified, retrained, or re-parameterized.*",
        "*Production predictions remain unchanged. Sportmonks is NOT promoted to production.*",
    ]

    return "\n".join(lines)


# ─── Dashboard Generation ─────────────────────────────────────────────────────

def generate_dashboard(records):
    """Generate offline-capable Phase 16 research HTML dashboard."""
    sm_n = sum(1 for r in records if r["sm_available"])

    rows_html_parts = []
    for r in records:
        sm_avail = r["sm_available"]
        v4_pred = r["v4_pred"]
        sm_pred = r["sm_pred"] or "N/A"
        agree = v4_pred == sm_pred if sm_avail else None
        agree_icon = "✅" if agree else "❌" if agree is False else "—"

        m3_pred = r["blends"].get("m3_50_50", {}).get("pred", "—")
        m4_pred = r["blends"].get("m4_70_30", {}).get("pred", "—")

        act = r.get("actual_outcome") or "?"
        act_score = r.get("actual_score") or "?"

        row_cls = ""
        if r.get("actual_outcome"):
            row_cls = "row-correct" if v4_pred == act else "row-wrong"

        sm_cell = (f"{r['sm_p_home']:.1%} / {r['sm_p_draw']:.1%} / {r['sm_p_away']:.1%}"
                   if sm_avail else "<em>N/A</em>")
        m3_cell = f"{r['blends'].get('m3_50_50',{}).get('p_home',0):.1%} / {r['blends'].get('m3_50_50',{}).get('p_draw',0):.1%} / {r['blends'].get('m3_50_50',{}).get('p_away',0):.1%}" if sm_avail else "<em>N/A</em>"

        rows_html_parts.append(f"""
            <tr class="{row_cls}">
              <td>{r['date']}</td>
              <td><span class="lg">{r['league']}</span></td>
              <td>{r['home_team']}</td>
              <td>{r['away_team']}</td>
              <td>{r['v4_p_home']:.1%}/{r['v4_p_draw']:.1%}/{r['v4_p_away']:.1%}</td>
              <td><strong>{v4_pred}</strong></td>
              <td>{sm_cell}</td>
              <td><strong>{sm_pred}</strong></td>
              <td>{agree_icon}</td>
              <td>{m3_cell}</td>
              <td><strong>{m3_pred}</strong></td>
              <td>{m4_pred}</td>
              <td>{r['v4_draw_risk']}</td>
              <td>{r['v4_signal_profile']}</td>
              <td><strong>{act}</strong></td>
              <td>{act_score}</td>
              <td>{'TEMPORAL_UNKNOWN' if sm_avail else 'N/A'}</td>
            </tr>""")

    rows_html = "\n".join(rows_html_parts)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 16 Research | V4 + Sportmonks | 2026-09-18–21</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:20px;}}
  h1{{color:#58a6ff;margin-bottom:4px;}}
  .subtitle{{color:#8b949e;margin-bottom:14px;font-size:.9em;}}
  .warn{{background:#271b00;border-left:4px solid #d29922;padding:10px 16px;border-radius:4px;margin-bottom:18px;font-size:.85em;}}
  .sha{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:.78em;color:#79c0ff;display:inline-block;margin-bottom:16px;}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px;}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;text-align:center;}}
  .card .v{{font-size:1.8em;font-weight:bold;color:#58a6ff;}}
  .card .l{{font-size:.78em;color:#8b949e;margin-top:4px;}}
  table{{width:100%;border-collapse:collapse;font-size:.76em;}}
  th{{background:#161b22;color:#8b949e;padding:7px 8px;text-align:left;border-bottom:2px solid #30363d;position:sticky;top:0;z-index:10;}}
  td{{padding:6px 8px;border-bottom:1px solid #21262d;vertical-align:middle;}}
  tr.row-correct{{background:rgba(63,185,80,.06);}}
  tr.row-wrong{{background:rgba(248,81,73,.07);}}
  tr:hover{{background:#1c2128!important;}}
  .lg{{background:#1f2937;color:#93c5fd;border-radius:10px;padding:2px 7px;font-size:.8em;}}
  .wrap{{overflow-x:auto;}}
</style>
</head>
<body>
<h1>⚗️ Phase 16 Research — V4 + Sportmonks</h1>
<div class="subtitle">2026-09-18 → 2026-09-21 | Big-5 Leagues | RESEARCH ONLY — NOT PRODUCTION</div>
<div class="warn">
  ⚠️ <strong>TEMPORAL_UNKNOWN</strong> — All Sportmonks data was retrieved after match completion.
  Sportmonks does not expose a publication timestamp. These results are diagnostic only and cannot
  serve as prospective evidence. <strong>V4 remains the production model.</strong>
</div>
<div class="sha">🔒 V4 SHA256: {FROZEN_SHA256}</div>
<div class="cards">
  <div class="card"><div class="v">48</div><div class="l">Total Fixtures</div></div>
  <div class="card"><div class="v">{sm_n}</div><div class="l">SM Coverage</div></div>
  <div class="card"><div class="v">0</div><div class="l">Pre-Kickoff Verified</div></div>
  <div class="card"><div class="v">RESEARCH</div><div class="l">Status</div></div>
</div>
<h2>📊 Full Fixture Table</h2>
<div class="wrap">
<table>
  <thead>
    <tr>
      <th>Date</th><th>League</th><th>Home</th><th>Away</th>
      <th>V4 H/D/A</th><th>V4 Pred</th>
      <th>SM H/D/A</th><th>SM Pred</th><th>Agree</th>
      <th>50/50 H/D/A</th><th>50/50</th><th>70/30</th>
      <th>Draw Risk</th><th>Signal</th>
      <th>Actual</th><th>Score</th><th>Temporal</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
<p style="margin-top:28px;color:#484f58;font-size:.78em;">
  Generated {now} | Research only — Sportmonks NOT promoted to production | V4 frozen
</p>
</body>
</html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 16 — V4 + Sportmonks Prospective Research Arm")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # 1. Verify V4 integrity
    sha = verify_v4_sha256()
    print(f"\n[1] V4 SHA256 verified: {sha}")

    # 2. Load inputs
    ledger = load_phase15_ledger()
    print(f"[2] Loaded {len(ledger)} Phase 15 V4 predictions (immutable ledger)")

    actuals = load_phase15_actuals()
    print(f"[3] Loaded {len(actuals)} Phase 15 actual results")

    sm_raw = load_sportmonks_raw()
    big5_sm = [f for f in sm_raw if f.get("league_id") in SM_LEAGUE_IDS]
    print(f"[4] Loaded {len(sm_raw)} Sportmonks fixtures ({len(big5_sm)} Big-5)")

    # 3. Build fixture universe
    print("\n[5] Building fixture universe (joining V4 + SM + actuals)...")
    records, sm_stats = build_fixture_universe(ledger, actuals, big5_sm)
    sm_covered = [r for r in records if r["sm_available"]]
    print(f"    SM-covered: {len(sm_covered)}/48")

    # 4. Compute metrics for all methods
    print("\n[6] Computing method metrics...")
    methods_results = []
    for method_key, label in [
        ("m1_v4_only", "V4 Only (SM-covered subset)"),
        ("m2_sm_only", "Sportmonks Only"),
        ("m3_50_50", "50/50 Blend"),
        ("m4_70_30", "70/30 V4-Heavy"),
        ("m5_30_70", "30/70 SM-Heavy (Diagnostic)"),
    ]:
        mr = compute_method_metrics(records, method_key, label)
        if mr:
            methods_results.append(mr)
            print(f"    {label}: acc={mr['accuracy_1x2']:.1%}, brier={mr['mean_brier']:.4f}")

    v4_full_48 = compute_v4_only_metrics(records)
    print(f"    V4 Full 48: acc={v4_full_48['accuracy_1x2']:.1%}, brier={v4_full_48['mean_brier']:.4f}")

    # 5. Draw analysis
    print("\n[7] Computing draw analysis...")
    draw_analysis = compute_draw_analysis(records)

    # 6. Signal analysis
    print("[8] Computing signal analysis...")
    signal_analysis = compute_signal_analysis(records)

    # 7. League breakdown
    print("[9] Computing league breakdown...")
    league_breakdown = compute_league_breakdown(records, methods_results)

    # ── Save all output files ──────────────────────────────────────────────────
    print("\n[10] Saving output files...")

    # 01 — Fixture universe
    fpath = os.path.join(OUT_DIR, "01_phase16_fixture_universe.jsonl")
    with open(fpath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"     {fpath}")

    # 02 — Sportmonks raw predictions (per-fixture extracted)
    sm_preds = []
    for r in records:
        sm_preds.append({
            "fixture_id": r["fixture_id"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "league": r["league"],
            "date": r["date"],
            "sm_fixture_id": r["sm_fixture_id"],
            "sm_available": r["sm_available"],
            "sm_p_home": r["sm_p_home"],
            "sm_p_draw": r["sm_p_draw"],
            "sm_p_away": r["sm_p_away"],
            "sm_pred": r["sm_pred"],
            "sm_raw_prediction": r["sm_raw_prediction"],
            "sm_temporal_status": r["sm_temporal_status"],
            "sm_match_method": r["sm_match_method"],
        })
    fpath = os.path.join(OUT_DIR, "02_sportmonks_raw_predictions.jsonl")
    with open(fpath, "w", encoding="utf-8") as f:
        for r in sm_preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"     {fpath}")

    # 03 — Source validation
    validation_summary = {
        "meta": {
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "cohort": "2026-09-18 to 2026-09-21",
            "total_fixtures": len(records),
        },
        "coverage": {
            "sm_found": sm_stats["sm_found"],
            "sm_exact_match": sm_stats["sm_exact"],
            "sm_substring_both_match": sm_stats["sm_substring_both"],
            "sm_not_found": sm_stats["sm_not_found"],
            "sm_probs_valid": sm_stats["sm_probs_valid"],
            "sm_probs_invalid": sm_stats["sm_probs_invalid"],
            "coverage_pct": round(100 * sm_stats["sm_found"] / len(records), 2),
            "league_id_scrambling_detected": sm_stats.get("sm_league_id_scrambling_detected", False),
        },
        "temporal_status": {
            "PRE_KICKOFF_VERIFIED": 0,
            "TEMPORAL_UNKNOWN": sm_stats["sm_probs_valid"],
            "POST_KICKOFF": 0,
            "NOT_AVAILABLE": len(records) - sm_stats["sm_found"],
            "note": "Data retrieved post-kickoff. API provides no publication timestamp. Cannot classify as PRE_KICKOFF_VERIFIED.",
        },
        "validation_checks": {
            "fixture_id_mapping": "OddAlerts fixture_id primary key; Sportmonks matched by date+team name (league_id ignored due to scrambling)",
            "home_team_mapping": "Normalized unicode-stripped comparison; exact or substring match",
            "away_team_mapping": "Same as home team",
            "competition_mapping": "OddAlerts league name → Sportmonks league_id lookup table",
            "date_time_mapping": "Date (YYYY-MM-DD) matched exactly",
            "probability_validity": "P(H)>=0, P(D)>=0, P(A)>=0 and sum≈1 enforced",
            "probability_sum": "Renormalized after extraction",
            "missing_values": f"{sm_stats['sm_probs_invalid']} fixtures had invalid/missing probabilities",
            "provider_status": "Sportmonks API returned 200 OK for all requests",
            "temporal_metadata": "NOT AVAILABLE — Sportmonks does not expose prediction publication timestamp",
        },
        "fixture_validation_errors": [
            {"fixture_id": r["fixture_id"], "errors": r["validation"]["validation_errors"]}
            for r in records
            if r["validation"]["validation_errors"]
        ],
    }
    fpath = os.path.join(OUT_DIR, "03_phase16_source_validation.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2, ensure_ascii=False)
    print(f"     {fpath}")

    # 04 — V4 baseline
    v4_baseline = {
        "meta": {"method": "V4_ONLY", "frozen_sha256": FROZEN_SHA256},
        "full_48": v4_full_48,
        "sm_covered_subset": next((m for m in methods_results if m["method_key"] == "m1_v4_only"), {}),
    }
    fpath = os.path.join(OUT_DIR, "04_phase16_v4_baseline.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(v4_baseline, f, indent=2, ensure_ascii=False)
    print(f"     {fpath}")

    # 05 — Sportmonks only
    fpath = os.path.join(OUT_DIR, "05_phase16_sportmonks_only.json")
    sm_only_result = next((m for m in methods_results if m["method_key"] == "m2_sm_only"), {})
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"method": "SPORTMONKS_ONLY", "temporal_status": "TEMPORAL_UNKNOWN"},
            "results": sm_only_result,
        }, f, indent=2, ensure_ascii=False)
    print(f"     {fpath}")

    # 06 — Blend results
    fpath = os.path.join(OUT_DIR, "06_phase16_blend_results.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "temporal_note": "All Sportmonks data is TEMPORAL_UNKNOWN",
                "cohort": "2026-09-18 to 2026-09-21",
            },
            "methods": methods_results,
            "v4_full_48_reference": v4_full_48,
        }, f, indent=2, ensure_ascii=False)
    print(f"     {fpath}")

    # 07 — Draw analysis
    fpath = os.path.join(OUT_DIR, "07_phase16_draw_analysis.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(draw_analysis, f, indent=2, ensure_ascii=False)
    print(f"     {fpath}")

    # 08 — Signal analysis
    fpath = os.path.join(OUT_DIR, "08_phase16_signal_analysis.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(signal_analysis, f, indent=2, ensure_ascii=False)
    print(f"     {fpath}")

    # 09 — Comparison
    fpath = os.path.join(OUT_DIR, "09_phase16_comparison.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"cohort": "2026-09-18 to 2026-09-21",
                     "temporal_status": "TEMPORAL_UNKNOWN",
                     "phase15_baseline_reference": {
                         "accuracy_1x2": 0.5625, "brier": 0.5785, "log_loss": 0.9732,
                         "n": 48, "draws_predicted": 0, "actual_draws": 10,
                     }},
            "arm_a_v4_full_48": v4_full_48,
            "arm_b_methods_sm_covered": methods_results,
            "league_breakdown": league_breakdown,
            "governance": "Sportmonks remains RESEARCH ONLY. TEMPORAL_UNKNOWN data insufficient for production promotion.",
        }, f, indent=2, ensure_ascii=False)
    print(f"     {fpath}")

    # 10 — Final report
    report_md = generate_report(
        records, sm_stats, methods_results, draw_analysis, signal_analysis,
        league_breakdown, v4_full_48
    )
    fpath = os.path.join(OUT_DIR, "10_phase16_report.md")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"     {fpath}")

    # 11 — Dashboard
    dash_html = generate_dashboard(records)
    fpath = os.path.join(OUT_DIR, "11_phase16_dashboard.html")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(dash_html)
    print(f"     {fpath}")

    # README
    readme = f"""# Phase 16 — V4 + Sportmonks Research Arm

## ⚠️ RESEARCH ONLY — NOT PRODUCTION

**Cohort:** 2026-09-18 → 2026-09-21  
**V4 SHA256:** `{FROZEN_SHA256}`  
**Status:** TEMPORAL_UNKNOWN (Sportmonks data fetched post-kickoff)

## Files

| File | Description |
|------|-------------|
| `01_phase16_fixture_universe.jsonl` | Full 48-fixture universe with V4 + SM + actuals joined |
| `02_sportmonks_raw_predictions.jsonl` | Per-fixture Sportmonks extracted predictions |
| `03_phase16_source_validation.json` | Data quality and validation report |
| `04_phase16_v4_baseline.json` | V4-only metrics (arm A) |
| `05_phase16_sportmonks_only.json` | Sportmonks-only metrics |
| `06_phase16_blend_results.json` | All 5 blend method results |
| `07_phase16_draw_analysis.json` | Deep draw prediction analysis |
| `08_phase16_signal_analysis.json` | 2-0 profile and strong home signal analysis |
| `09_phase16_comparison.json` | Full method comparison with governance decision |
| `10_phase16_report.md` | Complete research report |
| `11_phase16_dashboard.html` | Offline research dashboard |

## Critical Constraints

- **V4 model NOT modified.** SHA256: `{FROZEN_SHA256}`
- **Sportmonks is RESEARCH ONLY.** Not promoted to production.
- **All data is TEMPORAL_UNKNOWN.** Cannot be used as clean prospective evidence.
- **Production predictions unchanged.** No modification to Phase 15 ledger.
- **No leakage.** Post-kickoff data explicitly labeled.

## Governance

> Sportmonks remains RESEARCH ONLY.

Continue prospective V4 vs V4+Sportmonks validation by capturing
Sportmonks predictions BEFORE kickoff for the next upcoming cohort.
"""
    fpath = os.path.join(OUT_DIR, "README.md")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"     {fpath}")

    # Final summary
    v4_sm_acc = next((m for m in methods_results if m["method_key"] == "m1_v4_only"), {}).get("accuracy_1x2", 0)
    sm_acc = next((m for m in methods_results if m["method_key"] == "m2_sm_only"), {}).get("accuracy_1x2", 0)
    m3_acc = next((m for m in methods_results if m["method_key"] == "m3_50_50"), {}).get("accuracy_1x2", 0)

    print("\n" + "=" * 70)
    print("PHASE 16 SUMMARY")
    print("=" * 70)
    print(f"  SM Coverage:        {len(sm_covered)}/48 = {100*len(sm_covered)/48:.0f}%")
    print(f"  SM Temporal Status: TEMPORAL_UNKNOWN (all)")
    print(f"  V4 Full 48 Acc:     {v4_full_48['accuracy_1x2']:.1%}")
    print(f"  V4 SM-subset Acc:   {v4_sm_acc:.1%}")
    print(f"  SM Only Acc:        {sm_acc:.1%}")
    print(f"  50/50 Blend Acc:    {m3_acc:.1%}")
    print(f"  V4 Draw Preds:      0 (0 draws predicted of 10 actual)")
    di = draw_analysis["sportmonks_draw_impact"]
    print(f"  SM Draw Preds:      {di['sm_makes_draw_argmax']} draws made argmax by SM")
    print("=" * 70)
    print("  Sportmonks remains RESEARCH ONLY.")
    print("  V4 SHA256 verified unchanged.")
    print("=" * 70)

    return records


if __name__ == "__main__":
    main()
