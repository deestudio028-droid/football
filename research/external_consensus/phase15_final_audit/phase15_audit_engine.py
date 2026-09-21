"""
Phase 15 Final Results Audit Engine
====================================
ANALYSIS ONLY — Does NOT modify V4 model, predictions, or probabilities.

Inputs:
  - reports/upcoming_2026_09_18_to_2026_09_21_ledger.jsonl   (48-fixture immutable ledger)
  - data/phase15_oddalerts_raw.json                           (OddAlerts FT results for all 48 fixtures)

Outputs (in research/external_consensus/phase15_final_audit/):
  01_phase15_final_results_ledger.jsonl
  02_phase15_metrics.json
  03_phase15_risk_analysis.json
  04_phase15_signal_analysis.json
  05_phase15_league_analysis.json
  06_phase15_forensics.json
  07_phase15_final_report.md
  08_phase15_final_dashboard.html
  README.md
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LEDGER_PATH = os.path.join(ROOT, "reports", "upcoming_2026_09_18_to_2026_09_21_ledger.jsonl")
RAW_RESULTS_PATH = os.path.join(ROOT, "data", "phase15_oddalerts_raw.json")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FROZEN_SHA256 = "1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_ledger():
    records = []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_raw_results():
    with open(RAW_RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def parse_score(score_str):
    """Parse 'H-A' string to (int, int). Returns (None, None) on failure."""
    if not score_str or score_str in ("—", "N/A", ""):
        return None, None
    try:
        h, a = score_str.split("-")
        return int(h.strip()), int(a.strip())
    except Exception:
        return None, None


def outcome_1x2(home_goals, away_goals):
    """Return 'H', 'D', or 'A'."""
    if home_goals > away_goals:
        return "H"
    elif home_goals == away_goals:
        return "D"
    else:
        return "A"


def brier_score(p_h, p_d, p_a, actual_outcome):
    """One-vs-rest Brier score for a 3-class outcome."""
    o_h = 1 if actual_outcome == "H" else 0
    o_d = 1 if actual_outcome == "D" else 0
    o_a = 1 if actual_outcome == "A" else 0
    return (p_h - o_h) ** 2 + (p_d - o_d) ** 2 + (p_a - o_a) ** 2


def log_loss_single(p_correct):
    """Log loss for a single prediction given the probability assigned to the correct outcome."""
    p = max(min(p_correct, 1 - 1e-9), 1e-9)
    return -math.log(p)


def safe_div(a, b):
    return a / b if b else 0.0


def get_total_goals(home_goals, away_goals):
    return home_goals + away_goals if home_goals is not None and away_goals is not None else None


# ─── Main Audit ───────────────────────────────────────────────────────────────

def run_audit():
    print("=" * 70)
    print("Phase 15 Final Results Audit Engine")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    # 1. Load prediction ledger (immutable)
    ledger = load_ledger()
    print(f"\n[1] Loaded {len(ledger)} prediction records from immutable ledger")

    # 2. Verify V4 SHA256
    shas = {r["model_v4_sha256"] for r in ledger}
    assert shas == {FROZEN_SHA256}, f"SHA256 mismatch: {shas}"
    print(f"[2] V4 SHA256 verified: {FROZEN_SHA256}")

    # 3. Load actual results
    raw_results = load_raw_results()
    results_by_fid = {str(r["id"]): r for r in raw_results}
    print(f"[3] Loaded {len(results_by_fid)} actual result records from OddAlerts")

    # 4. Join and classify fixtures
    joined = []
    stats = {"ft": 0, "upcoming": 0, "postponed": 0, "error": 0}

    for pred in ledger:
        fid = str(pred["fixture_id"])
        actual = results_by_fid.get(fid)

        record = {
            "fixture_id": fid,
            "date": pred["Date (UTC)"],
            "league": pred["League"],
            "matchweek": pred["Matchweek"],
            "home_team": pred["Home Team"],
            "away_team": pred["Away Team"],
            # Predictions (immutable)
            "pred_outcome": pred["PRED"],
            "p_home": pred["p_home"],
            "p_draw": pred["p_draw"],
            "p_away": pred["p_away"],
            "canonical_predicted_score": pred["canonical_predicted_score"],
            "lambda_home": pred["lambda_home"],
            "lambda_away": pred["lambda_away"],
            "draw_risk_tier": pred["draw_risk_tier"],
            "draw_risk_score": pred["draw_risk_score"],
            "strong_home_profile": pred["strong_home_profile"],
            "is_2_0_profile": pred["is_2_0_profile"],
            "signal_profile": pred["signal_profile"],
            "modal_scoreline_probability": pred["modal_scoreline_probability"],
            "model_v4_sha256": pred["model_v4_sha256"],
            # Actuals (from OddAlerts)
            "actual_status": None,
            "actual_home_goals": None,
            "actual_away_goals": None,
            "actual_score": None,
            "actual_outcome": None,
            "ht_score": None,
            "winning_team": None,
            # Derived
            "outcome_correct": None,
            "exact_score_correct": None,
            "brier": None,
            "log_loss": None,
            "p_correct_outcome": None,
            "goal_diff": None,
            "total_goals": None,
            "classification": None,  # ft / upcoming / postponed
        }

        if actual is None:
            record["classification"] = "error"
            stats["error"] += 1
        elif actual["status"] in ("POSTP", "CANC", "SUSP", "ABN"):
            record["classification"] = "postponed"
            record["actual_status"] = actual["status"]
            stats["postponed"] += 1
        elif actual["status"] == "FT" and actual.get("home_goals") is not None:
            hg = actual["home_goals"]
            ag = actual["away_goals"]
            record["classification"] = "ft"
            record["actual_status"] = "FT"
            record["actual_home_goals"] = hg
            record["actual_away_goals"] = ag
            record["actual_score"] = f"{hg}-{ag}"
            record["actual_outcome"] = outcome_1x2(hg, ag)
            record["ht_score"] = actual.get("ht_score")
            record["winning_team"] = actual.get("winning_team")
            record["goal_diff"] = hg - ag
            record["total_goals"] = hg + ag

            # Accuracy
            record["outcome_correct"] = record["actual_outcome"] == record["pred_outcome"]
            ph, pg = parse_score(record["canonical_predicted_score"])
            record["exact_score_correct"] = (ph == hg and pg == ag)

            # Probabilistic metrics
            p_h = pred["p_home"]
            p_d = pred["p_draw"]
            p_a = pred["p_away"]
            actual_out = record["actual_outcome"]
            record["brier"] = round(brier_score(p_h, p_d, p_a, actual_out), 6)

            p_correct = {"H": p_h, "D": p_d, "A": p_a}[actual_out]
            record["p_correct_outcome"] = round(p_correct, 6)
            record["log_loss"] = round(log_loss_single(p_correct), 6)

            stats["ft"] += 1
        else:
            # Still UPCOMING / LIVE at time of audit (shouldn't happen for Sep 18-20 games)
            record["classification"] = "upcoming"
            record["actual_status"] = actual.get("status", "UPCOMING")
            stats["upcoming"] += 1

        joined.append(record)

    print(f"\n[4] Fixture classification:")
    print(f"    FT (completed):  {stats['ft']}")
    print(f"    Upcoming:        {stats['upcoming']}")
    print(f"    Postponed:       {stats['postponed']}")
    print(f"    Errors:          {stats['error']}")

    ft_records = [r for r in joined if r["classification"] == "ft"]

    # ─── 5. Compute metrics A-R ───────────────────────────────────────────────
    print(f"\n[5] Computing metrics on {len(ft_records)} completed fixtures...")

    # A. 1X2 accuracy
    correct_outcomes = [r for r in ft_records if r["outcome_correct"]]
    accuracy_1x2 = safe_div(len(correct_outcomes), len(ft_records))

    outcome_dist = defaultdict(int)
    pred_dist = defaultdict(int)
    for r in ft_records:
        outcome_dist[r["actual_outcome"]] += 1
        pred_dist[r["pred_outcome"]] += 1

    # B. Exact score accuracy
    exact_correct = [r for r in ft_records if r["exact_score_correct"]]
    accuracy_exact = safe_div(len(exact_correct), len(ft_records))

    # C. Brier score
    brier_scores = [r["brier"] for r in ft_records]
    mean_brier = safe_div(sum(brier_scores), len(brier_scores))

    # D. Log loss
    log_losses = [r["log_loss"] for r in ft_records]
    mean_log_loss = safe_div(sum(log_losses), len(log_losses))

    # E. Draw risk analysis
    draw_risk_stats = {}
    for tier in ("LOW", "MEDIUM", "HIGH"):
        tier_recs = [r for r in ft_records if r["draw_risk_tier"] == tier]
        draws_in_tier = [r for r in tier_recs if r["actual_outcome"] == "D"]
        draw_risk_stats[tier] = {
            "count": len(tier_recs),
            "actual_draws": len(draws_in_tier),
            "draw_rate": round(safe_div(len(draws_in_tier), len(tier_recs)), 4),
        }

    # F. 2-0 signal analysis
    signal_2_0 = [r for r in ft_records if r["is_2_0_profile"]]
    signal_2_0_exact = [r for r in signal_2_0 if r["exact_score_correct"]]
    signal_2_0_correct_outcome = [r for r in signal_2_0 if r["outcome_correct"]]

    # G. Strong home signal analysis
    strong_home = [r for r in ft_records if r["strong_home_profile"]]
    strong_home_correct = [r for r in strong_home if r["outcome_correct"]]

    # H. High-confidence fixtures (P(H) or P(A) > 0.65)
    high_conf = [r for r in ft_records if r["p_home"] > 0.65 or r["p_away"] > 0.65]
    high_conf_correct = [r for r in high_conf if r["outcome_correct"]]

    # I. Scoreline distribution
    pred_scores = defaultdict(int)
    actual_scores = defaultdict(int)
    for r in ft_records:
        pred_scores[r["canonical_predicted_score"]] += 1
        actual_scores[r["actual_score"]] += 1

    # J. Calibration
    calibration = {}
    for bucket_label, lo, hi in [
        ("<30%", 0.0, 0.30),
        ("30-50%", 0.30, 0.50),
        ("50-65%", 0.50, 0.65),
        ("65-80%", 0.65, 0.80),
        (">80%", 0.80, 1.01),
    ]:
        bucket_recs = [r for r in ft_records if lo <= r["p_home"] < hi or lo <= r["p_away"] < hi]
        bucket_correct = [r for r in bucket_recs if r["outcome_correct"]]
        mean_p_correct = safe_div(
            sum(r["p_correct_outcome"] for r in bucket_recs), len(bucket_recs)
        )
        calibration[bucket_label] = {
            "count": len(bucket_recs),
            "correct": len(bucket_correct),
            "accuracy": round(safe_div(len(bucket_correct), len(bucket_recs)), 4),
            "mean_p_correct": round(mean_p_correct, 4),
        }

    # K. League breakdown
    league_stats = defaultdict(lambda: {
        "count": 0, "correct": 0, "exact": 0,
        "brier_sum": 0.0, "log_loss_sum": 0.0
    })
    for r in ft_records:
        lg = r["league"]
        league_stats[lg]["count"] += 1
        if r["outcome_correct"]:
            league_stats[lg]["correct"] += 1
        if r["exact_score_correct"]:
            league_stats[lg]["exact"] += 1
        league_stats[lg]["brier_sum"] += r["brier"]
        league_stats[lg]["log_loss_sum"] += r["log_loss"]

    league_summary = {}
    for lg, s in league_stats.items():
        n = s["count"]
        league_summary[lg] = {
            "count": n,
            "accuracy_1x2": round(safe_div(s["correct"], n), 4),
            "accuracy_exact": round(safe_div(s["exact"], n), 4),
            "mean_brier": round(safe_div(s["brier_sum"], n), 4),
            "mean_log_loss": round(safe_div(s["log_loss_sum"], n), 4),
        }

    # L. Biggest misses (predicted H/A but wrong, sorted by confidence)
    misses = [r for r in ft_records if not r["outcome_correct"]]
    misses_sorted = sorted(
        misses,
        key=lambda r: max(r["p_home"], r["p_away"]),
        reverse=True
    )

    # M. Best signals (correct predictions at highest confidence)
    best_signals = sorted(
        [r for r in ft_records if r["outcome_correct"]],
        key=lambda r: r["p_correct_outcome"],
        reverse=True
    )[:10]

    # N. Phase 14 canonical fix validation
    # Correct canonical: modal_scoreline not rounding
    canonical_check = {
        "fixtures_with_canonical_score": len([r for r in ft_records if r["canonical_predicted_score"]]),
        "2_0_predictions": len([r for r in ft_records if r["canonical_predicted_score"] == "2-0"]),
        "exact_2_0_hit": len([r for r in ft_records if r["canonical_predicted_score"] == "2-0" and r["actual_score"] == "2-0"]),
        "note": "Phase 14 canonical fix: modal_scoreline() used instead of rounding lambda values."
    }

    # O. Live score data integrity
    live_data_check = {
        "total_queried": len(joined),
        "ft_returned": stats["ft"],
        "upcoming_remaining": stats["upcoming"],
        "postponed": stats["postponed"],
        "errors": stats["error"],
        "completeness_pct": round(100 * stats["ft"] / len(joined), 2),
        "source": "OddAlerts /fixtures/multiple",
        "query_timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # P. Sportmonks exclusion note
    sportmonks_note = {
        "excluded": True,
        "reason": "Phase 15 production predictions were generated WITHOUT Sportmonks data.",
        "v4_sha256_unchanged": FROZEN_SHA256,
    }

    # Q. Model change decision
    model_change_decision = {
        "accuracy_1x2": round(accuracy_1x2, 4),
        "mean_brier": round(mean_brier, 4),
        "mean_log_loss": round(mean_log_loss, 4),
        "principle": "ONE WEEK / ONE COHORT SHOULD NOT AUTOMATICALLY TRIGGER MODEL RETRAINING.",
        "recommendation": "HOLD — analysis only. No retraining warranted from a single cohort.",
        "rationale": (
            f"With {len(ft_records)} fixtures, accuracy={round(accuracy_1x2,4):.1%}. "
            "Systematic analysis of multiple cohorts required before any model update."
        ),
    }

    # R. Phase-over-phase comparison
    phase_comparison = {
        "phase15": {
            "cohort": "2026-09-18 to 2026-09-21",
            "fixtures_completed": len(ft_records),
            "accuracy_1x2": round(accuracy_1x2, 4),
            "accuracy_exact": round(accuracy_exact, 4),
            "mean_brier": round(mean_brier, 4),
            "mean_log_loss": round(mean_log_loss, 4),
        },
        "phase14_reference": {
            "cohort": "2026-09-10 to 2026-09-16",
            "note": "Phase 14 reference metrics would go here from 02_phase14_metrics.json",
            "accuracy_1x2": "see phase14 audit",
        },
        "note": "Phase-over-phase requires Phase 14 audit to be available.",
    }

    # ─── 6. Assemble output structures ────────────────────────────────────────
    metrics = {
        "meta": {
            "audit_timestamp": datetime.utcnow().isoformat() + "Z",
            "cohort": "2026-09-18 to 2026-09-21",
            "leagues": ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"],
            "total_fixtures": len(joined),
            "fixtures_completed": stats["ft"],
            "fixtures_upcoming": stats["upcoming"],
            "fixtures_postponed": stats["postponed"],
            "frozen_v4_sha256": FROZEN_SHA256,
            "sportmonks_excluded": True,
        },
        "A_1x2_accuracy": {
            "correct": len(correct_outcomes),
            "total": len(ft_records),
            "accuracy": round(accuracy_1x2, 4),
            "actual_outcome_distribution": dict(outcome_dist),
            "predicted_outcome_distribution": dict(pred_dist),
        },
        "B_exact_score": {
            "correct": len(exact_correct),
            "total": len(ft_records),
            "accuracy": round(accuracy_exact, 4),
            "exact_hits": [
                {"fixture_id": r["fixture_id"], "home": r["home_team"], "away": r["away_team"],
                 "score": r["actual_score"], "league": r["league"]}
                for r in exact_correct
            ],
        },
        "C_brier_score": {
            "mean": round(mean_brier, 6),
            "min": round(min(brier_scores), 6),
            "max": round(max(brier_scores), 6),
            "note": "Lower is better. Perfect = 0, Random baseline ≈ 0.667",
        },
        "D_log_loss": {
            "mean": round(mean_log_loss, 6),
            "min": round(min(log_losses), 6),
            "max": round(max(log_losses), 6),
            "note": "Lower is better. Perfect = 0.",
        },
        "H_high_confidence": {
            "count": len(high_conf),
            "correct": len(high_conf_correct),
            "accuracy": round(safe_div(len(high_conf_correct), len(high_conf)), 4),
        },
        "I_scoreline_distribution": {
            "predicted": dict(sorted(pred_scores.items(), key=lambda x: -x[1])[:15]),
            "actual": dict(sorted(actual_scores.items(), key=lambda x: -x[1])[:15]),
        },
        "J_calibration": calibration,
        "N_phase14_canonical_fix": canonical_check,
        "O_live_data_integrity": live_data_check,
        "P_sportmonks": sportmonks_note,
        "Q_model_change_decision": model_change_decision,
        "R_phase_comparison": phase_comparison,
    }

    risk_analysis = {
        "meta": {"cohort": "2026-09-18 to 2026-09-21"},
        "E_draw_risk_by_tier": draw_risk_stats,
        "overall_draws": {
            "actual": outcome_dist.get("D", 0),
            "total": len(ft_records),
            "rate": round(safe_div(outcome_dist.get("D", 0), len(ft_records)), 4),
        },
        "draw_risk_score_stats": {
            "mean": round(safe_div(sum(r["draw_risk_score"] for r in ft_records), len(ft_records)), 4),
            "by_tier": {
                t: round(safe_div(
                    sum(r["draw_risk_score"] for r in ft_records if r["draw_risk_tier"] == t),
                    max(1, len([r for r in ft_records if r["draw_risk_tier"] == t]))
                ), 4)
                for t in ("LOW", "MEDIUM", "HIGH")
            }
        }
    }

    signal_analysis = {
        "meta": {"cohort": "2026-09-18 to 2026-09-21"},
        "F_2_0_signal": {
            "fixtures": len(signal_2_0),
            "correct_outcome": len(signal_2_0_correct_outcome),
            "exact_score": len(signal_2_0_exact),
            "outcome_accuracy": round(safe_div(len(signal_2_0_correct_outcome), len(signal_2_0)), 4),
            "exact_accuracy": round(safe_div(len(signal_2_0_exact), len(signal_2_0)), 4),
            "details": [
                {"fixture_id": r["fixture_id"], "home": r["home_team"], "away": r["away_team"],
                 "pred": r["canonical_predicted_score"], "actual": r["actual_score"],
                 "outcome_ok": r["outcome_correct"], "exact_ok": r["exact_score_correct"],
                 "lambda_home": r["lambda_home"], "lambda_away": r["lambda_away"],
                 "p_home": r["p_home"]}
                for r in signal_2_0
            ],
        },
        "G_strong_home_signal": {
            "fixtures": len(strong_home),
            "correct_outcome": len(strong_home_correct),
            "outcome_accuracy": round(safe_div(len(strong_home_correct), len(strong_home)), 4),
            "details": [
                {"fixture_id": r["fixture_id"], "home": r["home_team"], "away": r["away_team"],
                 "pred": r["pred_outcome"], "actual": r["actual_outcome"],
                 "pred_score": r["canonical_predicted_score"], "actual_score": r["actual_score"],
                 "is_2_0_profile": r["is_2_0_profile"], "p_home": r["p_home"]}
                for r in strong_home
            ],
        },
        "M_best_signals": [
            {"fixture_id": r["fixture_id"], "home": r["home_team"], "away": r["away_team"],
             "pred": r["pred_outcome"], "actual": r["actual_outcome"],
             "p_correct": r["p_correct_outcome"], "score_pred": r["canonical_predicted_score"],
             "score_actual": r["actual_score"]}
            for r in best_signals
        ],
    }

    league_analysis = {
        "meta": {"cohort": "2026-09-18 to 2026-09-21"},
        "K_league_breakdown": league_summary,
    }

    forensics = {
        "meta": {"cohort": "2026-09-18 to 2026-09-21"},
        "L_biggest_misses": [
            {"fixture_id": r["fixture_id"], "home": r["home_team"], "away": r["away_team"],
             "pred_outcome": r["pred_outcome"], "actual_outcome": r["actual_outcome"],
             "pred_score": r["canonical_predicted_score"], "actual_score": r["actual_score"],
             "p_home": r["p_home"], "p_draw": r["p_draw"], "p_away": r["p_away"],
             "p_correct": r["p_correct_outcome"], "log_loss": r["log_loss"],
             "league": r["league"], "draw_risk_tier": r["draw_risk_tier"]}
            for r in misses_sorted[:15]
        ],
        "total_misses": len(misses),
        "total_ft": len(ft_records),
    }

    # ─── 7. Save output files ─────────────────────────────────────────────────

    # 01 — Final results ledger (JSONL)
    ledger_out_path = os.path.join(OUT_DIR, "01_phase15_final_results_ledger.jsonl")
    with open(ledger_out_path, "w", encoding="utf-8") as f:
        for r in joined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[7] Saved: {ledger_out_path}")

    # 02 — Metrics
    metrics_path = os.path.join(OUT_DIR, "02_phase15_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {metrics_path}")

    # 03 — Risk analysis
    risk_path = os.path.join(OUT_DIR, "03_phase15_risk_analysis.json")
    with open(risk_path, "w", encoding="utf-8") as f:
        json.dump(risk_analysis, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {risk_path}")

    # 04 — Signal analysis
    signal_path = os.path.join(OUT_DIR, "04_phase15_signal_analysis.json")
    with open(signal_path, "w", encoding="utf-8") as f:
        json.dump(signal_analysis, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {signal_path}")

    # 05 — League analysis
    league_path = os.path.join(OUT_DIR, "05_phase15_league_analysis.json")
    with open(league_path, "w", encoding="utf-8") as f:
        json.dump(league_analysis, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {league_path}")

    # 06 — Forensics
    forensics_path = os.path.join(OUT_DIR, "06_phase15_forensics.json")
    with open(forensics_path, "w", encoding="utf-8") as f:
        json.dump(forensics, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {forensics_path}")

    return joined, metrics, risk_analysis, signal_analysis, league_analysis, forensics, ft_records


def generate_report(metrics, risk_analysis, signal_analysis, league_analysis, forensics, ft_records):
    """Generate the markdown final report."""
    m = metrics
    A = m["A_1x2_accuracy"]
    B = m["B_exact_score"]
    C = m["C_brier_score"]
    D = m["D_log_loss"]
    H = m["H_high_confidence"]
    E = risk_analysis["E_draw_risk_by_tier"]
    F = signal_analysis["F_2_0_signal"]
    G = signal_analysis["G_strong_home_signal"]
    K = league_analysis["K_league_breakdown"]
    L = forensics["L_biggest_misses"]

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    cohort = "2026-09-18 → 2026-09-21"
    n = A["total"]
    acc = A["accuracy"]

    lines = [
        "# Phase 15 Final Results Audit Report",
        f"",
        f"**Cohort:** {cohort} | **Leagues:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1  ",
        f"**Generated:** {now}  ",
        f"**V4 SHA256:** `{FROZEN_SHA256}`  ",
        f"**Status:** ANALYSIS ONLY — V4 model NOT modified",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Fixtures completed | {n} / 48 |",
        f"| 1X2 Accuracy | **{acc:.1%}** ({A['correct']}/{n}) |",
        f"| Exact Score Accuracy | **{B['accuracy']:.1%}** ({B['correct']}/{n}) |",
        f"| Mean Brier Score | {C['mean']:.4f} |",
        f"| Mean Log Loss | {D['mean']:.4f} |",
        f"| High-Confidence Accuracy | {H['accuracy']:.1%} ({H['correct']}/{H['count']}) |",
        "",
        "---",
        "",
        "## A. 1X2 Accuracy",
        "",
        f"**{A['correct']}/{n} = {acc:.1%}** correct 1X2 predictions.",
        "",
        "**Actual outcome distribution:**",
        "",
    ]
    od = A["actual_outcome_distribution"]
    pd_ = A["predicted_outcome_distribution"]
    lines += [
        f"| Outcome | Actual | Predicted |",
        f"|---------|--------|-----------|",
        f"| Home Win (H) | {od.get('H', 0)} | {pd_.get('H', 0)} |",
        f"| Draw (D) | {od.get('D', 0)} | {pd_.get('D', 0)} |",
        f"| Away Win (A) | {od.get('A', 0)} | {pd_.get('A', 0)} |",
        "",
    ]

    lines += [
        "## B. Exact Score Accuracy",
        "",
        f"**{B['correct']}/{n} = {B['accuracy']:.1%}** exact score hits.",
        "",
    ]
    if B["exact_hits"]:
        lines += ["**Exact score hits:**", ""]
        lines += ["| Fixture | Score | League |", "|---------|-------|--------|"]
        for h in B["exact_hits"]:
            lines.append(f"| {h['home']} vs {h['away']} | **{h['score']}** | {h['league']} |")
        lines.append("")
    else:
        lines += ["No exact score hits this cohort.", ""]

    lines += [
        "## C. Brier Score",
        "",
        f"Mean Brier: **{C['mean']:.4f}** (range: {C['min']:.4f}–{C['max']:.4f})",
        "",
        "> Lower is better. Perfect prediction = 0. Random 3-class baseline ≈ 0.667.",
        "",
        "## D. Log Loss",
        "",
        f"Mean Log Loss: **{D['mean']:.4f}** (range: {D['min']:.4f}–{D['max']:.4f})",
        "",
        "## E. Draw Risk Analysis",
        "",
        "| Tier | Fixtures | Actual Draws | Draw Rate |",
        "|------|----------|-------------|-----------|",
    ]
    for tier in ("LOW", "MEDIUM", "HIGH"):
        s = E.get(tier, {"count": 0, "actual_draws": 0, "draw_rate": 0.0})
        lines.append(f"| {tier} | {s['count']} | {s['actual_draws']} | {s['draw_rate']:.1%} |")
    lines.append("")

    lines += [
        "## F. 2-0 Signal Analysis",
        "",
        f"**{F['fixtures']} fixture(s)** flagged as `2-0_PROFILE`:",
        "",
        f"- Outcome accuracy: **{F['outcome_accuracy']:.1%}** ({F['correct_outcome']}/{F['fixtures']})",
        f"- Exact score accuracy: **{F['exact_accuracy']:.1%}** ({F['exact_score']}/{F['fixtures']})",
        "",
        "| Fixture | λ_H | λ_A | P(H) | Pred | Actual | Outcome ✓ | Exact ✓ |",
        "|---------|-----|-----|------|------|--------|-----------|---------|",
    ]
    for d in F["details"]:
        ok_out = "✅" if d["outcome_ok"] else "❌"
        ok_ex = "✅" if d["exact_ok"] else "❌"
        lines.append(
            f"| {d['home']} vs {d['away']} | {d['lambda_home']:.2f} | {d['lambda_away']:.2f} "
            f"| {d['p_home']:.1%} | {d['pred']} | {d['actual']} | {ok_out} | {ok_ex} |"
        )
    lines.append("")

    lines += [
        "## G. Strong Home Signal Analysis",
        "",
        f"**{G['fixtures']} fixture(s)** with `strong_home_profile=True`:",
        f"- Outcome accuracy: **{G['outcome_accuracy']:.1%}** ({G['correct_outcome']}/{G['fixtures']})",
        "",
        "| Fixture | Pred | Actual | Score Pred | Score Actual | ✓ |",
        "|---------|------|--------|------------|-------------|---|",
    ]
    for d in G["details"]:
        ok = "✅" if d["pred"] == d["actual"] else "❌"
        lines.append(
            f"| {d['home']} vs {d['away']} | {d['pred']} | {d['actual']} "
            f"| {d['pred_score']} | {d['actual_score']} | {ok} |"
        )
    lines.append("")

    lines += [
        "## H. High-Confidence Predictions (>65%)",
        "",
        f"**{H['count']} fixtures**, accuracy: **{H['accuracy']:.1%}** ({H['correct']}/{H['count']})",
        "",
        "## K. League Breakdown",
        "",
        "| League | N | 1X2 Acc | Exact Acc | Brier | Log Loss |",
        "|--------|---|---------|-----------|-------|----------|",
    ]
    for lg, s in sorted(K.items()):
        lines.append(
            f"| {lg} | {s['count']} | {s['accuracy_1x2']:.1%} | {s['accuracy_exact']:.1%} "
            f"| {s['mean_brier']:.4f} | {s['mean_log_loss']:.4f} |"
        )
    lines.append("")

    lines += [
        "## L. Biggest Misses (Top 15 by Confidence)",
        "",
        "| Fixture | Pred | Actual | P(pred) | Score Pred | Score Actual | League |",
        "|---------|------|--------|---------|------------|-------------|--------|",
    ]
    for r in L[:15]:
        lines.append(
            f"| {r['home']} vs {r['away']} | **{r['pred_outcome']}** | {r['actual_outcome']} "
            f"| {r['p_home'] if r['pred_outcome']=='H' else r['p_away'] if r['pred_outcome']=='A' else r['p_draw']:.1%} "
            f"| {r['pred_score']} | {r['actual_score']} | {r['league']} |"
        )
    lines.append("")

    lines += [
        "## N. Phase 14 Canonical Fix Validation",
        "",
        f"- Fixtures using `canonical_predicted_score`: **{metrics['N_phase14_canonical_fix']['fixtures_with_canonical_score']}**",
        f"- Predictions of `2-0`: **{metrics['N_phase14_canonical_fix']['2_0_predictions']}**",
        f"- Exact `2-0` hits: **{metrics['N_phase14_canonical_fix']['exact_2_0_hit']}**",
        "",
        "## Q. Model Change Decision",
        "",
        f"> **{metrics['Q_model_change_decision']['recommendation']}**",
        "",
        f"{metrics['Q_model_change_decision']['rationale']}",
        "",
        "---",
        "",
        "## Appendix — Frozen V4 Model",
        "",
        f"```",
        f"model_v4_sha256: {FROZEN_SHA256}",
        f"```",
        "",
        "*This audit is read-only. V4 model was NOT modified, retrained, or re-parameterized.*",
    ]

    return "\n".join(lines)


def generate_dashboard(joined, metrics, ft_records):
    """Generate the Phase 15 final results HTML dashboard."""
    n = len(ft_records)
    acc = metrics["A_1x2_accuracy"]["accuracy"]
    exact_acc = metrics["B_exact_score"]["accuracy"]
    brier = metrics["C_brier_score"]["mean"]
    log_loss = metrics["D_log_loss"]["mean"]
    h_count = metrics["A_1x2_accuracy"]["actual_outcome_distribution"].get("H", 0)
    d_count = metrics["A_1x2_accuracy"]["actual_outcome_distribution"].get("D", 0)
    a_count = metrics["A_1x2_accuracy"]["actual_outcome_distribution"].get("A", 0)

    correct_n = metrics["A_1x2_accuracy"]["correct"]

    rows = []
    for r in joined:
        cls = r["classification"]
        if cls == "ft":
            outcome_ok = "✅" if r["outcome_correct"] else "❌"
            exact_ok = "✅" if r["exact_score_correct"] else "❌"
            badge_cls = "badge-correct" if r["outcome_correct"] else "badge-wrong"
            rows.append(f"""
            <tr class="{'row-correct' if r['outcome_correct'] else 'row-wrong'}">
              <td>{r['date']}</td>
              <td><span class="league-badge">{r['league']}</span></td>
              <td>{r['home_team']}</td>
              <td>{r['away_team']}</td>
              <td>{r['p_home']:.1%}</td>
              <td>{r['p_draw']:.1%}</td>
              <td>{r['p_away']:.1%}</td>
              <td><strong>{r['pred_outcome']}</strong></td>
              <td>{r['canonical_predicted_score']}</td>
              <td><strong>{r['actual_outcome']}</strong></td>
              <td>{r['actual_score']}</td>
              <td><span class="{badge_cls}">{outcome_ok}</span></td>
              <td>{exact_ok}</td>
              <td>{r['draw_risk_tier']}</td>
              <td>{r['signal_profile']}</td>
            </tr>""")
        else:
            status_label = r.get("actual_status") or cls.upper()
            rows.append(f"""
            <tr class="row-pending">
              <td>{r['date']}</td>
              <td><span class="league-badge">{r['league']}</span></td>
              <td>{r['home_team']}</td>
              <td>{r['away_team']}</td>
              <td>{r['p_home']:.1%}</td>
              <td>{r['p_draw']:.1%}</td>
              <td>{r['p_away']:.1%}</td>
              <td><strong>{r['pred_outcome']}</strong></td>
              <td>{r['canonical_predicted_score']}</td>
              <td colspan="6"><em>{status_label}</em></td>
            </tr>""")

    rows_html = "\n".join(rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 15 Final Results Audit | 2026-09-18 to 2026-09-21</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  h2 {{ color: #8b949e; margin-top: 28px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }}
  .sha {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 8px 14px;
          font-family: monospace; font-size: 0.8em; color: #79c0ff; display: inline-block; margin-bottom: 20px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 28px; }}
  .metric-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
  .metric-card .value {{ font-size: 2em; font-weight: bold; color: #58a6ff; }}
  .metric-card .label {{ font-size: 0.8em; color: #8b949e; margin-top: 4px; }}
  .metric-card.good .value {{ color: #3fb950; }}
  .metric-card.warn .value {{ color: #d29922; }}
  .metric-card.info .value {{ color: #58a6ff; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.82em; }}
  th {{ background: #161b22; color: #8b949e; padding: 8px 10px; text-align: left;
         border-bottom: 2px solid #30363d; position: sticky; top: 0; z-index: 10; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #21262d; vertical-align: middle; }}
  tr.row-correct {{ background: rgba(63, 185, 80, 0.06); }}
  tr.row-wrong {{ background: rgba(248, 81, 73, 0.07); }}
  tr.row-pending {{ background: rgba(88, 166, 255, 0.05); }}
  tr:hover {{ background: #1c2128 !important; }}
  .badge-correct {{ background: #1a4726; color: #3fb950; border-radius: 4px; padding: 2px 8px; font-weight: bold; }}
  .badge-wrong {{ background: #4a1419; color: #f85149; border-radius: 4px; padding: 2px 8px; font-weight: bold; }}
  .league-badge {{ background: #1f2937; color: #93c5fd; border-radius: 12px; padding: 2px 8px; font-size: 0.82em; }}
  .frozen-note {{ background: #161b22; border-left: 4px solid #3fb950; padding: 10px 16px;
                  border-radius: 4px; margin-bottom: 20px; font-size: 0.85em; color: #8b949e; }}
  .table-wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>⚽ Phase 15 Final Results Audit</h1>
<div class="subtitle">Cohort: 2026-09-18 → 2026-09-21 &nbsp;|&nbsp; Big-5 Leagues &nbsp;|&nbsp; Analysis Only</div>
<div class="sha">🔒 V4 SHA256: {FROZEN_SHA256}</div>
<div class="frozen-note">
  ✅ <strong>Frozen V4 Model</strong> — Predictions are pre-kickoff, immutable, and unmodified.
  All 48 fixtures used the same model checkpoint. No retraining performed.
</div>

<div class="metrics-grid">
  <div class="metric-card {'good' if acc >= 0.55 else 'warn'}">
    <div class="value">{acc:.1%}</div>
    <div class="label">1X2 Accuracy<br>({correct_n}/{n})</div>
  </div>
  <div class="metric-card info">
    <div class="value">{exact_acc:.1%}</div>
    <div class="label">Exact Score<br>({metrics['B_exact_score']['correct']}/{n})</div>
  </div>
  <div class="metric-card info">
    <div class="value">{brier:.4f}</div>
    <div class="label">Mean Brier<br>(↓ better)</div>
  </div>
  <div class="metric-card info">
    <div class="value">{log_loss:.4f}</div>
    <div class="label">Mean Log Loss<br>(↓ better)</div>
  </div>
  <div class="metric-card info">
    <div class="value">{h_count}</div>
    <div class="label">Actual H wins</div>
  </div>
  <div class="metric-card info">
    <div class="value">{d_count}</div>
    <div class="label">Actual Draws</div>
  </div>
  <div class="metric-card info">
    <div class="value">{a_count}</div>
    <div class="label">Actual A wins</div>
  </div>
  <div class="metric-card info">
    <div class="value">{n}</div>
    <div class="label">Fixtures<br>Completed</div>
  </div>
</div>

<h2>📊 Full Fixture Audit</h2>
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>Date</th>
      <th>League</th>
      <th>Home</th>
      <th>Away</th>
      <th>P(H)</th>
      <th>P(D)</th>
      <th>P(A)</th>
      <th>Pred</th>
      <th>Pred Score</th>
      <th>Actual</th>
      <th>Actual Score</th>
      <th>✓/✗</th>
      <th>Exact</th>
      <th>Draw Risk</th>
      <th>Signal</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</div>

<p style="margin-top:30px; color:#484f58; font-size:0.8em;">
  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp;
  Analysis only — V4 model frozen, predictions immutable &nbsp;|&nbsp;
  Source: OddAlerts /fixtures/multiple
</p>
</body>
</html>"""
    return html


if __name__ == "__main__":
    joined, metrics, risk_analysis, signal_analysis, league_analysis, forensics, ft_records = run_audit()

    # 07 — Final Report
    report_md = generate_report(metrics, risk_analysis, signal_analysis, league_analysis, forensics, ft_records)
    report_path = os.path.join(OUT_DIR, "07_phase15_final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"    Saved: {report_path}")

    # 08 — Final Dashboard
    dashboard_html = generate_dashboard(joined, metrics, ft_records)
    dashboard_path = os.path.join(OUT_DIR, "08_phase15_final_dashboard.html")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    print(f"    Saved: {dashboard_path}")

    # README
    readme = f"""# Phase 15 Final Results Audit

**Cohort:** 2026-09-18 → 2026-09-21  
**Big-5 Leagues:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1  
**V4 SHA256:** `{FROZEN_SHA256}`  
**Status:** ANALYSIS ONLY

## Files

| File | Description |
|------|-------------|
| `01_phase15_final_results_ledger.jsonl` | Joined predictions + actuals for all 48 fixtures |
| `02_phase15_metrics.json` | Core metrics (1X2, Brier, LogLoss, calibration, etc.) |
| `03_phase15_risk_analysis.json` | Draw risk tier analysis |
| `04_phase15_signal_analysis.json` | 2-0 profile and strong home signal analysis |
| `05_phase15_league_analysis.json` | Per-league accuracy breakdown |
| `06_phase15_forensics.json` | Biggest misses, miss analysis |
| `07_phase15_final_report.md` | Full markdown audit report |
| `08_phase15_final_dashboard.html` | Interactive HTML dashboard |

## Critical Constraints

- V4 model was NOT modified, retrained, or re-parameterized.
- All predictions are pre-kickoff, pre-locked.
- Postponed fixtures are NOT counted as failures.
- Actual results sourced from OddAlerts /fixtures/multiple.
- Sportmonks data NOT included (excluded by design).
"""
    readme_path = os.path.join(OUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"    Saved: {readme_path}")

    # Print summary
    A = metrics["A_1x2_accuracy"]
    B = metrics["B_exact_score"]
    C = metrics["C_brier_score"]
    D = metrics["D_log_loss"]

    print("\n" + "=" * 70)
    print("PHASE 15 AUDIT SUMMARY")
    print("=" * 70)
    print(f"  Total fixtures:    {len(joined)}")
    print(f"  Completed (FT):    {len(ft_records)}")
    print(f"  1X2 Accuracy:      {A['correct']}/{A['total']} = {A['accuracy']:.1%}")
    print(f"  Exact Score:       {B['correct']}/{B['total']} = {B['accuracy']:.1%}")
    print(f"  Mean Brier:        {C['mean']:.4f}")
    print(f"  Mean Log Loss:     {D['mean']:.4f}")
    print("=" * 70)
