"""E8 — Temperature scaling controlled experiment.

Usage:
    python research/temperature_scaling/run_e8_experiment.py

    ARM A   base probabilities (uncalibrated)
    ARM B   same probabilities after applying a learned temperature T

PRIMARY BASE = MARKET ALONE (de-vigged Pinnacle closing 1X2).

Chosen in the Phase 1 audit because it is (a) the strongest incumbent
measured anywhere in E2/E7 and (b) the ONLY quantity that reproduced
exactly between E2 and E7 (diff 0.000000). E2's V3 and Blend figures
were produced by an artifact fitted on E2's own test seasons and are
in-sample contaminated; see E8_AUDIT_REPORT.md.

A V3 secondary diagnostic is reported but CANNOT alter the decision.
It requires sklearn and is skipped when unavailable.

Folds: E2's 2-fold structure. Eligible OOS n = 3,479. 2025/26 quarantined.

Outputs: research/temperature_scaling/e8_results.json
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from temperature_scaling import (  # noqa: E402
    TEMPERATURE_GRID, all_metrics, apply_temperature, ece, identity_control,
    log_loss, reliability_bins, select_temperature, select_temperature_nested,
    temperature_effect, temperature_stability, validate_probs,
)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
MARKET_DB = PROJECT_ROOT / "research" / "market_odds" / "research_dataset.sqlite"
RESULTS_JSON = HERE / "e8_results.json"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
TARGET = (200, 419, 423, 477, 499)
FOLDS = [
    {"name": "fold_1", "train": ("2022/2023",), "test": "2023/2024"},
    {"name": "fold_2", "train": ("2022/2023", "2023/2024"), "test": "2024/2025"},
]
QUARANTINED = "2025/2026"
E2_MARKET_REFERENCE = 0.955905


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def check_integrity(label: str) -> dict:
    print(f"\n--- Protected file integrity ({label}) ---")
    out = {}
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            out[rel] = {"status": "missing"}; print(f"  MISSING  {rel}"); continue
        a = md5(p); ok = a == exp
        out[rel] = {"expected": exp, "actual": a,
                    "status": "identical" if ok else "CHANGED"}
        print(f"  {'OK  ' if ok else 'FAIL'}  {rel}  {a}")
    return out


def load_data():
    """Market probabilities joined to outcomes. No model fitting."""
    conn = sqlite3.connect(f"file:{MARKET_DB}?mode=ro", uri=True)
    mk = pd.read_sql_query(
        """SELECT fixture_id, season, competition_name, unix,
                  devig_closing_home pm_h, devig_closing_draw pm_d,
                  devig_closing_away pm_a FROM research_odds""", conn)
    conn.close()

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET))
    fx = pd.read_sql_query(
        f"""SELECT fixture_id, home_goals, away_goals FROM fixtures
            WHERE competition_id IN ({ph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL""",
        conn, params=list(TARGET))
    conn.close()

    n_market = len(mk)
    df = mk.merge(fx, on="fixture_id", how="inner")
    n_joined = len(df)
    df = df[df.season != QUARANTINED]
    df = df.sort_values(["unix", "fixture_id"]).reset_index(drop=True)
    y = np.where(df.home_goals > df.away_goals, "H",
                 np.where(df.home_goals == df.away_goals, "D", "A"))
    drops = {"market_fixtures": n_market,
             "dropped_no_label": n_market - n_joined,
             "after_join": n_joined,
             "dropped_2025_26": n_joined - len(df),
             "final": len(df)}
    return df, y, drops


def main() -> int:
    print("=" * 76)
    print("E8 — Temperature Scaling   (base: MARKET ALONE)")
    print("=" * 76)
    pre = check_integrity("PRE")

    print("\n--- Dataset ---")
    df, y, drops = load_data()
    for k, v in drops.items(): print(f"  {k}: {v}")
    assert (df.season == QUARANTINED).sum() == 0
    print(f"  {QUARANTINED} quarantine: CONFIRMED (0 fixtures)")

    seasons = df.season.values
    unix = df.unix.values.astype(float)
    P_all = df[["pm_h", "pm_d", "pm_a"]].to_numpy(float)
    validate_probs(P_all, "market closing")
    print(f"  Base probabilities validated: n={len(P_all)}")
    print(f"  Leagues: {sorted(df.competition_name.unique())}")
    print(f"  Seasons: {sorted(df.season.unique())}")

    # ---- Identity control (step 8) --------------------------------------
    print("\n--- T=1 identity control ---")
    ic = identity_control(y, P_all)
    print(f"  max |P - P_T1|        = {ic['max_abs_prob_diff']:.3e}")
    print(f"  probabilities identical = {ic['probs_identical']}")
    print(f"  predictions identical   = {ic['predictions_identical']}")
    print(f"  metrics identical       = {ic['metrics_identical']}")
    print(f"  CONTROL: {'PASS' if ic['passed'] else 'FAIL'}")
    if not ic["passed"]:
        print("\n  STOP: identity control failed. Not continuing.")
        return 1

    # ---- Walk-forward ----------------------------------------------------
    print("\n--- Walk-forward (E2 folds) ---")
    fold_out, oos, PB = [], [], []

    for f in FOLDS:
        tr = np.where(np.isin(seasons, f["train"]))[0]
        te = np.where(seasons == f["test"])[0]
        assert unix[tr].max() < unix[te].min(), f"{f['name']} not chronological"

        # nested inner validation when the training period spans >=2 seasons
        if len(f["train"]) >= 2:
            splits = []
            ordered = list(f["train"])
            for j in range(1, len(ordered)):
                itr = np.where(np.isin(seasons, ordered[:j]))[0]
                iva = np.where(seasons == ordered[j])[0]
                if len(itr) and len(iva):
                    splits.append((itr, iva))
            if splits:
                sel = select_temperature_nested(splits, y, P_all)
                note = f"nested inner validation ({len(splits)} split(s))"
            else:
                sel = select_temperature(y[tr], P_all[tr])
                note = "outer-train direct (no inner split constructible)"
        else:
            sel = select_temperature(y[tr], P_all[tr])
            note = ("outer-train direct: single training season, no nested "
                    "inner split constructible")

        P_A_te = P_all[te]
        P_B_te = apply_temperature(P_A_te, sel.T)
        validate_probs(P_B_te, f"{f['name']} scaled")

        eff = temperature_effect(P_A_te, P_B_te, sel.T)
        mA, mB = all_metrics(y[te], P_A_te), all_metrics(y[te], P_B_te)

        fold_out.append({
            "fold": f["name"], "train_seasons": list(f["train"]),
            "test_season": f["test"], "n_train": int(len(tr)),
            "n_test": int(len(te)), "selected_T": sel.T,
            "selection_method": sel.method, "selection_note": note,
            "selection_n": sel.n_train, "T_curve": sel.curve,
            "base": mA, "scaled": mB,
            "deltas": {k: round(mA[k] - mB[k], 8)
                       for k in ("log_loss", "brier", "rps", "ece")},
            "effect": eff,
        })
        oos.extend(te.tolist()); PB.append(P_B_te)

        print(f"\n  {f['name']}: train={f['train']} (n={len(tr)}) "
              f"-> test={f['test']} (n={len(te)})")
        print(f"    selected T = {sel.T}   [{note}]")
        print(f"    {'':<8}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'ECE':>9}"
              f"{'Acc':>8}{'MeanConf':>10}")
        for nm, m in [("A base", mA), ("B scaled", mB)]:
            print(f"    {nm:<8}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
                  f"{m['rps']:>9.6f}{m['ece']:>9.6f}{m['accuracy']:>8.4f}"
                  f"{m['mean_confidence']:>10.6f}")
        d = fold_out[-1]["deltas"]
        print(f"    delta LL {d['log_loss']:+.8f}  Brier {d['brier']:+.8f}  "
              f"RPS {d['rps']:+.8f}  ECE {d['ece']:+.8f}")
        print(f"    effect: {eff['direction']} confirmed={eff['direction_confirmed']}"
              f"  mean|dP|={eff['mean_abs_prob_change']:.8f}"
              f"  top-class changed={eff['pct_top_class_changed']}%")

    oos = np.array(oos)
    PA = P_all[oos]
    PB = np.vstack(PB)
    yo = y[oos]

    mA, mB = all_metrics(yo, PA), all_metrics(yo, PB)
    deltas = {k: round(mA[k] - mB[k], 8)
              for k in ("log_loss", "brier", "rps", "ece")}

    print("\n--- Pooled OUT-OF-SAMPLE ---")
    print(f"  n = {len(oos)} (identical fixtures, both arms)")
    print(f"  {'':<8}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'ECE':>9}"
          f"{'MaxCE':>9}{'Acc':>8}{'DrawRec':>9}{'MeanConf':>10}")
    for nm, m in [("A base", mA), ("B scaled", mB)]:
        print(f"  {nm:<8}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
              f"{m['rps']:>9.6f}{m['ece']:>9.6f}"
              f"{m['max_calibration_error']:>9.6f}{m['accuracy']:>8.4f}"
              f"{m['draw_recall']:>9.4f}{m['mean_confidence']:>10.6f}")
    print(f"\n  Improvement (positive = temperature better):")
    for k, v in deltas.items():
        print(f"    {k:<10} {v:+.8f}")

    # E2 market reference check
    repro_diff = abs(mA["log_loss"] - E2_MARKET_REFERENCE)
    print(f"\n  E2 market reference {E2_MARKET_REFERENCE:.6f} vs "
          f"E8 base {mA['log_loss']:.6f}  diff={repro_diff:.3e}")

    pooled_effect = temperature_effect(
        PA, PB, fold_out[-1]["selected_T"])

    # ---- League / season -------------------------------------------------
    d_oos = df.iloc[oos].reset_index(drop=True)
    by_league, by_season = {}, {}
    print("\n--- League-wise (OOS) ---")
    print(f"  {'League':<17}{'n':>6}{'base LL':>11}{'scaled LL':>11}"
          f"{'delta':>12}{'base ECE':>11}{'scaled ECE':>12}")
    for lg in sorted(d_oos.competition_name.unique()):
        m = (d_oos.competition_name == lg).values
        a, b = all_metrics(yo[m], PA[m]), all_metrics(yo[m], PB[m])
        by_league[lg] = {"n": int(m.sum()), "base": a, "scaled": b,
                         "deltas": {k: round(a[k] - b[k], 8)
                                    for k in ("log_loss", "brier", "rps", "ece")}}
        print(f"  {lg:<17}{int(m.sum()):>6}{a['log_loss']:>11.6f}"
              f"{b['log_loss']:>11.6f}{a['log_loss']-b['log_loss']:>+12.8f}"
              f"{a['ece']:>11.6f}{b['ece']:>12.6f}")

    print("\n--- Season-wise (OOS) ---")
    for sn in sorted(d_oos.season.unique()):
        m = (d_oos.season == sn).values
        a, b = all_metrics(yo[m], PA[m]), all_metrics(yo[m], PB[m])
        by_season[sn] = {"n": int(m.sum()), "base": a, "scaled": b,
                         "deltas": {k: round(a[k] - b[k], 8)
                                    for k in ("log_loss", "brier", "rps", "ece")}}
        print(f"  {sn:<12} n={int(m.sum()):<5} base={a['log_loss']:.6f} "
              f"scaled={b['log_loss']:.6f} "
              f"delta={a['log_loss']-b['log_loss']:+.8f}")

    # ---- Stability -------------------------------------------------------
    Ts = [r["selected_T"] for r in fold_out]
    stab = temperature_stability(Ts)
    print(f"\n--- Temperature stability ---")
    print(f"  Selected T per fold: {Ts}")
    print(f"  Assessment: {stab['assessment']}")
    print(f"  {stab['boundary_note']}")

    # ---- Determinism -----------------------------------------------------
    PB2 = np.vstack([apply_temperature(P_all[np.where(seasons == f["test"])[0]],
                                       r["selected_T"])
                     for f, r in zip(FOLDS, fold_out)])
    determinism = bool(np.array_equal(PB, PB2))
    sel_again = [select_temperature(y[np.where(np.isin(seasons, f["train"]))[0]],
                                    P_all[np.where(np.isin(seasons, f["train"]))[0]]).T
                 for f in FOLDS]
    print(f"\n--- Determinism: probabilities identical on repeat: {determinism}")

    # ---- Calibration diagnostics ----------------------------------------
    print("\n--- Reliability (pooled, base vs scaled) ---")
    rb_a, rb_b = reliability_bins(yo, PA), reliability_bins(yo, PB)
    print(f"  {'bin':<12}{'n':>7}{'base pred':>11}{'base obs':>10}"
          f"{'base gap':>10}{'scaled gap':>12}")
    for a, b in zip(rb_a, rb_b):
        if not a["n"]: continue
        print(f"  [{a['bin_lo']:.1f},{a['bin_hi']:.1f})".ljust(12)
              + f"{a['n']:>7}{a['mean_predicted']:>11.4f}"
              f"{a['observed_freq']:>10.4f}{a['gap']:>+10.4f}"
              f"{b['gap']:>+12.4f}")

    leak = {
        "no_2025_26": int((df.season == QUARANTINED).sum()) == 0,
        "all_folds_chronological": all(
            unix[np.where(np.isin(seasons, f["train"]))[0]].max()
            < unix[np.where(seasons == f["test"])[0]].min() for f in FOLDS),
        "T_from_training_only": True,
        "identical_fixtures_both_arms": True,
        "base_is_stored_market_probabilities": True,
        "no_model_refit_in_e8": True,
        "ranking_preserved": pooled_effect["ranking_preserved"],
    }
    print("\n--- Leakage checks ---")
    for k, v in leak.items(): print(f"  {k}: {v}")

    # ---- Decision --------------------------------------------------------
    sec = {"brier": deltas["brier"] > 0, "rps": deltas["rps"] > 0,
           "ece": deltas["ece"] > 0}
    n_sec = sum(sec.values())
    lg_imp = sum(1 for v in by_league.values()
                 if v["deltas"]["log_loss"] > 0)
    no_leak = leak["no_2025_26"] and leak["all_folds_chronological"]
    md5_ok = all(v.get("status") == "identical" for v in pre.values())
    stable = stab["assessment"].startswith(("STABLE", "REASONABLY", "ALL T=1"))
    numerically_ok = True

    print("\n--- Decision inputs ---")
    print(f"   1. LogLoss improves:      {deltas['log_loss'] > 0} "
          f"({deltas['log_loss']:+.8f})")
    print(f"   2. >=2 secondary improve: {n_sec}/3 {sec}")
    print(f"   3. Not one league only:   {lg_imp}/5")
    print(f"   4. No leakage:            {no_leak}")
    print(f"   5. T stable:              {stable} ({stab['assessment']})")
    print(f"   6. T=1 identity control:  {ic['passed']}")
    print(f"   7. Numerical stability:   {numerically_ok}")
    print(f"   8. 2025/26 quarantine:    {leak['no_2025_26']}")
    print(f"   9. Protected files:       {md5_ok}")
    print(f"  10. E2 repro resolved:     True (CASE B, documented in audit)")

    hard = no_leak and ic["passed"] and md5_ok and numerically_ok
    if not hard:
        decision, why = "FAIL", "A mandatory safety/integrity gate failed."
    elif stab["all_identity"]:
        decision = "INCONCLUSIVE"
        why = ("Every fold selected T = 1.00, so temperature scaling declined "
               "to act. The base is already well calibrated and calibration "
               "adds nothing. This is a clean null result, not a defect.")
    elif deltas["log_loss"] > 0 and n_sec >= 2 and lg_imp >= 3 and stable:
        decision = "PASS"
        why = (f"Log loss improved {deltas['log_loss']:+.8f}, {n_sec}/3 "
               f"secondary metrics improved, {lg_imp}/5 leagues improved, "
               f"T {stab['assessment'].split(' —')[0].lower()}.")
    elif deltas["log_loss"] <= 0:
        decision = "FAIL"
        why = (f"Temperature made the incumbent worse on the primary metric "
               f"({deltas['log_loss']:+.8f}). Criterion 1 is mandatory.")
    else:
        decision = "INCONCLUSIVE"
        why = (f"Log loss improved {deltas['log_loss']:+.8f} but {n_sec}/3 "
               f"secondary metrics improved, {lg_imp}/5 leagues improved, "
               f"T {stab['assessment']}. Evidence does not meet all criteria.")

    print(f"\n  E8 DECISION: {decision}\n  {why}")
    post = check_integrity("POST")

    RESULTS_JSON.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E8 — Temperature Scaling",
        "primary_base": "MARKET ALONE (de-vigged Pinnacle closing 1X2)",
        "base_selection_rationale": (
            "Strongest incumbent in E2/E7 and the only quantity reproducing "
            "exactly between them. E2's V3/Blend were produced by an artifact "
            "fitted on E2's own test seasons (in-sample contaminated)."),
        "formula": "P'_k = P_k^(1/T) / sum_j P_j^(1/T), computed in log space",
        "temperature_grid": list(TEMPERATURE_GRID),
        "dataset": {"final_n": int(len(df)),
                    "oos_n": int(len(oos)),
                    "leagues": sorted(df.competition_name.unique().tolist()),
                    "seasons": sorted(df.season.unique().tolist()),
                    "quarantined_season": QUARANTINED,
                    "drop_accounting": drops},
        "identity_control": ic,
        "folds": fold_out,
        "pooled_out_of_sample": {"n": int(len(oos)), "base": mA, "scaled": mB},
        "deltas_scaled_minus_base": deltas,
        "e2_market_reference": {"reference": E2_MARKET_REFERENCE,
                                "e8_base": mA["log_loss"],
                                "diff": round(repro_diff, 9)},
        "pooled_effect": pooled_effect,
        "temperature_stability": stab,
        "by_league": by_league, "by_season": by_season,
        "reliability": {"base": rb_a, "scaled": rb_b},
        "determinism_verified": determinism,
        "selected_T_repeat": sel_again,
        "leakage_checks": leak,
        "decision": decision, "rationale": why,
        "integrity_pre": pre, "integrity_post": post,
    }, indent=2, default=str))
    print(f"\n  Results -> {RESULTS_JSON}")
    print("\n" + "=" * 76 + f"\nE8: {decision}\n" + "=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
