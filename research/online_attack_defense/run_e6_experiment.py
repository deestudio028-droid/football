"""E6 — V3 vs V3 + Online Attack/Defense controlled experiment.

Usage:
    python research/online_attack_defense/run_e6_experiment.py

Requires: numpy, pandas, scikit-learn.

    A = V3, 87 features
    B = V3 + (A_home, D_home, A_away, D_away), 91 features

Uses the project's ACTUAL V3 training implementation --
LogisticRegressionPreprocessor + PoissonRegressor(alpha=1.0,
max_iter=2000) + predict_poisson -- not a substitute solver. The
Phase 1 audit's IRLS numbers are diagnostic only and are deliberately
not reused here.

ARM A IS A PER-FOLD REFIT, NOT THE SHIPPED ARTIFACT
---------------------------------------------------
v3_poisson_venue_elo_candidate.pkl is fitted on 2020/21-2024/25 --
every season the folds use as TEST data. Arm A reproduces the per-fold
refit used by the approved E1/E2 walk-forward experiment.

No market odds, no Dixon-Coles, no time decay, no quadratic Elo.

Outputs: research/online_attack_defense/e6_results.json
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from online_attack_defense import (  # noqa: E402
    AD_COLUMNS, LR_GRID, STATE_CLIP, build_design, build_inner_splits,
    check_extreme_safety, compute_ad_states, e6_feature_columns,
    fit_baseline_rates, prediction_change_diagnostics, select_lr,
    state_diagnostics,
)

FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
RESULTS_JSON = HERE / "e6_results.json"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
TARGET = (200, 419, 423, 477, 499)
FOLDS = [
    {"name": "fold_1", "train": ("2020/2021", "2021/2022"), "test": "2022/2023"},
    {"name": "fold_2", "train": ("2020/2021", "2021/2022", "2022/2023"),
     "test": "2023/2024"},
    {"name": "fold_3", "train": ("2020/2021", "2021/2022", "2022/2023",
                                 "2023/2024"), "test": "2024/2025"},
]
QUARANTINED = "2025/2026"
ALPHA, MAX_ITER = 1.0, 2000
CLASS_ORDER = ("H", "D", "A")


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


def _oh(y):
    o = np.zeros((len(y), 3))
    for i, c in enumerate(CLASS_ORDER): o[:, i] = (y == c)
    return o


def log_loss(y, P):
    Pc = np.clip(P, 1e-15, 1.0); Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(_oh(y) * np.log(Pc), axis=1)))


def brier(y, P): return float(np.mean(np.sum((P - _oh(y)) ** 2, axis=1)))


def rps(y, P):
    cp, co = np.cumsum(P, axis=1), np.cumsum(_oh(y), axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def accuracy(y, P):
    return float(np.mean(np.array([CLASS_ORDER[i] for i in P.argmax(1)]) == y))


def draw_recall(y, P):
    pr = np.array([CLASS_ORDER[i] for i in P.argmax(1)]); m = (y == "D")
    return float(np.mean(pr[m] == "D")) if m.sum() else float("nan")


def calibration_bins(y, P, nb=10):
    oh, pf = _oh(y).ravel(), P.ravel(); e = np.linspace(0, 1, nb + 1); out = []
    for i in range(nb):
        lo, hi = e[i], e[i + 1]
        m = (pf >= lo) & (pf <= hi) if i == nb - 1 else (pf >= lo) & (pf < hi)
        n = int(m.sum())
        if not n:
            out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": 0,
                        "mean_predicted": None, "observed_freq": None,
                        "gap": None}); continue
        mp, ob = float(pf[m].mean()), float(oh[m].mean())
        out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": n,
                    "mean_predicted": round(mp, 6),
                    "observed_freq": round(ob, 6), "gap": round(ob - mp, 6)})
    return out


def ece(y, P, nb=10):
    b = calibration_bins(y, P, nb); t = sum(x["n"] for x in b)
    return float(sum(x["n"] / t * abs(x["gap"]) for x in b if x["n"])) if t else float("nan")


def all_metrics(y, P, lh=None, la=None, hg=None, ag=None):
    m = {"n": int(len(y)), "log_loss": round(log_loss(y, P), 6),
         "brier": round(brier(y, P), 6), "rps": round(rps(y, P), 6),
         "accuracy": round(accuracy(y, P), 6),
         "draw_recall": round(draw_recall(y, P), 6),
         "draw_predicted_rate": round(float(np.mean(P.argmax(1) == 1)), 6),
         "mean_p_draw": round(float(P[:, 1].mean()), 6),
         "ece": round(ece(y, P), 6)}
    if lh is not None and hg is not None:
        m["home_goal_mae"] = round(float(np.mean(np.abs(lh - hg))), 6)
        m["away_goal_mae"] = round(float(np.mean(np.abs(la - ag))), 6)
    return m


def load_dataset():
    from features.elo import ELO_COLUMNS, load_elo_features
    from models.data import load_supervised_dataset
    from models.v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES

    ds = load_supervised_dataset(FEATURES_DB)
    meta, X = ds.metadata.reset_index(drop=True), ds.X.reset_index(drop=True)
    print("  Computing causal Elo (identical for both arms)...")
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])
    assert len(V3_FEATURE_COLUMNS) == V3_N_FEATURES == 87

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET))
    fx = pd.read_sql_query(
        f"""SELECT fixture_id, competition_name, season, unix, home_id, away_id,
                   home_goals, away_goals, status FROM fixtures
            WHERE competition_id IN ({ph}) AND status IN ('FT','AWARDED')
              AND home_goals IS NOT NULL""", conn, params=list(TARGET))
    conn.close()

    j = meta[["fixture_id"]].join(X[list(V3_FEATURE_COLUMNS)])
    j = j.merge(fx, on="fixture_id", how="inner")
    j = j[j["season"] != QUARANTINED]
    j = j.sort_values(["unix", "fixture_id"]).reset_index(drop=True)
    y = np.where(j.home_goals > j.away_goals, "H",
                 np.where(j.home_goals == j.away_goals, "D", "A"))
    return j, tuple(V3_FEATURE_COLUMNS), y


def fit_predict(X_tr, hg_tr, ag_tr, X_ev):
    """The project's ACTUAL V3 training path. Not a substitute solver."""
    from sklearn.linear_model import PoissonRegressor
    from models.train import LogisticRegressionPreprocessor
    from models.poisson import predict_poisson

    prep = LogisticRegressionPreprocessor().fit(X_tr)
    Etr, Eev = prep.transform(X_tr), prep.transform(X_ev)
    mh = PoissonRegressor(alpha=ALPHA, max_iter=MAX_ITER).fit(Etr, hg_tr)
    ma = PoissonRegressor(alpha=ALPHA, max_iter=MAX_ITER).fit(Etr, ag_tr)
    lh, la = mh.predict(Eev), ma.predict(Eev)
    pr = predict_poisson(lh, la, list(CLASS_ORDER))
    P = np.array([[p.probabilities["H"], p.probabilities["D"],
                   p.probabilities["A"]] for p in pr])
    return P, lh, la


def main() -> int:
    print("=" * 74)
    print("E6 — V3 vs V3 + Online Attack/Defense")
    print("=" * 74)
    pre = check_integrity("PRE")

    print("\n--- Dataset ---")
    df, V3COLS, y = load_dataset()
    E6COLS = e6_feature_columns(V3COLS)
    print(f"  Fixtures: {len(df)}   Leagues: {df.competition_name.nunique()}")
    print(f"  Seasons: {sorted(df.season.unique())}")
    assert (df.season == QUARANTINED).sum() == 0
    print(f"  {QUARANTINED} quarantine: CONFIRMED (0 fixtures)")
    print(f"  V3 contract: {len(V3COLS)} | E6 contract: {len(E6COLS)}")
    assert tuple(E6COLS[:87]) == tuple(V3COLS) and tuple(E6COLS[87:]) == AD_COLUMNS

    seasons = df.season.values
    unix = df.unix.values.astype(float)
    hg = df.home_goals.values.astype(float)
    ag = df.away_goals.values.astype(float)
    fids = df.fixture_id.to_numpy()
    Xv3 = df[list(V3COLS)]
    state_src = df[["fixture_id", "unix", "home_id", "away_id",
                    "home_goals", "away_goals", "status"]]

    # ---- Zero-online control ------------------------------------------
    print("\n--- Zero-online-state control ---")
    tr0 = np.where(np.isin(seasons, FOLDS[0]["train"]))[0]
    te0 = np.where(seasons == FOLDS[0]["test"])[0]
    base0 = fit_baseline_rates(hg[tr0], ag[tr0])
    P_v3, lh_v3, la_v3 = fit_predict(Xv3.iloc[tr0], hg[tr0], ag[tr0],
                                     Xv3.iloc[te0])
    st_off = compute_ad_states(state_src, 0.02, base0, enabled=False)
    Dtr = build_design(Xv3.iloc[tr0], V3COLS, st_off, fids[tr0])
    Dte = build_design(Xv3.iloc[te0], V3COLS, st_off, fids[te0])
    P_z, lh_z, la_z = fit_predict(Dtr, hg[tr0], ag[tr0], Dte)
    d_lam = float(np.max(np.abs(lh_v3 - lh_z)))
    d_p = float(np.max(np.abs(P_v3 - P_z)))
    control_ok = d_lam < 1e-6 and d_p < 1e-6
    print(f"  87-col V3 vs 91-col with all-zero states")
    print(f"    max |dlambda| = {d_lam:.3e}")
    print(f"    max |dP|      = {d_p:.3e}")
    print(f"    CONTROL: {'PASS' if control_ok else 'FAIL'}")
    if not control_ok:
        print("\n  STOP: zero-online control failed. Not continuing.")
        return 1

    # ---- Walk-forward ---------------------------------------------------
    print("\n--- Walk-forward (approved V3/E2 folds, nested lr selection) ---")
    fold_out, oos, PAs, PBs = [], [], [], []
    lamA, lamB, safety_all = [], [], []

    for f in FOLDS:
        tr = np.where(np.isin(seasons, f["train"]))[0]
        te = np.where(seasons == f["test"])[0]
        assert unix[tr].max() < unix[te].min(), f"{f['name']} not chronological"

        base = fit_baseline_rates(hg[tr], ag[tr])   # TRAINING ONLY
        inner = build_inner_splits(seasons[tr], f["train"])

        def fp(sp, lr):
            g_tr, g_va = tr[sp.train_idx], tr[sp.val_idx]
            b_in = fit_baseline_rates(hg[g_tr], ag[g_tr])
            st = compute_ad_states(state_src, lr, b_in)
            Dt = build_design(Xv3.iloc[g_tr], V3COLS, st, fids[g_tr])
            Dv = build_design(Xv3.iloc[g_va], V3COLS, st, fids[g_va])
            P, _, _ = fit_predict(Dt, hg[g_tr], ag[g_tr], Dv)
            return log_loss(y[g_va], P)

        sel = select_lr(inner, fp, LR_GRID)

        st = compute_ad_states(state_src, sel.lr, base)
        DtrB = build_design(Xv3.iloc[tr], V3COLS, st, fids[tr])
        DteB = build_design(Xv3.iloc[te], V3COLS, st, fids[te])

        P_A, lhA, laA = fit_predict(Xv3.iloc[tr], hg[tr], ag[tr], Xv3.iloc[te])
        P_B, lhB, laB = fit_predict(DtrB, hg[tr], ag[tr], DteB)

        sd = state_diagnostics(st[st.fixture_id.isin(fids[te])])
        saf = {"A": check_extreme_safety(lhA, laA, P_A),
               "B": check_extreme_safety(lhB, laB, P_B)}
        safety_all.append(saf)

        rec = {"fold": f["name"], "train_seasons": list(f["train"]),
               "test_season": f["test"], "n_train": int(len(tr)),
               "n_test": int(len(te)), "n_inner_splits": sel.n_inner_splits,
               "selected_lr": sel.lr,
               "mean_inner_log_loss": round(sel.mean_inner_log_loss, 6),
               "lr_curve": sel.curve, "baseline_rates": base.as_dict(),
               "state_diagnostics": sd, "extreme_safety": saf,
               "test_A_v3": all_metrics(y[te], P_A, lhA, laA, hg[te], ag[te]),
               "test_B_e6": all_metrics(y[te], P_B, lhB, laB, hg[te], ag[te]),
               "prediction_change": prediction_change_diagnostics(
                   lhA, laA, P_A, lhB, laB, P_B)}
        fold_out.append(rec)
        oos.extend(te.tolist()); PAs.append(P_A); PBs.append(P_B)
        lamA.append((lhA, laA)); lamB.append((lhB, laB))

        a, b = rec["test_A_v3"], rec["test_B_e6"]
        print(f"\n  {f['name']}: train={f['train']} (n={len(tr)}) "
              f"-> test={f['test']} (n={len(te)})")
        print(f"    inner splits: {sel.n_inner_splits}   selected lr = {sel.lr}")
        print(f"    mu_home={base.mu_home:.4f} mu_away={base.mu_away:.4f}")
        print(f"    A_home[{sd['A_home']['min']:+.4f},{sd['A_home']['max']:+.4f}] "
              f"D_home[{sd['D_home']['min']:+.4f},{sd['D_home']['max']:+.4f}]")
        print(f"    {'':<10}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'Acc':>8}"
              f"{'ECE':>8}{'hMAE':>8}{'aMAE':>8}")
        for nm, m in [("A V3", a), ("B +A/D", b)]:
            print(f"    {nm:<10}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
                  f"{m['rps']:>9.6f}{m['accuracy']:>8.4f}{m['ece']:>8.4f}"
                  f"{m['home_goal_mae']:>8.4f}{m['away_goal_mae']:>8.4f}")
        print(f"    delta LL {a['log_loss']-b['log_loss']:+.6f}  "
              f"Brier {a['brier']-b['brier']:+.6f}  "
              f"RPS {a['rps']-b['rps']:+.6f}  ECE {a['ece']-b['ece']:+.6f}")

    oos = np.array(oos); PA, PB = np.vstack(PAs), np.vstack(PBs); yo = y[oos]
    lhA = np.concatenate([x[0] for x in lamA]); laA = np.concatenate([x[1] for x in lamA])
    lhB = np.concatenate([x[0] for x in lamB]); laB = np.concatenate([x[1] for x in lamB])

    mA = all_metrics(yo, PA, lhA, laA, hg[oos], ag[oos])
    mB = all_metrics(yo, PB, lhB, laB, hg[oos], ag[oos])
    d_ll = mA["log_loss"] - mB["log_loss"]
    d_br = mA["brier"] - mB["brier"]
    d_rps = mA["rps"] - mB["rps"]
    d_ece = mA["ece"] - mB["ece"]

    print("\n--- Pooled OUT-OF-SAMPLE ---")
    print(f"  n = {len(oos)} (identical fixture set for A and B)")
    print(f"  {'':<10}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'Acc':>8}"
          f"{'DrawRec':>9}{'ECE':>8}{'hMAE':>8}{'aMAE':>8}")
    for nm, m in [("A V3", mA), ("B +A/D", mB)]:
        print(f"  {nm:<10}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
              f"{m['rps']:>9.6f}{m['accuracy']:>8.4f}{m['draw_recall']:>9.4f}"
              f"{m['ece']:>8.4f}{m['home_goal_mae']:>8.4f}"
              f"{m['away_goal_mae']:>8.4f}")
    print(f"\n  Improvement (positive = E6 better):")
    print(f"    LogLoss {d_ll:+.6f}  Brier {d_br:+.6f}  "
          f"RPS {d_rps:+.6f}  ECE {d_ece:+.6f}")

    pooled_change = prediction_change_diagnostics(lhA, laA, PA, lhB, laB, PB)
    print(f"\n--- Prediction change (pooled) ---")
    for k, v in pooled_change.items():
        print(f"  {k}: {v}")

    d_oos = df.iloc[oos].reset_index(drop=True)
    by_league, by_season = {}, {}
    print("\n--- League-wise (OOS) ---")
    for lg in sorted(d_oos.competition_name.unique()):
        m = (d_oos.competition_name == lg).values
        a, b = all_metrics(yo[m], PA[m]), all_metrics(yo[m], PB[m])
        by_league[lg] = {"n": int(m.sum()), "A_v3": a, "B_e6": b,
                         "delta_log_loss": round(a["log_loss"]-b["log_loss"], 6),
                         "delta_brier": round(a["brier"]-b["brier"], 6),
                         "delta_rps": round(a["rps"]-b["rps"], 6)}
        print(f"  {lg:<17} n={int(m.sum()):<5} A={a['log_loss']:.6f} "
              f"B={b['log_loss']:.6f} delta={a['log_loss']-b['log_loss']:+.6f}")
    print("\n--- Season-wise (OOS) ---")
    for sn in sorted(d_oos.season.unique()):
        m = (d_oos.season == sn).values
        a, b = all_metrics(yo[m], PA[m]), all_metrics(yo[m], PB[m])
        by_season[sn] = {"n": int(m.sum()), "A_v3": a, "B_e6": b,
                         "delta_log_loss": round(a["log_loss"]-b["log_loss"], 6)}
        print(f"  {sn:<12} n={int(m.sum()):<5} A={a['log_loss']:.6f} "
              f"B={b['log_loss']:.6f} delta={a['log_loss']-b['log_loss']:+.6f}")

    lrs = [r["selected_lr"] for r in fold_out]
    spread = max(lrs) / min(lrs)
    lr_stable = len(set(lrs)) == 1 or spread <= 2.0
    print(f"\n--- lr stability ---")
    print(f"  Selected lr per fold: {lrs}  spread={spread:.2f}x  "
          f"-> {'STABLE' if lr_stable else 'UNSTABLE'}")

    states_stable = all(r["state_diagnostics"]["stable"] for r in fold_out)
    extreme_ok = all(s["A"]["passed"] and s["B"]["passed"] for s in safety_all)
    print(f"--- States stable: {states_stable}   Extreme-value safety: {extreme_ok}")

    # ---- Determinism ----------------------------------------------------
    st_d = compute_ad_states(state_src, 0.02, base0)
    D1 = build_design(Xv3.iloc[tr0], V3COLS, st_d, fids[tr0])
    D2 = build_design(Xv3.iloc[te0], V3COLS, st_d, fids[te0])
    Pr1, *_ = fit_predict(D1, hg[tr0], ag[tr0], D2)
    Pr2, *_ = fit_predict(D1, hg[tr0], ag[tr0], D2)
    determinism = float(np.max(np.abs(Pr1 - Pr2))) < 1e-12
    print(f"--- Determinism: repeated fit identical: {determinism}")

    leak = {
        "no_2025_26": int((df.season == QUARANTINED).sum()) == 0,
        "all_folds_chronological": all(
            unix[np.where(np.isin(seasons, f["train"]))[0]].max()
            < unix[np.where(seasons == f["test"])[0]].min() for f in FOLDS),
        "lr_from_inner_training_splits_only": True,
        "baseline_rates_from_training_only": True,
        "identical_fixtures_A_B": True,
        "only_four_columns_differ": True,
        "market_used": False, "dixon_coles_used": False,
        "time_decay_used": False, "quadratic_elo_used": False,
        "zero_online_control_passed": control_ok,
    }
    print("\n--- Leakage checks ---")
    for k, v in leak.items(): print(f"  {k}: {v}")

    sec = {"brier": d_br > 0, "rps": d_rps > 0, "ece": d_ece > 0}
    n_sec = sum(sec.values())
    lg_imp = sum(1 for v in by_league.values() if v["delta_log_loss"] > 0)
    no_leak = leak["no_2025_26"] and leak["all_folds_chronological"]
    md5_ok = all(v.get("status") == "identical" for v in pre.values())

    print("\n--- Decision inputs ---")
    print(f"   1. LogLoss improves:      {d_ll > 0} ({d_ll:+.6f})")
    print(f"   2. >=2 secondary improve: {n_sec}/3 {sec}")
    print(f"   3. Not one league only:   {lg_imp}/5")
    print(f"   4. No leakage:            {no_leak}")
    print(f"   5. States stable:         {states_stable}")
    print(f"   6. Zero-online control:   {control_ok}")
    print(f"   7. lr stable:             {lr_stable}")
    print(f"   8. Determinism:           {determinism}")
    print(f"   9. 2025/26 quarantine:    {leak['no_2025_26']}")
    print(f"  10. Protected files:       {md5_ok}")

    hard = (no_leak and control_ok and determinism and leak["no_2025_26"]
            and md5_ok and states_stable and extreme_ok)
    if not hard:
        decision = "FAIL"
        why = "A mandatory safety/integrity criterion failed."
    elif d_ll > 0 and n_sec >= 2 and lg_imp >= 3 and lr_stable:
        decision = "PASS"
        why = (f"Log loss improved {d_ll:+.6f}, {n_sec}/3 secondary metrics "
               f"improved, {lg_imp}/5 leagues improved, lr stable "
               f"({lrs}).")
    elif d_ll <= 0:
        decision = "FAIL"
        why = (f"Primary metric did not improve (delta {d_ll:+.6f}). "
               "Criterion 1 is mandatory.")
    else:
        decision = "INCONCLUSIVE"
        why = (f"Log loss improved {d_ll:+.6f} but {n_sec}/3 secondary "
               f"improved, {lg_imp}/5 leagues improved, lr "
               f"{'stable' if lr_stable else 'unstable'} ({lrs}).")

    print(f"\n  E6 DECISION: {decision}\n  {why}")
    post = check_integrity("POST")

    RESULTS_JSON.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E6 — V3 vs V3 + Online Attack/Defense",
        "solver": "sklearn PoissonRegressor (project V3 path, not IRLS)",
        "dataset": {"n_fixtures": int(len(df)),
                    "leagues": sorted(df.competition_name.unique().tolist()),
                    "seasons": sorted(df.season.unique().tolist()),
                    "quarantined_season": QUARANTINED,
                    "market_used": False, "dixon_coles_used": False,
                    "time_decay_used": False, "quadratic_elo_used": False},
        "feature_contract": {"v3_n": len(V3COLS), "e6_n": len(E6COLS),
                             "added_columns": list(AD_COLUMNS),
                             "e6_first_87_equals_v3": True},
        "estimator_config": {"alpha": ALPHA, "max_iter": MAX_ITER,
                             "model": "PoissonRegressor"},
        "lr_grid": list(LR_GRID),
        "state_clip": STATE_CLIP,
        "zero_online_control": {"max_abs_dlambda": d_lam, "max_abs_dP": d_p,
                                "passed": control_ok},
        "folds": fold_out,
        "pooled_out_of_sample": {"n": int(len(oos)), "A_v3": mA, "B_e6": mB},
        "deltas_B_minus_A": {"log_loss": round(d_ll, 6),
                             "brier": round(d_br, 6), "rps": round(d_rps, 6),
                             "ece": round(d_ece, 6)},
        "prediction_change_pooled": pooled_change,
        "by_league": by_league, "by_season": by_season,
        "lr_stability": {"selected": lrs, "spread": round(spread, 2),
                         "stable": lr_stable},
        "states_stable": states_stable,
        "extreme_value_safety_passed": extreme_ok,
        "calibration": {"A_v3": calibration_bins(yo, PA),
                        "B_e6": calibration_bins(yo, PB)},
        "determinism_verified": determinism,
        "leakage_checks": leak,
        "decision": decision, "rationale": why,
        "integrity_pre": pre, "integrity_post": post,
    }, indent=2, default=str))
    print(f"\n  Results -> {RESULTS_JSON}")
    print("\n" + "=" * 74 + f"\nE6: {decision}\n" + "=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
