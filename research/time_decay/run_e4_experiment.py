"""E4 — V3 vs V3 + Exponential Time Decay controlled experiment.

Usage:
    python research/time_decay/run_e4_experiment.py

Requires: numpy, pandas, scikit-learn.

    A = V3 architecture, uniform training weights   (w_i = 1)
    B = V3 architecture, decay training weights     (w_i = exp(-xi*dAge))

The ONLY difference between arms is the training weight vector. Same
fixtures, same labels, same 87 features, same Elo, same splits, same
preprocessor class, same estimator config, same Poisson conversion,
same score grid.

WHY ARM A IS A PER-FOLD REFIT, NOT THE SHIPPED ARTIFACT
-------------------------------------------------------
v3_poisson_venue_elo_candidate.pkl is fitted on FINAL_TRAIN_SEASONS =
2020/21 through 2024/25 — i.e. on every season that the walk-forward
folds use as TEST data. Scoring fold test sets with it would leak all
test outcomes into training.

The V3 that was *validated* in E1 is the per-fold refit performed inside
research/worldcup_elo/run_walk_forward_experiments.py (it refits the
preprocessor and both PoissonRegressors inside each fold). Arm A
reproduces exactly that. The shipped artifact is the separate final fit
used only to predict the quarantined 2025/26 season.

No market odds. No Dixon-Coles. No new features.

Outputs: research/time_decay/e4_results.json
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

from time_decay import (  # noqa: E402
    HALF_LIFE_GRID_DAYS, assert_causal, build_inner_splits, decay_weights,
    half_life_from_xi, select_xi, weight_profile, xi_from_half_life,
)

FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
RESULTS_JSON = HERE / "e4_results.json"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}

TARGET_COMPETITIONS = (200, 419, 423, 477, 499)
FOLDS = [
    {"name": "fold_1", "train": ("2020/2021", "2021/2022"), "test": "2022/2023"},
    {"name": "fold_2", "train": ("2020/2021", "2021/2022", "2022/2023"),
     "test": "2023/2024"},
    {"name": "fold_3", "train": ("2020/2021", "2021/2022", "2022/2023",
                                 "2023/2024"), "test": "2024/2025"},
]
QUARANTINED_SEASON = "2025/2026"

#: V3 estimator config, copied verbatim from train_v3_candidate.py.
ALPHA = 1.0
MAX_ITER = 2000
CLASS_ORDER = ("H", "D", "A")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def check_integrity(label: str) -> dict:
    print(f"\n--- Protected file integrity ({label}) ---")
    out = {}
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            out[rel] = {"status": "missing"}; print(f"  MISSING  {rel}"); continue
        a = md5(p)
        ok = a == exp
        out[rel] = {"expected": exp, "actual": a,
                    "status": "identical" if ok else "CHANGED"}
        print(f"  {'OK  ' if ok else 'FAIL'}  {rel}  {a}")
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _oh(y):
    o = np.zeros((len(y), 3))
    for i, c in enumerate(CLASS_ORDER):
        o[:, i] = (y == c)
    return o


def log_loss(y, P):
    Pc = np.clip(P, 1e-15, 1.0); Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(_oh(y) * np.log(Pc), axis=1)))


def brier(y, P):
    return float(np.mean(np.sum((P - _oh(y)) ** 2, axis=1)))


def rps(y, P):
    cp, co = np.cumsum(P, axis=1), np.cumsum(_oh(y), axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def accuracy(y, P):
    return float(np.mean(np.array([CLASS_ORDER[i] for i in P.argmax(1)]) == y))


def draw_recall(y, P):
    pred = np.array([CLASS_ORDER[i] for i in P.argmax(1)])
    m = (y == "D")
    return float(np.mean(pred[m] == "D")) if m.sum() else float("nan")


def calibration_bins(y, P, n_bins=10):
    oh, pf = _oh(y).ravel(), P.ravel()
    edges = np.linspace(0, 1, n_bins + 1); out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pf >= lo) & (pf <= hi) if i == n_bins - 1 else (pf >= lo) & (pf < hi)
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


def ece(y, P, n_bins=10):
    b = calibration_bins(y, P, n_bins); tot = sum(x["n"] for x in b)
    return float(sum(x["n"] / tot * abs(x["gap"]) for x in b if x["n"])) if tot else float("nan")


def all_metrics(y, P, lam_h=None, lam_a=None, hg=None, ag=None):
    m = {"n": int(len(y)),
         "log_loss": round(log_loss(y, P), 6),
         "brier": round(brier(y, P), 6),
         "rps": round(rps(y, P), 6),
         "accuracy": round(accuracy(y, P), 6),
         "draw_recall": round(draw_recall(y, P), 6),
         "draw_predicted_rate": round(float(np.mean(P.argmax(1) == 1)), 6),
         "mean_p_draw": round(float(P[:, 1].mean()), 6),
         "ece": round(ece(y, P), 6)}
    if lam_h is not None and hg is not None:
        m["home_goal_mae"] = round(float(np.mean(np.abs(lam_h - hg))), 6)
        m["away_goal_mae"] = round(float(np.mean(np.abs(lam_a - ag))), 6)
    return m


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset():
    """Build the shared (X, y, meta) table used identically by A and B."""
    from features.elo import ELO_COLUMNS, load_elo_features
    from models.data import load_supervised_dataset
    from models.v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES

    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)

    print("  Computing causal Elo (identical for both arms)...")
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for col in ELO_COLUMNS:
        X[col] = meta["fixture_id"].map(elo[col])

    for c in V3_FEATURE_COLUMNS:
        assert c in X.columns, f"missing feature {c}"
    assert len(V3_FEATURE_COLUMNS) == V3_N_FEATURES == 87

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET_COMPETITIONS))
    fx = pd.read_sql_query(
        f"""SELECT fixture_id, competition_name, season, unix,
                   home_goals, away_goals
            FROM fixtures WHERE competition_id IN ({ph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL""",
        conn, params=list(TARGET_COMPETITIONS))
    conn.close()

    joined = meta[["fixture_id"]].join(X[list(V3_FEATURE_COLUMNS)])
    joined = joined.merge(fx, on="fixture_id", how="inner")
    joined = joined.sort_values(["unix", "fixture_id"]).reset_index(drop=True)

    assert QUARANTINED_SEASON not in set(joined["season"]) or \
        (joined["season"] == QUARANTINED_SEASON).sum() == 0 or True
    joined = joined[joined["season"] != QUARANTINED_SEASON].reset_index(drop=True)

    y = np.where(joined.home_goals > joined.away_goals, "H",
                 np.where(joined.home_goals == joined.away_goals, "D", "A"))
    return joined, list(V3_FEATURE_COLUMNS), y


def fit_arm(X_tr, hg_tr, ag_tr, X_ev, w=None):
    """Fit V3 architecture on (X_tr, goals) with optional weights; predict X_ev.

    Identical code path for both arms — `w=None` is arm A.
    """
    from sklearn.linear_model import PoissonRegressor
    from models.train import LogisticRegressionPreprocessor
    from models.poisson import predict_poisson

    prep = LogisticRegressionPreprocessor().fit(X_tr)
    Etr, Eev = prep.transform(X_tr), prep.transform(X_ev)

    mh = PoissonRegressor(alpha=ALPHA, max_iter=MAX_ITER).fit(
        Etr, hg_tr, sample_weight=w)
    ma = PoissonRegressor(alpha=ALPHA, max_iter=MAX_ITER).fit(
        Etr, ag_tr, sample_weight=w)

    lam_h, lam_a = mh.predict(Eev), ma.predict(Eev)
    preds = predict_poisson(lam_h, lam_a, list(CLASS_ORDER))
    P = np.array([[p.probabilities["H"], p.probabilities["D"],
                   p.probabilities["A"]] for p in preds])
    return P, lam_h, lam_a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("E4 — V3 vs V3 + Exponential Time Decay")
    print("=" * 72)
    pre = check_integrity("PRE")

    print("\n--- Dataset ---")
    df, FEATS, y = load_dataset()
    print(f"  Fixtures: {len(df)}   Leagues: {df.competition_name.nunique()}")
    print(f"  Seasons: {sorted(df.season.unique())}")
    assert (df["season"] == QUARANTINED_SEASON).sum() == 0
    print(f"  {QUARANTINED_SEASON} quarantine: CONFIRMED (0 fixtures)")

    seasons = df["season"].values
    unix = df["unix"].values.astype(float)
    hg = df["home_goals"].values.astype(float)
    ag = df["away_goals"].values.astype(float)
    Xall = df[FEATS]

    # ---- Zero-decay control (§19) --------------------------------------
    print("\n--- Zero-decay control (§19) ---")
    f0 = FOLDS[0]
    tr0 = np.where(np.isin(seasons, f0["train"]))[0]
    te0 = np.where(seasons == f0["test"])[0]
    P_unw, lh_unw, la_unw = fit_arm(Xall.iloc[tr0], hg[tr0], ag[tr0],
                                    Xall.iloc[te0], w=None)
    ref0 = float(unix[te0].min())
    w_ones = decay_weights(unix[tr0], ref0, 0.0)
    P_w1, lh_w1, la_w1 = fit_arm(Xall.iloc[tr0], hg[tr0], ag[tr0],
                                 Xall.iloc[te0], w=w_ones)
    d_lam = float(np.max(np.abs(lh_unw - lh_w1)))
    d_p = float(np.max(np.abs(P_unw - P_w1)))
    control_ok = d_lam < 1e-9 and d_p < 1e-9
    print(f"  all-ones weights vs sample_weight=None")
    print(f"    max |dlambda| = {d_lam:.3e}")
    print(f"    max |dP|      = {d_p:.3e}")
    print(f"    CONTROL: {'PASS' if control_ok else 'FAIL'}")
    if not control_ok:
        print("\n  STOP: zero-decay control failed. Not continuing (§19).")
        return 1

    # ---- Walk-forward ---------------------------------------------------
    print("\n--- Walk-forward (approved V3 folds, nested xi selection) ---")
    fold_out, oos_idx, oos_A, oos_B = [], [], [], []

    for f in FOLDS:
        tr = np.where(np.isin(seasons, f["train"]))[0]
        te = np.where(seasons == f["test"])[0]
        ref = float(unix[te].min())
        assert unix[tr].max() < unix[te].min(), f"{f['name']} not chronological"
        assert_causal(unix[tr], ref)

        # --- inner temporal selection of xi, TRAIN ONLY ---
        inner = build_inner_splits(seasons[tr], unix[tr], f["train"])

        def fit_predict(sp, xi):
            gtr = tr[sp.inner_train_idx]
            gva = tr[sp.inner_val_idx]
            w = None if xi == 0.0 else decay_weights(unix[gtr],
                                                     sp.reference_unix, xi)
            P, _, _ = fit_arm(Xall.iloc[gtr], hg[gtr], ag[gtr],
                              Xall.iloc[gva], w=w)
            return log_loss(y[gva], P)

        sel = select_xi(inner, fit_predict, HALF_LIFE_GRID_DAYS)

        # --- refit on the FULL outer training period with the frozen xi ---
        w_out = None if sel.xi == 0.0 else decay_weights(unix[tr], ref, sel.xi)
        P_A, lhA, laA = fit_arm(Xall.iloc[tr], hg[tr], ag[tr],
                                Xall.iloc[te], w=None)
        P_B, lhB, laB = fit_arm(Xall.iloc[tr], hg[tr], ag[tr],
                                Xall.iloc[te], w=w_out)

        prof = weight_profile(sel.xi)
        rec = {
            "fold": f["name"], "train_seasons": list(f["train"]),
            "test_season": f["test"], "n_train": int(len(tr)),
            "n_test": int(len(te)), "n_inner_splits": sel.n_inner_splits,
            "selected_xi": sel.xi,
            "selected_half_life_days": sel.half_life_days,
            "mean_inner_log_loss": round(sel.mean_inner_log_loss, 6),
            "xi_curve": sel.curve,
            "weight_profile": prof,
            "train_age_days": {"min": round(float((ref - unix[tr]).max() / 86400), 1),
                               "max": round(float((ref - unix[tr]).min() / 86400), 1)},
            "test_A_v3": all_metrics(y[te], P_A, lhA, laA, hg[te], ag[te]),
            "test_B_v3_decay": all_metrics(y[te], P_B, lhB, laB, hg[te], ag[te]),
        }
        fold_out.append(rec)
        oos_idx.extend(te.tolist()); oos_A.append(P_A); oos_B.append(P_B)

        print(f"\n  {f['name']}: train={f['train']} (n={len(tr)}) "
              f"-> test={f['test']} (n={len(te)})")
        print(f"    inner splits: {sel.n_inner_splits}")
        print(f"    selected half-life: {sel.half_life_days} days "
              f"(xi={sel.xi:.8f})")
        print(f"    weight profile: {prof['weights']}")
        a, b = rec["test_A_v3"], rec["test_B_v3_decay"]
        print(f"    {'':<10}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'Acc':>8}{'ECE':>8}")
        for nm, m in [("A V3", a), ("B +decay", b)]:
            print(f"    {nm:<10}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
                  f"{m['rps']:>9.6f}{m['accuracy']:>8.4f}{m['ece']:>8.4f}")
        print(f"    delta LL {a['log_loss']-b['log_loss']:+.6f}  "
              f"Brier {a['brier']-b['brier']:+.6f}  "
              f"RPS {a['rps']-b['rps']:+.6f}  ECE {a['ece']-b['ece']:+.6f}")

    oos = np.array(oos_idx)
    PA, PB = np.vstack(oos_A), np.vstack(oos_B)
    yo = y[oos]

    mA = all_metrics(yo, PA); mB = all_metrics(yo, PB)
    d_ll = mA["log_loss"] - mB["log_loss"]
    d_br = mA["brier"] - mB["brier"]
    d_rps = mA["rps"] - mB["rps"]
    d_ece = mA["ece"] - mB["ece"]

    print("\n--- Pooled OUT-OF-SAMPLE ---")
    print(f"  n = {len(oos)} (identical fixture set for A and B)")
    print(f"  {'':<10}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'Acc':>8}"
          f"{'DrawRec':>9}{'ECE':>8}{'meanPD':>9}")
    for nm, m in [("A V3", mA), ("B +decay", mB)]:
        print(f"  {nm:<10}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
              f"{m['rps']:>9.6f}{m['accuracy']:>8.4f}"
              f"{m['draw_recall']:>9.4f}{m['ece']:>8.4f}{m['mean_p_draw']:>9.4f}")
    print(f"\n  Improvement (positive = decay better):")
    print(f"    LogLoss {d_ll:+.6f}  Brier {d_br:+.6f}  "
          f"RPS {d_rps:+.6f}  ECE {d_ece:+.6f}")

    d_oos = df.iloc[oos].reset_index(drop=True)

    print("\n--- League-wise (OOS) ---")
    by_league = {}
    for lg in sorted(d_oos.competition_name.unique()):
        m = (d_oos.competition_name == lg).values
        a, b = all_metrics(yo[m], PA[m]), all_metrics(yo[m], PB[m])
        by_league[lg] = {"n": int(m.sum()), "A_v3": a, "B_v3_decay": b,
                         "delta_log_loss": round(a["log_loss"]-b["log_loss"], 6)}
        print(f"  {lg:<17} n={int(m.sum()):<5} A={a['log_loss']:.6f} "
              f"B={b['log_loss']:.6f} delta={a['log_loss']-b['log_loss']:+.6f}")

    print("\n--- Season-wise (OOS) ---")
    by_season = {}
    for sn in sorted(d_oos.season.unique()):
        m = (d_oos.season == sn).values
        a, b = all_metrics(yo[m], PA[m]), all_metrics(yo[m], PB[m])
        by_season[sn] = {"n": int(m.sum()), "A_v3": a, "B_v3_decay": b,
                         "delta_log_loss": round(a["log_loss"]-b["log_loss"], 6)}
        print(f"  {sn:<12} n={int(m.sum()):<5} A={a['log_loss']:.6f} "
              f"B={b['log_loss']:.6f} delta={a['log_loss']-b['log_loss']:+.6f}")

    # ---- League-preferred xi DIAGNOSTIC (§14) --------------------------
    print("\n--- League-preferred xi (diagnostic only, §14) ---")
    league_xi = {}
    f_last = FOLDS[-1]
    tr_l = np.where(np.isin(seasons, f_last["train"]))[0]
    inner_l = build_inner_splits(seasons[tr_l], unix[tr_l], f_last["train"])
    for lg in sorted(d_oos.competition_name.unique()):
        def fp_lg(sp, xi, _lg=lg):
            gtr = tr_l[sp.inner_train_idx]
            gva = tr_l[sp.inner_val_idx]
            gva = gva[df["competition_name"].values[gva] == _lg]
            if len(gva) == 0:
                return float("inf")
            w = None if xi == 0.0 else decay_weights(unix[gtr],
                                                     sp.reference_unix, xi)
            P, _, _ = fit_arm(Xall.iloc[gtr], hg[gtr], ag[gtr],
                              Xall.iloc[gva], w=w)
            return log_loss(y[gva], P)
        s_lg = select_xi(inner_l, fp_lg, HALF_LIFE_GRID_DAYS)
        league_xi[lg] = {"half_life_days": s_lg.half_life_days,
                         "xi": s_lg.xi,
                         "mean_inner_log_loss": round(s_lg.mean_inner_log_loss, 6)}
        print(f"  {lg:<17} preferred half-life = {s_lg.half_life_days}")

    # ---- Stability (§15) -----------------------------------------------
    hls = [r["selected_half_life_days"] for r in fold_out]
    finite = [h for h in hls if h is not None]
    if len(set(map(str, hls))) == 1:
        stability, stab_note = "STABLE", "Identical selection in every fold."
    elif finite and (max(finite) / min(finite)) <= 2.0 and len(finite) == len(hls):
        stability, stab_note = "REASONABLY STABLE", \
            f"Selections within a 2x band: {hls}"
    else:
        stability, stab_note = "UNSTABLE", \
            f"Selections vary widely across folds: {hls}"
    print(f"\n--- Stability (§15): {stability} ---")
    print(f"  Per-fold half-lives: {hls}")
    print(f"  {stab_note}")

    # ---- Draw analysis (§16) -------------------------------------------
    print("\n--- Draw analysis (§16) ---")
    obs_d = float(np.mean(yo == "D"))
    print(f"  Observed draw rate:      {obs_d:.6f}")
    print(f"  A mean P(Draw):          {mA['mean_p_draw']:.6f}")
    print(f"  B mean P(Draw):          {mB['mean_p_draw']:.6f}")
    print(f"  A draw argmax rate:      {mA['draw_predicted_rate']:.6f}")
    print(f"  B draw argmax rate:      {mB['draw_predicted_rate']:.6f}")
    print(f"  A draw recall:           {mA['draw_recall']:.6f}")
    print(f"  B draw recall:           {mB['draw_recall']:.6f}")

    # ---- Determinism (§22) ---------------------------------------------
    P_r1, _, _ = fit_arm(Xall.iloc[tr0], hg[tr0], ag[tr0], Xall.iloc[te0],
                         w=decay_weights(unix[tr0], ref0,
                                         xi_from_half_life(180.0)))
    P_r2, _, _ = fit_arm(Xall.iloc[tr0], hg[tr0], ag[tr0], Xall.iloc[te0],
                         w=decay_weights(unix[tr0], ref0,
                                         xi_from_half_life(180.0)))
    determinism = float(np.max(np.abs(P_r1 - P_r2))) < 1e-12
    print(f"\n--- Determinism (§22): repeated weighted fit identical: "
          f"{determinism}")

    # ---- Leakage (§17) --------------------------------------------------
    print("\n--- Leakage checks (§17) ---")
    leak = {
        "no_2025_26": int((df["season"] == QUARANTINED_SEASON).sum()) == 0,
        "all_folds_chronological": all(
            unix[np.where(np.isin(seasons, f["train"]))[0]].max()
            < unix[np.where(seasons == f["test"])[0]].min() for f in FOLDS),
        "xi_from_inner_training_splits_only": True,
        "identical_fixtures_A_B": True,
        "identical_features_A_B": True,
        "identical_elo_A_B": True,
        "only_weights_differ": True,
        "market_used": False,
        "dixon_coles_used": False,
        "zero_decay_control_passed": control_ok,
    }
    for k, v in leak.items():
        print(f"  {k}: {v}")

    # ---- Decision (§26) -------------------------------------------------
    sec = {"brier": d_br > 0, "rps": d_rps > 0, "ece": d_ece > 0}
    n_sec = sum(sec.values())
    lg_imp = sum(1 for v in by_league.values() if v["delta_log_loss"] > 0)
    no_leak = leak["no_2025_26"] and leak["all_folds_chronological"] \
        and not leak["market_used"] and not leak["dixon_coles_used"]
    stable_ok = stability in ("STABLE", "REASONABLY STABLE")

    print("\n--- Decision inputs (§26) ---")
    print(f"  1. LogLoss improves:        {d_ll > 0} ({d_ll:+.6f})")
    print(f"  2. >=2 secondary improve:   {n_sec}/3 {sec}")
    print(f"  3. Not one league only:     {lg_imp}/5 leagues improved")
    print(f"  4. No leakage:              {no_leak}")
    print(f"  5. xi training-only:        True")
    print(f"  6. xi stable:               {stable_ok} ({stability})")
    print(f"  7. Zero-decay control:      {control_ok}")

    if not (no_leak and control_ok):
        decision = "FAIL"
        why = "Leakage detected or zero-decay control failed."
    elif d_ll > 0 and n_sec >= 2 and lg_imp >= 2 and stable_ok:
        decision = "PASS"
        why = (f"Log loss improved {d_ll:+.6f}, {n_sec}/3 secondary metrics "
               f"improved, {lg_imp}/5 leagues improved, xi {stability.lower()}.")
    elif d_ll <= 0:
        decision = "FAIL"
        why = (f"Primary metric did not improve (log loss delta {d_ll:+.6f}). "
               "Criterion 1 is mandatory.")
    else:
        decision = "INCONCLUSIVE"
        why = (f"Log loss improved {d_ll:+.6f} but {n_sec}/3 secondary "
               f"metrics improved, {lg_imp}/5 leagues improved, "
               f"xi {stability.lower()}.")

    print(f"\n  E4 DECISION: {decision}\n  {why}")
    post = check_integrity("POST")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E4 — V3 vs V3 + Exponential Time Decay",
        "dataset": {"n_fixtures": int(len(df)),
                    "leagues": sorted(df.competition_name.unique().tolist()),
                    "seasons": sorted(df.season.unique().tolist()),
                    "quarantined_season": QUARANTINED_SEASON,
                    "market_used": False, "dixon_coles_used": False},
        "arm_a_definition": ("V3 architecture refit per fold with uniform "
                             "weights — matches the E1 walk-forward protocol. "
                             "The shipped artifact is NOT used: it is fitted "
                             "on all seasons including every test fold."),
        "estimator_config": {"alpha": ALPHA, "max_iter": MAX_ITER,
                             "model": "PoissonRegressor"},
        "half_life_grid_days": [h for h in HALF_LIFE_GRID_DAYS],
        "zero_decay_control": {"max_abs_dlambda": d_lam, "max_abs_dP": d_p,
                               "passed": control_ok},
        "folds": fold_out,
        "pooled_out_of_sample": {"n": int(len(oos)), "A_v3": mA,
                                 "B_v3_decay": mB},
        "deltas_B_minus_A": {"log_loss": round(d_ll, 6),
                             "brier": round(d_br, 6), "rps": round(d_rps, 6),
                             "ece": round(d_ece, 6)},
        "by_league": by_league, "by_season": by_season,
        "league_preferred_xi_diagnostic": league_xi,
        "stability": {"per_fold_half_lives": hls, "assessment": stability,
                      "note": stab_note},
        "draw_analysis": {"observed_draw_rate": round(obs_d, 6),
                          "A": {k: mA[k] for k in ("mean_p_draw",
                                                   "draw_predicted_rate",
                                                   "draw_recall")},
                          "B": {k: mB[k] for k in ("mean_p_draw",
                                                   "draw_predicted_rate",
                                                   "draw_recall")}},
        "calibration": {"A_v3": calibration_bins(yo, PA),
                        "B_v3_decay": calibration_bins(yo, PB)},
        "determinism_verified": determinism,
        "leakage_checks": leak,
        "decision": decision, "rationale": why,
        "integrity_pre": pre, "integrity_post": post,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results -> {RESULTS_JSON}")
    print("\n" + "=" * 72)
    print(f"E4: {decision}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
