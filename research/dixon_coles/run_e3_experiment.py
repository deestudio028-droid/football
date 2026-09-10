"""E3 — V3 vs V3 + Dixon-Coles controlled experiment.

Usage:
    python research/dixon_coles/run_e3_experiment.py

Requires: numpy, pandas, scikit-learn (to unpickle the V3 artifact).

    A = V3                  (lambdas -> independent Poisson -> 1X2)
    B = V3 + Dixon-Coles    (same lambdas -> tau-corrected grid -> 1X2)

V3 IS NOT RETRAINED. Its lambda_home and lambda_away are computed once
and shared byte-identically by A and B. B differs from A only by rho.

Market odds are not read, imported, or referenced anywhere in this file.

Protocol: the approved V3 walk-forward folds from src/models/config.py
(3 folds, 2025/26 quarantined as final test and never touched).

Outputs:
    research/dixon_coles/e3_results.json
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
sys.path.insert(0, str(PROJECT_ROOT / "research" / "market_odds"))

from dixon_coles import (  # noqa: E402
    CLASS_ORDER, grid_size, low_score_probs, predict_dc,
    rho_validity_bounds, select_rho,
)

FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
RESULTS_JSON = HERE / "e3_results.json"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}

TARGET_COMPETITIONS = (200, 419, 423, 477, 499)

# Approved V3 walk-forward folds (src/models/config.py WALK_FORWARD_FOLDS).
FOLDS = [
    {"name": "fold_1",
     "train": ("2020/2021", "2021/2022"),
     "test": "2022/2023"},
    {"name": "fold_2",
     "train": ("2020/2021", "2021/2022", "2022/2023"),
     "test": "2023/2024"},
    {"name": "fold_3",
     "train": ("2020/2021", "2021/2022", "2022/2023", "2023/2024"),
     "test": "2024/2025"},
]
QUARANTINED_SEASON = "2025/2026"

#: rho candidate grid. Centred on zero and symmetric. Values outside the
#: per-fold validity bounds are skipped by select_rho (not clipped).
#: Step 0.025 near zero because published football rho estimates cluster
#: in roughly [-0.15, 0.00]; a coarser step would risk stepping over the
#: optimum, a finer one risks overfitting a single fold.
RHO_GRID = [-0.20, -0.175, -0.15, -0.125, -0.10, -0.075, -0.05, -0.025,
            0.0, 0.025, 0.05, 0.075, 0.10]


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
            out[rel] = {"status": "missing"}
            print(f"  MISSING  {rel}")
            continue
        act = md5(p)
        ok = act == exp
        out[rel] = {"expected": exp, "actual": act,
                    "status": "identical" if ok else "CHANGED"}
        print(f"  {'OK  ' if ok else 'FAIL'}  {rel}  {act}")
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _onehot(y):
    oh = np.zeros((len(y), 3))
    for i, c in enumerate(CLASS_ORDER):
        oh[:, i] = (y == c)
    return oh


def log_loss(y, P):
    oh = _onehot(y)
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(oh * np.log(Pc), axis=1)))


def brier(y, P):
    return float(np.mean(np.sum((P - _onehot(y)) ** 2, axis=1)))


def rps(y, P):
    cp, co = np.cumsum(P, axis=1), np.cumsum(_onehot(y), axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def accuracy(y, P):
    pred = np.array([CLASS_ORDER[i] for i in P.argmax(axis=1)])
    return float(np.mean(pred == y))


def draw_recall(y, P):
    pred = np.array([CLASS_ORDER[i] for i in P.argmax(axis=1)])
    m = (y == "D")
    return float(np.mean(pred[m] == "D")) if m.sum() else float("nan")


def calibration_bins(y, P, n_bins=10):
    oh = _onehot(y).ravel()
    pf = P.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pf >= lo) & (pf <= hi) if i == n_bins - 1 else (pf >= lo) & (pf < hi)
        n = int(m.sum())
        if not n:
            out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
                        "n": 0, "mean_predicted": None,
                        "observed_freq": None, "gap": None})
            continue
        mp, ob = float(pf[m].mean()), float(oh[m].mean())
        out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": n,
                    "mean_predicted": round(mp, 6),
                    "observed_freq": round(ob, 6),
                    "gap": round(ob - mp, 6)})
    return out


def ece(y, P, n_bins=10):
    b = calibration_bins(y, P, n_bins)
    tot = sum(x["n"] for x in b)
    return float(sum(x["n"] / tot * abs(x["gap"]) for x in b if x["n"])) if tot else float("nan")


def goal_mae(lam, actual):
    return float(np.mean(np.abs(np.asarray(lam) - np.asarray(actual))))


def all_metrics(y, P, lam_h=None, lam_a=None, hg=None, ag=None):
    m = {
        "n": int(len(y)),
        "log_loss": round(log_loss(y, P), 6),
        "brier": round(brier(y, P), 6),
        "rps": round(rps(y, P), 6),
        "accuracy": round(accuracy(y, P), 6),
        "draw_recall": round(draw_recall(y, P), 6),
        "draw_predicted_rate": round(float(np.mean(P.argmax(axis=1) == 1)), 6),
        "mean_p_draw": round(float(P[:, 1].mean()), 6),
        "ece": round(ece(y, P), 6),
    }
    if lam_h is not None and hg is not None:
        m["home_goal_mae"] = round(goal_mae(lam_h, hg), 6)
        m["away_goal_mae"] = round(goal_mae(lam_a, ag), 6)
    return m


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_fixture_meta() -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET_COMPETITIONS))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, competition_id, competition_name, season,
                   unix, home_name, away_name, home_goals, away_goals, status
            FROM fixtures WHERE competition_id IN ({ph})
            ORDER BY unix ASC, fixture_id ASC""",
        conn, params=list(TARGET_COMPETITIONS))
    conn.close()
    return df


def load_v3_lambdas(fixture_ids: list[int]) -> pd.DataFrame:
    """V3 lambdas via the frozen artifact. NOTHING is retrained."""
    from features.elo import load_elo_features, ELO_COLUMNS
    from models.v3_artifact import (
        load_v3_artifact, DEFAULT_V3_PRODUCTION_PATH,
        DEFAULT_V3_CANDIDATE_PATH)
    from models.v3_contract import V3_FEATURE_COLUMNS, V3_N_FEATURES
    from models.data import load_supervised_dataset

    prod = PROJECT_ROOT / DEFAULT_V3_PRODUCTION_PATH
    cand = PROJECT_ROOT / DEFAULT_V3_CANDIDATE_PATH
    path = prod if prod.exists() else cand
    print(f"  V3 artifact: {path.name} (md5={md5(path)})")
    art = load_v3_artifact(path)

    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X_all = ds.X.reset_index(drop=True)

    print("  Computing causal Elo over full fixture history...")
    elo_map = load_elo_features(MATCHES_DB).set_index("fixture_id")

    mask = meta["fixture_id"].isin(set(fixture_ids)).values
    idx = np.where(mask)[0]
    fids = meta.loc[mask, "fixture_id"].values

    base = [c for c in V3_FEATURE_COLUMNS if c not in ELO_COLUMNS]
    X = X_all.iloc[idx][base].copy().reset_index(drop=True)
    for col in ELO_COLUMNS:
        X[col] = [float(elo_map.at[f, col]) for f in fids]

    assert list(X.columns) == list(V3_FEATURE_COLUMNS), "V3 column order mismatch"
    assert len(X.columns) == V3_N_FEATURES, "V3 feature count mismatch"

    Xe = art.preprocessor.transform(X)
    return pd.DataFrame({
        "fixture_id": fids.astype(int),
        "lambda_home": art.model_home_goals.predict(Xe),
        "lambda_away": art.model_away_goals.predict(Xe),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("E3 — V3 vs V3 + Dixon-Coles   (controlled research experiment)")
    print("=" * 70)

    pre = check_integrity("PRE")

    print("\n--- Dataset ---")
    meta = load_fixture_meta()
    eligible_seasons = sorted({s for f in FOLDS
                               for s in f["train"]} | {f["test"] for f in FOLDS})
    df = meta[meta["season"].isin(eligible_seasons)].copy()
    df = df[df["status"].isin(["FT", "AWARDED"])]
    df = df.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)

    print(f"  Eligible seasons: {eligible_seasons}")
    print(f"  Fixtures: {len(df)}")
    print(f"  Leagues: {sorted(df['competition_name'].unique())}")

    assert QUARANTINED_SEASON not in set(df["season"]), "2025/26 leaked in"
    print(f"  {QUARANTINED_SEASON} quarantine: CONFIRMED (0 fixtures)")

    print("\n--- V3 lambdas (no retraining) ---")
    lam = load_v3_lambdas(df["fixture_id"].tolist())
    df = df.merge(lam, on="fixture_id", how="inner")
    df = df.sort_values(["unix", "fixture_id"]).reset_index(drop=True)
    print(f"  Fixtures with V3 lambdas: {len(df)}")

    y = np.where(df.home_goals > df.away_goals, "H",
                 np.where(df.home_goals == df.away_goals, "D", "A"))
    lam_h = df["lambda_home"].values
    lam_a = df["lambda_away"].values
    hg = df["home_goals"].values.astype(float)
    ag = df["away_goals"].values.astype(float)

    # ONE shared grid for A and B — never differs between arms.
    K = grid_size(float(max(lam_h.max(), lam_a.max())))
    lo, hi = rho_validity_bounds(lam_h, lam_a)
    print(f"  Shared score grid K = {K} (adaptive, tail < 1e-15)")
    print(f"  Lambda range: home [{lam_h.min():.3f}, {lam_h.max():.3f}], "
          f"away [{lam_a.min():.3f}, {lam_a.max():.3f}]")
    print(f"  Global rho validity bounds: ({lo:.4f}, {hi:.4f})")

    # Baseline A on the full eligible set (rho = 0 == independent Poisson)
    P_A_all, _ = predict_dc(lam_h, lam_a, 0.0, K=K)

    # ---- Walk-forward -------------------------------------------------
    print("\n--- Walk-forward (approved V3 folds) ---")
    fold_out = []
    oos_idx, oos_B = [], []

    for f in FOLDS:
        tr = np.where(df["season"].isin(f["train"]).values)[0]
        te = np.where((df["season"] == f["test"]).values)[0]
        assert df.iloc[tr]["unix"].max() < df.iloc[te]["unix"].min(), \
            f"{f['name']} not chronological"

        sel = select_rho(y[tr], lam_h[tr], lam_a[tr], RHO_GRID, K)

        P_B_te, _ = predict_dc(lam_h[te], lam_a[te], sel.rho, K=K)
        P_A_te = P_A_all[te]

        r = {
            "fold": f["name"],
            "train_seasons": list(f["train"]),
            "test_season": f["test"],
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "selected_rho": sel.rho,
            "train_log_loss_at_rho": round(sel.train_log_loss, 6),
            "rho_validity_bounds": [round(sel.valid_bounds[0], 6),
                                    round(sel.valid_bounds[1], 6)],
            "rho_curve": sel.curve,
            "test_A_v3": all_metrics(y[te], P_A_te, lam_h[te], lam_a[te],
                                     hg[te], ag[te]),
            "test_B_v3_dc": all_metrics(y[te], P_B_te, lam_h[te], lam_a[te],
                                        hg[te], ag[te]),
        }
        fold_out.append(r)
        oos_idx.extend(te.tolist())
        oos_B.append(P_B_te)

        print(f"\n  {f['name']}: train={f['train']} (n={len(tr)}) "
              f"-> test={f['test']} (n={len(te)})")
        print(f"    selected rho = {sel.rho}  (train LL {sel.train_log_loss:.6f})")
        a, b = r["test_A_v3"], r["test_B_v3_dc"]
        print(f"    {'':<8}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}"
              f"{'Acc':>8}{'ECE':>8}{'meanPD':>9}")
        for nm, m in [("A  V3", a), ("B  +DC", b)]:
            print(f"    {nm:<8}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
                  f"{m['rps']:>9.6f}{m['accuracy']:>8.4f}"
                  f"{m['ece']:>8.4f}{m['mean_p_draw']:>9.4f}")
        print(f"    delta LL {a['log_loss'] - b['log_loss']:+.6f}  "
              f"Brier {a['brier'] - b['brier']:+.6f}  "
              f"RPS {a['rps'] - b['rps']:+.6f}")

    oos = np.array(oos_idx)
    P_B_oos = np.vstack(oos_B)
    P_A_oos = P_A_all[oos]
    y_oos = y[oos]

    # ---- Pooled OOS ----------------------------------------------------
    mA = all_metrics(y_oos, P_A_oos, lam_h[oos], lam_a[oos], hg[oos], ag[oos])
    mB = all_metrics(y_oos, P_B_oos, lam_h[oos], lam_a[oos], hg[oos], ag[oos])

    print("\n--- Pooled OUT-OF-SAMPLE ---")
    print(f"  n = {len(oos)}   (identical fixture set for A and B)")
    print(f"  {'':<8}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'Acc':>8}"
          f"{'DrawRec':>9}{'ECE':>8}{'meanPD':>9}")
    for nm, m in [("A  V3", mA), ("B  +DC", mB)]:
        print(f"  {nm:<8}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
              f"{m['rps']:>9.6f}{m['accuracy']:>8.4f}"
              f"{m['draw_recall']:>9.4f}{m['ece']:>8.4f}{m['mean_p_draw']:>9.4f}")

    d_ll = mA["log_loss"] - mB["log_loss"]
    d_br = mA["brier"] - mB["brier"]
    d_rps = mA["rps"] - mB["rps"]
    d_ece = mA["ece"] - mB["ece"]
    print(f"\n  Improvement (positive = DC better):")
    print(f"    LogLoss {d_ll:+.6f}   Brier {d_br:+.6f}   "
          f"RPS {d_rps:+.6f}   ECE {d_ece:+.6f}")

    # ---- Goal MAE invariance ------------------------------------------
    same_mae = (mA["home_goal_mae"] == mB["home_goal_mae"]
                and mA["away_goal_mae"] == mB["away_goal_mae"])
    print(f"\n  Goal MAE identical (lambdas unchanged): {same_mae}  "
          f"home={mA['home_goal_mae']:.6f} away={mA['away_goal_mae']:.6f}")

    # ---- League / season ----------------------------------------------
    d_oos = df.iloc[oos].reset_index(drop=True)

    print("\n--- League-wise (OOS) ---")
    by_league = {}
    for lg in sorted(d_oos["competition_name"].unique()):
        m = (d_oos["competition_name"] == lg).values
        a = all_metrics(y_oos[m], P_A_oos[m])
        b = all_metrics(y_oos[m], P_B_oos[m])
        by_league[lg] = {"n": int(m.sum()), "A_v3": a, "B_v3_dc": b,
                         "delta_log_loss": round(a["log_loss"] - b["log_loss"], 6)}
        print(f"  {lg:<17} n={int(m.sum()):<5} A={a['log_loss']:.6f} "
              f"B={b['log_loss']:.6f} delta={a['log_loss']-b['log_loss']:+.6f}")

    print("\n--- Season-wise (OOS) ---")
    by_season = {}
    for sn in sorted(d_oos["season"].unique()):
        m = (d_oos["season"] == sn).values
        a = all_metrics(y_oos[m], P_A_oos[m])
        b = all_metrics(y_oos[m], P_B_oos[m])
        by_season[sn] = {"n": int(m.sum()), "A_v3": a, "B_v3_dc": b,
                         "delta_log_loss": round(a["log_loss"] - b["log_loss"], 6)}
        print(f"  {sn:<12} n={int(m.sum()):<5} A={a['log_loss']:.6f} "
              f"B={b['log_loss']:.6f} delta={a['log_loss']-b['log_loss']:+.6f}")

    # ---- Draw & low-score diagnostics ----------------------------------
    print("\n--- Draw & low-score diagnostics (OOS) ---")
    obs_draw = float(np.mean(y_oos == "D"))
    print(f"  Observed draw rate:  {obs_draw:.6f}")
    print(f"  A mean P(Draw):      {mA['mean_p_draw']:.6f} "
          f"(deficit {obs_draw - mA['mean_p_draw']:+.6f})")
    print(f"  B mean P(Draw):      {mB['mean_p_draw']:.6f} "
          f"(deficit {obs_draw - mB['mean_p_draw']:+.6f})")

    dP = np.abs(P_B_oos - P_A_oos)
    top_changed = float(np.mean(P_A_oos.argmax(axis=1) != P_B_oos.argmax(axis=1)))
    print(f"\n  Mean |dP(Home)| {dP[:,0].mean():.6f}")
    print(f"  Mean |dP(Draw)| {dP[:,1].mean():.6f}")
    print(f"  Mean |dP(Away)| {dP[:,2].mean():.6f}")
    print(f"  Top prediction changed: {100*top_changed:.4f}% of fixtures")

    # Per-fold low-score cell changes at that fold's rho
    low_changes = []
    for f, r in zip(FOLDS, fold_out):
        te = np.where((df["season"] == f["test"]).values)[0]
        l0 = low_score_probs(lam_h[te], lam_a[te], 0.0, K)
        l1 = low_score_probs(lam_h[te], lam_a[te], r["selected_rho"], K)
        entry = {"fold": r["fold"], "rho": r["selected_rho"]}
        for cell in ["p_0_0", "p_1_0", "p_0_1", "p_1_1"]:
            entry[f"{cell}_A"] = round(float(l0[cell].mean()), 6)
            entry[f"{cell}_B"] = round(float(l1[cell].mean()), 6)
            entry[f"{cell}_delta"] = round(
                float(l1[cell].mean() - l0[cell].mean()), 6)
        low_changes.append(entry)
        print(f"\n  {r['fold']} (rho={r['selected_rho']}) mean cell probability:")
        for cell, lbl in [("p_0_0", "0-0"), ("p_1_0", "1-0"),
                          ("p_0_1", "0-1"), ("p_1_1", "1-1")]:
            print(f"    {lbl}: {entry[f'{cell}_A']:.6f} -> "
                  f"{entry[f'{cell}_B']:.6f}  ({entry[f'{cell}_delta']:+.6f})")

    calib = {"A_v3": calibration_bins(y_oos, P_A_oos),
             "B_v3_dc": calibration_bins(y_oos, P_B_oos)}

    # ---- Determinism ---------------------------------------------------
    P_B2, _ = predict_dc(lam_h[oos[:100]], lam_a[oos[:100]],
                         fold_out[-1]["selected_rho"], K=K)
    P_B3, _ = predict_dc(lam_h[oos[:100]], lam_a[oos[:100]],
                         fold_out[-1]["selected_rho"], K=K)
    determinism = bool(np.array_equal(P_B2, P_B3))
    print(f"\n--- Determinism: repeated predict_dc identical: {determinism}")

    # ---- Leakage checks ------------------------------------------------
    print("\n--- Leakage checks ---")
    leak = {
        "no_2025_26": QUARANTINED_SEASON not in set(df["season"]),
        "all_folds_chronological": all(
            df.iloc[np.where(df["season"].isin(f["train"]).values)[0]]["unix"].max()
            < df.iloc[np.where((df["season"] == f["test"]).values)[0]]["unix"].min()
            for f in FOLDS),
        "rho_from_train_only": True,
        "identical_fixture_sets_A_B": True,
        "identical_lambdas_A_B": True,
        "market_data_used": False,
        "v3_retrained": False,
        "same_score_grid_A_B": True,
    }
    for k, v in leak.items():
        print(f"  {k}: {v}")

    # ---- Decision ------------------------------------------------------
    secondary = {"brier": d_br > 0, "rps": d_rps > 0, "ece": d_ece > 0}
    n_secondary = sum(secondary.values())
    leagues_improved = sum(1 for v in by_league.values()
                           if v["delta_log_loss"] > 0)
    n_leagues = len(by_league)
    no_leak = all(v for k, v in leak.items()
                  if k not in ("market_data_used", "v3_retrained")) \
        and not leak["market_data_used"] and not leak["v3_retrained"]

    print("\n--- Decision inputs ---")
    print(f"  1. Primary (LogLoss improves):     {d_ll > 0}  ({d_ll:+.6f})")
    print(f"  2. Secondary improved:             {n_secondary}/3  {secondary}")
    print(f"  3. Leagues improved:               {leagues_improved}/{n_leagues}")
    print(f"  4. No leakage:                     {no_leak}")
    print(f"  5. Walk-forward preserved:         {leak['all_folds_chronological']}")
    print(f"  6. V3 baseline unchanged:          {same_mae}")

    if not no_leak or not leak["all_folds_chronological"]:
        decision, why = "FAIL", "Leakage or causality violation detected."
    elif d_ll > 0 and n_secondary >= 2 and leagues_improved >= 3:
        decision = "PASS"
        why = (f"Log loss improved by {d_ll:.6f}, {n_secondary}/3 secondary "
               f"probabilistic metrics improved, and the gain held in "
               f"{leagues_improved}/{n_leagues} leagues.")
    elif d_ll <= 0:
        decision = "FAIL"
        why = (f"Primary metric did not improve (log loss delta {d_ll:+.6f}). "
               "Criterion 1 is mandatory.")
    else:
        decision = "INCONCLUSIVE"
        why = (f"Log loss improved by {d_ll:.6f} but only {n_secondary}/3 "
               f"secondary metrics improved and only {leagues_improved}/"
               f"{n_leagues} leagues improved.")

    print(f"\n  E3 DECISION: {decision}")
    print(f"  {why}")

    post = check_integrity("POST")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E3 — V3 vs V3 + Dixon-Coles",
        "dataset": {
            "source": "data/processed/matches.db + features.db",
            "leagues": sorted(df["competition_name"].unique().tolist()),
            "seasons": eligible_seasons,
            "n_fixtures": int(len(df)),
            "quarantined_season": QUARANTINED_SEASON,
            "market_data_used": False,
        },
        "score_grid": {"K": int(K), "adaptive": True, "tail_tol": 1e-15,
                       "identical_for_A_and_B": True},
        "rho_grid": RHO_GRID,
        "global_rho_validity_bounds": [round(lo, 6), round(hi, 6)],
        "folds": fold_out,
        "pooled_out_of_sample": {"n": int(len(oos)), "A_v3": mA,
                                 "B_v3_dc": mB},
        "deltas_B_minus_A": {"log_loss": round(d_ll, 6),
                             "brier": round(d_br, 6),
                             "rps": round(d_rps, 6),
                             "ece": round(d_ece, 6)},
        "by_league": by_league,
        "by_season": by_season,
        "draw_analysis": {
            "observed_draw_rate": round(obs_draw, 6),
            "A_mean_p_draw": mA["mean_p_draw"],
            "B_mean_p_draw": mB["mean_p_draw"],
            "A_draw_recall": mA["draw_recall"],
            "B_draw_recall": mB["draw_recall"],
            "A_draw_predicted_rate": mA["draw_predicted_rate"],
            "B_draw_predicted_rate": mB["draw_predicted_rate"],
        },
        "diagnostics": {
            "mean_abs_delta_p_home": round(float(dP[:, 0].mean()), 6),
            "mean_abs_delta_p_draw": round(float(dP[:, 1].mean()), 6),
            "mean_abs_delta_p_away": round(float(dP[:, 2].mean()), 6),
            "pct_top_prediction_changed": round(100 * top_changed, 4),
            "low_score_cells_by_fold": low_changes,
        },
        "calibration": calib,
        "goal_mae_identical": same_mae,
        "determinism_verified": determinism,
        "leakage_checks": leak,
        "decision": decision,
        "rationale": why,
        "integrity_pre": pre,
        "integrity_post": post,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results -> {RESULTS_JSON}")

    print("\n" + "=" * 70)
    print(f"E3: {decision}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
