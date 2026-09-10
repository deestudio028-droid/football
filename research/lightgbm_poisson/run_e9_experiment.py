"""E9 — V3 linear Poisson vs LightGBM Poisson.

Usage:
    python research/lightgbm_poisson/run_e9_experiment.py

Requires: numpy, pandas, scikit-learn, lightgbm.

    ARM A   87 features -> PoissonRegressor(alpha=1.0, max_iter=2000)
    ARM B   87 features -> LGBMRegressor(objective="poisson")

Single-variable experiment: the ONLY conceptual change is the learner.
Both arms receive a byte-identical encoded matrix (median-imputed from
training-fold statistics, standardised, one-hot competition_id), the
same targets, the same folds, and the SAME adaptive score-grid K.

Universes reported SEPARATELY, never mixed:
    PRIMARY  full V3/E6 OOS  n = 5,331   (honest V3 reference ~0.987025)
    MARKET   overlap subset  n = 3,479   (market reference 0.955905)

No market features. No Dixon-Coles, time decay, quadratic Elo or online
A/D. No calibration. No early stopping.

Outputs: research/lightgbm_poisson/e9_results.json
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

from lightgbm_poisson import (  # noqa: E402
    CLASS_ORDER, E9Error, V3Preprocessor, all_metrics, build_inner_splits,
    config_params, feature_importance, fit_lgbm_arm, fit_v3_arm,
    hda_from_lambdas, lambda_diagnostics, lightgbm_version, load_grid,
    log_loss, overfitting_diagnostics, prediction_change, reliability_bins,
    select_config, shared_grid_k, sklearn_version, validate_probs,
)

FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
MARKET_DB = PROJECT_ROOT / "research" / "market_odds" / "research_dataset.sqlite"
RESULTS_JSON = HERE / "e9_results.json"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
TARGET = (200, 419, 423, 477, 499)
SEASONS = ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025")
FOLDS = [
    {"name": "fold_1", "train": ("2020/2021", "2021/2022"), "test": "2022/2023"},
    {"name": "fold_2", "train": ("2020/2021", "2021/2022", "2022/2023"),
     "test": "2023/2024"},
    {"name": "fold_3", "train": ("2020/2021", "2021/2022", "2022/2023",
                                 "2023/2024"), "test": "2024/2025"},
]
QUARANTINED = "2025/2026"

#: Honest references, recorded before the run (see E9_AUDIT_REPORT.md).
V3_REF_5331 = 0.987025      # E6's measured V3 on the full OOS universe
V3_REF_3479 = 0.990435      # E7's measured V3 on the market overlap
MARKET_REF_3479 = 0.955905  # E2/E8 market alone, reproduced exactly


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
    assert len(V3_FEATURE_COLUMNS) == V3_N_FEATURES == 87, "contract drift"

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET)); sph = ",".join("?" * len(SEASONS))
    fx = pd.read_sql_query(
        f"""SELECT fixture_id, competition_name, season, unix,
                   home_goals, away_goals FROM fixtures
            WHERE competition_id IN ({ph}) AND season IN ({sph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL""",
        conn, params=[*TARGET, *SEASONS])
    conn.close()

    j = meta[["fixture_id"]].join(X[list(V3_FEATURE_COLUMNS)])
    j = j.merge(fx, on="fixture_id", how="inner")
    j = j[j["season"] != QUARANTINED]
    j = j.sort_values(["unix", "fixture_id"]).reset_index(drop=True)

    market_ids = set()
    if MARKET_DB.exists():
        conn = sqlite3.connect(f"file:{MARKET_DB}?mode=ro", uri=True)
        mk = pd.read_sql_query("SELECT fixture_id, season FROM research_odds", conn)
        conn.close()
        market_ids = set(mk[mk.season.isin(["2023/2024", "2024/2025"])].fixture_id)

    y = np.where(j.home_goals > j.away_goals, "H",
                 np.where(j.home_goals == j.away_goals, "D", "A"))
    return j, tuple(V3_FEATURE_COLUMNS), y, market_ids


def main() -> int:
    print("=" * 78)
    print("E9 — V3 linear Poisson  vs  LightGBM Poisson")
    print("=" * 78)
    pre = check_integrity("PRE")

    print("\n--- Environment ---")
    lgb_v, sk_v = lightgbm_version(), sklearn_version()
    print(f"  lightgbm : {lgb_v}")
    print(f"  sklearn  : {sk_v}")
    print(f"  python   : {sys.version.split()[0]}")

    print("\n--- Dataset ---")
    df, V3COLS, y, market_ids = load_dataset()
    print(f"  Fixtures: {len(df)}   Leagues: {df.competition_name.nunique()}")
    print(f"  Seasons : {sorted(df.season.unique())}")
    assert (df.season == QUARANTINED).sum() == 0, "2025/26 leaked in"
    print(f"  {QUARANTINED} quarantine: CONFIRMED (0 fixtures)")
    print(f"  Feature contract: {len(V3COLS)} columns")

    seasons = df.season.values
    unix = df.unix.values.astype(float)
    hg = df.home_goals.values.astype(float)
    ag = df.away_goals.values.astype(float)
    fids = df.fixture_id.to_numpy()
    Xall = df[list(V3COLS)]

    grid = load_grid()
    configs, fixed = grid["configurations"], grid["fixed_for_all_configs"]
    print(f"  Frozen configurations: {len(configs)}")

    fold_out, oos = [], []
    PA_l, PB_l, lamA_l, lamB_l = [], [], [], []
    importance_by_fold = {}

    for f in FOLDS:
        tr = np.where(np.isin(seasons, f["train"]))[0]
        te = np.where(seasons == f["test"])[0]
        assert unix[tr].max() < unix[te].min(), f"{f['name']} not chronological"

        print(f"\n  {f['name']}: train={f['train']} (n={len(tr)}) "
              f"-> test={f['test']} (n={len(te)})")

        # ---- nested configuration selection, TRAINING ONLY -------------
        inner = build_inner_splits(seasons[tr], f["train"])

        def fit_eval(sp, params):
            g_tr, g_va = tr[sp.train_idx], tr[sp.val_idx]
            pre_in = V3Preprocessor().fit(Xall.iloc[g_tr])
            Etr, Eva = (pre_in.transform(Xall.iloc[g_tr]),
                        pre_in.transform(Xall.iloc[g_va]))
            lh, la = fit_lgbm_arm(Etr, hg[g_tr], ag[g_tr], Eva, params)
            k = shared_grid_k(lh, la)
            return log_loss(y[g_va], hda_from_lambdas(lh, la, k))

        sel = select_config(inner, fit_eval, configs, fixed)
        print(f"    inner splits: {sel.n_inner_splits}   "
              f"selected config: {sel.config_id}")
        print(f"    params: leaves={sel.params['num_leaves']} "
              f"lr={sel.params['learning_rate']} "
              f"trees={sel.params['n_estimators']} "
              f"min_child={sel.params['min_child_samples']} "
              f"sub={sel.params['subsample']} "
              f"col={sel.params['colsample_bytree']} "
              f"L2={sel.params['reg_lambda']}")

        # ---- ONE preprocessor -> byte-identical matrix for BOTH arms ---
        prep = V3Preprocessor().fit(Xall.iloc[tr])
        Etr = prep.transform(Xall.iloc[tr])
        Ete = prep.transform(Xall.iloc[te])

        lam_h_a, lam_a_a = fit_v3_arm(Etr, hg[tr], ag[tr], Ete)
        lam_h_b, lam_a_b, models = fit_lgbm_arm(
            Etr, hg[tr], ag[tr], Ete, sel.params, return_models=True)

        # in-sample lambdas for the overfitting diagnostic
        lh_a_tr, la_a_tr = fit_v3_arm(Etr, hg[tr], ag[tr], Etr)
        lh_b_tr, la_b_tr = fit_lgbm_arm(Etr, hg[tr], ag[tr], Etr, sel.params)

        # ---- SHARED adaptive K across BOTH arms ------------------------
        K = shared_grid_k(lam_h_a, lam_a_a, lam_h_b, lam_a_b)
        P_A = hda_from_lambdas(lam_h_a, lam_a_a, K)
        P_B = hda_from_lambdas(lam_h_b, lam_a_b, K)
        validate_probs(P_A, "Arm A"); validate_probs(P_B, "Arm B")
        print(f"    shared score-grid K = {K} (pooled max lambda, both arms)")

        K_tr = shared_grid_k(lh_a_tr, la_a_tr, lh_b_tr, la_b_tr)
        P_A_tr = hda_from_lambdas(lh_a_tr, la_a_tr, K_tr)
        P_B_tr = hda_from_lambdas(lh_b_tr, la_b_tr, K_tr)

        diag_a = lambda_diagnostics(lam_h_a, lam_a_a)
        diag_b = lambda_diagnostics(lam_h_b, lam_a_b)
        if not diag_b["passed"]:
            print("\n    STOP: LightGBM produced unsafe lambdas (§19).")
            print(f"    {json.dumps(diag_b, indent=4)}")
            return 1

        mA = all_metrics(y[te], P_A, lam_h_a, lam_a_a, hg[te], ag[te])
        mB = all_metrics(y[te], P_B, lam_h_b, lam_a_b, hg[te], ag[te])

        over_a = overfitting_diagnostics(y[tr], P_A_tr, None, None, y[te], P_A)
        over_b = overfitting_diagnostics(y[tr], P_B_tr, None, None, y[te], P_B)
        over_b["inner_val_log_loss"] = round(sel.mean_inner_log_loss, 6)
        over_b["val_test_gap"] = round(
            mB["log_loss"] - sel.mean_inner_log_loss, 6)

        fi = feature_importance(models, list(V3COLS))
        importance_by_fold[f["name"]] = fi

        fold_out.append({
            "fold": f["name"], "train_seasons": list(f["train"]),
            "test_season": f["test"], "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_inner_splits": sel.n_inner_splits,
            "selected_config_id": sel.config_id,
            "selected_params": {k: v for k, v in sel.params.items()
                                if k not in ("verbose",)},
            "mean_inner_log_loss": round(sel.mean_inner_log_loss, 6),
            "config_curve": sel.curve,
            "shared_grid_K": int(K),
            "arm_a_v3": mA, "arm_b_lgbm": mB,
            "deltas_b_minus_a": {k: round(mA[k] - mB[k], 8)
                                 for k in ("log_loss", "brier", "rps", "ece")},
            "lambda_diagnostics": {"arm_a": diag_a, "arm_b": diag_b},
            "overfitting": {"arm_a": over_a, "arm_b": over_b},
            "prediction_change": prediction_change(
                P_A, P_B, lam_h_a, lam_a_a, lam_h_b, lam_a_b),
            "feature_importance_top5_home": fi["home"][:5],
        })
        oos.extend(te.tolist())
        PA_l.append(P_A); PB_l.append(P_B)
        lamA_l.append((lam_h_a, lam_a_a)); lamB_l.append((lam_h_b, lam_a_b))

        print(f"    {'':<9}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'ECE':>9}"
              f"{'Acc':>8}{'hMAE':>8}{'aMAE':>8}")
        for nm, m in [("A V3", mA), ("B LGBM", mB)]:
            print(f"    {nm:<9}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
                  f"{m['rps']:>9.6f}{m['ece']:>9.6f}{m['accuracy']:>8.4f}"
                  f"{m['home_goal_mae']:>8.4f}{m['away_goal_mae']:>8.4f}")
        d = fold_out[-1]["deltas_b_minus_a"]
        print(f"    delta LL {d['log_loss']:+.6f}  Brier {d['brier']:+.6f}  "
              f"RPS {d['rps']:+.6f}  ECE {d['ece']:+.6f}")
        print(f"    train/test gap: A={over_a['train_test_gap']:+.6f}  "
              f"B={over_b['train_test_gap']:+.6f}")

    oos = np.array(oos)
    PA, PB = np.vstack(PA_l), np.vstack(PB_l)
    yo = y[oos]
    lhA = np.concatenate([x[0] for x in lamA_l])
    laA = np.concatenate([x[1] for x in lamA_l])
    lhB = np.concatenate([x[0] for x in lamB_l])
    laB = np.concatenate([x[1] for x in lamB_l])

    mA = all_metrics(yo, PA, lhA, laA, hg[oos], ag[oos])
    mB = all_metrics(yo, PB, lhB, laB, hg[oos], ag[oos])
    deltas = {k: round(mA[k] - mB[k], 8)
              for k in ("log_loss", "brier", "rps", "ece")}

    print("\n" + "=" * 78)
    print("PRIMARY UNIVERSE — full V3/E6 OOS")
    print("=" * 78)
    print(f"  n = {len(oos)} (identical fixtures, both arms)")
    print(f"  {'':<9}{'LogLoss':>10}{'Brier':>9}{'RPS':>9}{'ECE':>9}"
          f"{'Acc':>8}{'DrawRec':>9}{'hMAE':>8}{'aMAE':>8}")
    for nm, m in [("A V3", mA), ("B LGBM", mB)]:
        print(f"  {nm:<9}{m['log_loss']:>10.6f}{m['brier']:>9.6f}"
              f"{m['rps']:>9.6f}{m['ece']:>9.6f}{m['accuracy']:>8.4f}"
              f"{m['draw_recall']:>9.4f}{m['home_goal_mae']:>8.4f}"
              f"{m['away_goal_mae']:>8.4f}")
    print(f"\n  LightGBM vs V3 (positive = LightGBM better):")
    for k, v in deltas.items():
        rel = 100 * v / mA[k] if mA[k] else 0.0
        print(f"    {k:<10} {v:+.8f}   ({rel:+.4f}% relative)")
    print(f"\n  Honest V3 reference (E6, same universe): {V3_REF_5331:.6f}")
    print(f"  E9 Arm A measured:                       {mA['log_loss']:.6f}  "
          f"(diff {abs(mA['log_loss'] - V3_REF_5331):.6f})")

    # ---- Market-overlap universe, reported SEPARATELY -------------------
    d_oos = df.iloc[oos].reset_index(drop=True)
    mk_mask = d_oos.fixture_id.isin(market_ids).values
    market_block = None
    if mk_mask.sum():
        mA_mk = all_metrics(yo[mk_mask], PA[mk_mask])
        mB_mk = all_metrics(yo[mk_mask], PB[mk_mask])
        market_block = {
            "n": int(mk_mask.sum()),
            "arm_a_v3": mA_mk, "arm_b_lgbm": mB_mk,
            "market_reference_log_loss": MARKET_REF_3479,
            "v3_reference_log_loss": V3_REF_3479,
            "lgbm_vs_market_log_loss": round(
                MARKET_REF_3479 - mB_mk["log_loss"], 8),
            "lgbm_beats_market": bool(mB_mk["log_loss"] < MARKET_REF_3479),
        }
        print("\n" + "=" * 78)
        print("MARKET-OVERLAP UNIVERSE — reported separately, never mixed")
        print("=" * 78)
        print(f"  n = {int(mk_mask.sum())}")
        print(f"  A  V3      LogLoss = {mA_mk['log_loss']:.6f}   "
              f"(E7 reference {V3_REF_3479:.6f})")
        print(f"  B  LGBM    LogLoss = {mB_mk['log_loss']:.6f}")
        print(f"     MARKET  LogLoss = {MARKET_REF_3479:.6f}  (reference only)")
        print(f"  LightGBM vs market: "
              f"{MARKET_REF_3479 - mB_mk['log_loss']:+.6f}  "
              f"-> beats market: {mB_mk['log_loss'] < MARKET_REF_3479}")

    # ---- League / season -------------------------------------------------
    by_league, by_season = {}, {}
    print("\n--- League-wise (primary universe) ---")
    print(f"  {'League':<17}{'n':>6}{'V3 LL':>11}{'LGBM LL':>11}{'delta':>12}")
    for lg in sorted(d_oos.competition_name.unique()):
        m = (d_oos.competition_name == lg).values
        a, b = all_metrics(yo[m], PA[m]), all_metrics(yo[m], PB[m])
        by_league[lg] = {"n": int(m.sum()), "arm_a_v3": a, "arm_b_lgbm": b,
                         "deltas": {k: round(a[k] - b[k], 8)
                                    for k in ("log_loss", "brier", "rps", "ece")}}
        print(f"  {lg:<17}{int(m.sum()):>6}{a['log_loss']:>11.6f}"
              f"{b['log_loss']:>11.6f}{a['log_loss']-b['log_loss']:>+12.6f}")

    print("\n--- Season-wise (primary universe) ---")
    for sn in sorted(d_oos.season.unique()):
        m = (d_oos.season == sn).values
        a, b = all_metrics(yo[m], PA[m]), all_metrics(yo[m], PB[m])
        by_season[sn] = {"n": int(m.sum()), "arm_a_v3": a, "arm_b_lgbm": b,
                         "deltas": {k: round(a[k] - b[k], 8)
                                    for k in ("log_loss", "brier", "rps", "ece")}}
        print(f"  {sn:<12} n={int(m.sum()):<5} V3={a['log_loss']:.6f} "
              f"LGBM={b['log_loss']:.6f} "
              f"delta={a['log_loss']-b['log_loss']:+.6f}")

    # ---- Pooled diagnostics ---------------------------------------------
    pooled_change = prediction_change(PA, PB, lhA, laA, lhB, laB)
    diag_pooled = {"arm_a": lambda_diagnostics(lhA, laA),
                   "arm_b": lambda_diagnostics(lhB, laB)}
    print("\n--- Prediction change (pooled) ---")
    for k, v in pooled_change.items(): print(f"  {k}: {v}")
    print("\n--- Lambda diagnostics (pooled, Arm B) ---")
    print(f"  {json.dumps(diag_pooled['arm_b'], indent=2)}")

    print("\n--- Top 10 features by gain (fold_3, home model) ---")
    for r in importance_by_fold[FOLDS[-1]["name"]]["home"][:10]:
        print(f"  {r['feature']:<45}{r['gain_pct']:>8.3f}%  splits={r['split']}")

    # ---- Config stability ------------------------------------------------
    cfg_ids = [r["selected_config_id"] for r in fold_out]
    cfg_stable = len(set(cfg_ids)) <= 2
    print(f"\n--- Configuration stability ---")
    print(f"  Selected config per fold: {cfg_ids}  "
          f"-> {'STABLE' if cfg_stable else 'UNSTABLE'}")

    # ---- Determinism ------------------------------------------------------
    f0 = FOLDS[0]
    tr0 = np.where(np.isin(seasons, f0["train"]))[0]
    te0 = np.where(seasons == f0["test"])[0]
    p0 = V3Preprocessor().fit(Xall.iloc[tr0])
    E0t, E0e = p0.transform(Xall.iloc[tr0]), p0.transform(Xall.iloc[te0])
    pr = config_params(configs[0], fixed)
    r1 = fit_lgbm_arm(E0t, hg[tr0], ag[tr0], E0e, pr)
    r2 = fit_lgbm_arm(E0t, hg[tr0], ag[tr0], E0e, pr)
    determinism = bool(np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1]))
    print(f"--- Determinism: repeated LightGBM fit identical: {determinism}")

    leak = {
        "no_2025_26": int((df.season == QUARANTINED).sum()) == 0,
        "all_folds_chronological": all(
            unix[np.where(np.isin(seasons, f["train"]))[0]].max()
            < unix[np.where(seasons == f["test"])[0]].min() for f in FOLDS),
        "config_from_inner_training_only": True,
        "medians_from_training_only": True,
        "identical_matrix_both_arms": True,
        "shared_score_grid_K": True,
        "market_used_as_feature": False,
        "online_ad_used": False, "dixon_coles_used": False,
        "time_decay_used": False, "quadratic_elo_used": False,
        "early_stopping_used": False,
        "clipping_applied": False,
    }
    print("\n--- Leakage checks ---")
    for k, v in leak.items(): print(f"  {k}: {v}")

    # ---- Decision --------------------------------------------------------
    sec = {"brier": deltas["brier"] > 0, "rps": deltas["rps"] > 0,
           "ece": deltas["ece"] > 0}
    n_sec = sum(sec.values())
    folds_improved = sum(1 for r in fold_out
                         if r["deltas_b_minus_a"]["log_loss"] > 0)
    lg_imp = sum(1 for v in by_league.values() if v["deltas"]["log_loss"] > 0)
    beats_market = market_block["lgbm_beats_market"] if market_block else None
    md5_ok = all(v.get("status") == "identical" for v in pre.values())
    safe = diag_pooled["arm_a"]["passed"] and diag_pooled["arm_b"]["passed"]
    no_leak = leak["no_2025_26"] and leak["all_folds_chronological"]

    print("\n--- Decision inputs ---")
    print(f"  1. Beats honest V3 (LogLoss): {deltas['log_loss'] > 0} "
          f"({deltas['log_loss']:+.8f})")
    print(f"  2. >=2 secondary improve:     {n_sec}/3 {sec}")
    print(f"  3. Improves in >=2 folds:     {folds_improved}/3")
    print(f"  4. Not one league only:       {lg_imp}/5")
    print(f"  5. Beats market:              {beats_market}")
    print(f"     No leakage:                {no_leak}")
    print(f"     Lambda safety:             {safe}")
    print(f"     Determinism:               {determinism}")
    print(f"     Protected files:           {md5_ok}")

    hard = no_leak and safe and determinism and md5_ok
    beats_v3 = (deltas["log_loss"] > 0 and n_sec >= 2
                and folds_improved >= 2 and lg_imp >= 3)

    if not hard:
        decision = "FAIL"
        why = "A mandatory safety/integrity gate failed."
        champion = False
    elif beats_v3 and beats_market:
        decision = "PASS"
        champion = True
        why = (f"LightGBM beat honest V3 by {deltas['log_loss']:+.6f}, "
               f"{n_sec}/3 secondary metrics, {folds_improved}/3 folds, "
               f"{lg_imp}/5 leagues, AND beat market alone. "
               "Champion candidate for E12.")
    elif beats_v3:
        decision = "PASS"
        champion = False
        why = (f"LightGBM improved on honest V3 by {deltas['log_loss']:+.6f} "
               f"({n_sec}/3 secondary, {folds_improved}/3 folds, "
               f"{lg_imp}/5 leagues) but did NOT beat market alone. "
               "Improved model over V3; market remains stronger.")
    elif deltas["log_loss"] <= 0:
        decision = "FAIL"
        champion = False
        why = (f"LightGBM did not beat honest V3 on the primary metric "
               f"({deltas['log_loss']:+.8f}). V3's linear Poisson structure "
               "is sufficient for this feature representation.")
    else:
        decision = "INCONCLUSIVE"
        champion = False
        why = (f"LightGBM improved log loss by {deltas['log_loss']:+.6f} but "
               f"only {n_sec}/3 secondary metrics, {folds_improved}/3 folds "
               f"and {lg_imp}/5 leagues improved. Evidence unstable.")

    print(f"\n  E9 DECISION: {decision}")
    print(f"  CHAMPION CANDIDATE: {champion}")
    print(f"  {why}")
    post = check_integrity("POST")

    RESULTS_JSON.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E9 — LightGBM Poisson vs V3 linear Poisson",
        "environment": {"lightgbm": lgb_v, "sklearn": sk_v,
                        "python": sys.version.split()[0]},
        "missing_data_policy": ("median imputation for BOTH arms, medians "
                                "from training-fold rows only; native NaN "
                                "handling deliberately NOT used"),
        "dataset": {"n_fixtures": int(len(df)),
                    "primary_oos_n": int(len(oos)),
                    "market_overlap_n": int(mk_mask.sum()) if mk_mask.sum() else 0,
                    "leagues": sorted(df.competition_name.unique().tolist()),
                    "seasons": sorted(df.season.unique().tolist()),
                    "quarantined_season": QUARANTINED},
        "feature_contract": {"n": len(V3COLS), "columns": list(V3COLS)},
        "hyperparameter_grid": grid,
        "folds": fold_out,
        "primary_universe": {"n": int(len(oos)), "arm_a_v3": mA,
                             "arm_b_lgbm": mB,
                             "deltas_b_minus_a": deltas,
                             "honest_v3_reference": V3_REF_5331},
        "market_overlap_universe": market_block,
        "by_league": by_league, "by_season": by_season,
        "prediction_change_pooled": pooled_change,
        "lambda_diagnostics_pooled": diag_pooled,
        "feature_importance": importance_by_fold,
        "config_stability": {"selected": cfg_ids, "stable": cfg_stable},
        "calibration": {"arm_a_v3": reliability_bins(yo, PA),
                        "arm_b_lgbm": reliability_bins(yo, PB)},
        "determinism_verified": determinism,
        "leakage_checks": leak,
        "decision": decision,
        "champion_candidate": champion,
        "rationale": why,
        "integrity_pre": pre, "integrity_post": post,
    }, indent=2, default=str))
    print(f"\n  Results -> {RESULTS_JSON}")
    print("\n" + "=" * 78 + f"\nE9: {decision}   (champion candidate: {champion})\n"
          + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
