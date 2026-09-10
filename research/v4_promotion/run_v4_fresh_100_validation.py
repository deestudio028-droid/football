"""V4 fresh-100 OOS validation — V2 vs V3 vs V4 vs Pinnacle market.

Usage:
    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python research/v4_promotion/run_v4_fresh_100_validation.py

Run test_v4_fresh_100_validation.py FIRST — it must report 0 failures.

STRICTLY OBSERVATIONAL. This script loads existing artifacts, never
retrains or tunes, never modifies production, and never promotes.

The 100 fixtures are read from the frozen fresh_100_fixture_ids.json.
This script contains NO fixture-selection logic, so it is structurally
incapable of reordering, filtering or cherry-picking the sample.

Market is REFERENCE ONLY and never enters any feature matrix.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.elo import ELO_COLUMNS, load_elo_features            # noqa: E402
from features.online_attack_defense import (AD_COLUMNS,            # noqa: E402
                                            compute_ad_states,
                                            fit_baseline_rates)
from models.config import FINAL_TRAIN_SEASONS, SEASON_NAME_TO_IDS  # noqa: E402
from models.data import load_supervised_dataset                    # noqa: E402
from models.poisson import predict_poisson                         # noqa: E402
from models.v2_artifact import load as load_v2                     # noqa: E402
from models.v3_artifact import load_v3_artifact                    # noqa: E402
from models.v4_artifact import load_v4_artifact                    # noqa: E402

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
FRESH_MARKET_DB = HERE / "fresh_100_market_odds.sqlite"
FROZEN_IDS = HERE / "fresh_100_fixture_ids.json"
RESULTS_20 = HERE / "v4_20_validation_results.json"
RESULTS_50 = HERE / "v4_50_validation_results.json"

V2_ARTIFACT = PROJECT_ROOT / "data/models/v2_poisson_venue.pkl"
V3_ARTIFACT = PROJECT_ROOT / "data/models/v3_poisson_venue_elo_candidate.pkl"
V4_ARTIFACT = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"

RESULTS_JSON = HERE / "v4_fresh_100_validation_results.json"
REPORT_MD = HERE / "v4_fresh_100_validation_report.md"
MANIFEST_JSON = HERE / "v4_fresh_100_validation_manifest.json"

PINNED = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
    "research/v4_promotion/fresh_100_market_odds.sqlite": "2cb80b79d772fbedd4f3707b39a32c13",
    "research/v4_promotion/fresh_100_fixture_ids.json": "761ad5cc571643e6985e671bd9c3d83a",
}
TARGET_COMPS = (200, 419, 423, 477, 499)
CLASS_ORDER = ("H", "D", "A")
E6_LR = 0.02
MATERIAL = 0.01            # same threshold as the 50-match harness
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260820
CUTPOINTS = [1, 21, 41, 61, 81, 100]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def stop(msg: str):
    print(f"\n{'=' * 78}\nSTOP\n{'=' * 78}\n  {msg}")
    raise SystemExit(1)


def audit(label: str) -> dict:
    print(f"\n--- {label} ---")
    out = {}
    for rel, exp in PINNED.items():
        p = PROJECT_ROOT / rel
        if not p.exists(): stop(f"missing: {rel}")
        a = md5(p)
        out[rel] = {"expected": exp, "actual": a,
                    "status": "identical" if a == exp else "CHANGED"}
        print(f"  [{'OK  ' if a == exp else 'FAIL'}] {Path(rel).name:<40} {a}")
        if a != exp: stop(f"protected file changed: {rel}")
    return out


# --- metrics ------------------------------------------------------------

def _oh(y):
    o = np.zeros((len(y), 3))
    for i, c in enumerate(CLASS_ORDER): o[:, i] = (y == c)
    return o


def log_loss(y, P):
    Pc = np.clip(P, 1e-15, 1.0); Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(_oh(y) * np.log(Pc), axis=1)))


def per_fixture_ll(y, P):
    Pc = np.clip(P, 1e-15, 1.0); Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return -np.sum(_oh(y) * np.log(Pc), axis=1)


def brier(y, P): return float(np.mean(np.sum((P - _oh(y)) ** 2, axis=1)))


def rps(y, P):
    cp, co = np.cumsum(P, axis=1), np.cumsum(_oh(y), axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def preds_of(P): return np.array([CLASS_ORDER[i] for i in P.argmax(1)])


def draw_recall(y, P):
    m = (y == "D")
    return float(np.mean(preds_of(P)[m] == "D")) if m.sum() else float("nan")


def reliability(y, P, nb=10):
    oh, pf = _oh(y).ravel(), np.asarray(P, float).ravel()
    e = np.linspace(0, 1, nb + 1); out = []
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
    b = reliability(y, P, nb); t = sum(x["n"] for x in b)
    return float(sum(x["n"] / t * abs(x["gap"]) for x in b if x["n"])) if t else float("nan")


def metrics(y, P):
    return {"n": int(len(y)), "correct": int(np.sum(preds_of(P) == y)),
            "accuracy": round(float(np.mean(preds_of(P) == y)), 6),
            "log_loss": round(log_loss(y, P), 6),
            "brier": round(brier(y, P), 6), "rps": round(rps(y, P), 6),
            "draw_recall": round(draw_recall(y, P), 6),
            "ece": round(ece(y, P), 6)}


def bootstrap(y, Pa, Pb, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    d = per_fixture_ll(y, Pa) - per_fixture_ll(y, Pb)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"mean_delta": round(float(d.mean()), 6),
            "ci_lower_2.5": round(lo, 6), "ci_upper_97.5": round(hi, 6),
            "pct_favouring_first": round(100 * float(np.mean(boot < 0)), 2),
            "ci_crosses_zero": bool(lo < 0 < hi),
            "n_resamples": n, "seed": seed,
            "interpretation": "negative = first model better on log loss"}


def main() -> int:
    print("=" * 78)
    print("V4 FRESH-100 OOS VALIDATION")
    print("V2 vs V3 vs V4 vs Pinnacle market (reference only)")
    print("=" * 78)

    pre = audit("PHASE 1 — protected integrity (PRE)")
    frozen_pre = {f: md5(HERE / f) for f in
                  ("v4_20_validation_results.json",
                   "v4_20_validation_manifest.json",
                   "v4_50_validation_results.json",
                   "v4_50_validation_manifest.json") if (HERE / f).exists()}
    old_mkt = HERE / "promotion_market_odds.sqlite"
    old_mkt_pre = md5(old_mkt) if old_mkt.exists() else None

    # ---- PHASE 2 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 2 — FROZEN 100 FIXTURES (read, never regenerated)")
    print("=" * 78)
    ids = [int(x) for x in json.loads(
        FROZEN_IDS.read_text(encoding="utf-8"))["fixture_ids"]]
    if len(ids) != 100 or len(set(ids)) != 100:
        stop("frozen list is not 100 unique ids")

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * 100)
    df = pd.read_sql_query(
        f"""SELECT fixture_id, date, competition_name, home_name, away_name,
                   home_goals, away_goals, status, unix
            FROM fixtures WHERE fixture_id IN ({ph})""", conn, params=ids)
    sid = SEASON_NAME_TO_IDS["2025/2026"]
    qs = ",".join("?" * len(sid))
    old50 = [r[0] for r in conn.execute(
        f"""SELECT fixture_id FROM fixtures WHERE season_id IN ({qs})
            AND status='FT' ORDER BY unix ASC, fixture_id ASC LIMIT 50""",
        list(sid)).fetchall()]
    conn.close()
    df = df.set_index("fixture_id").loc[ids].reset_index()

    if len(df) != 100: stop("not all frozen ids exist in matches.db")
    if not (df.status == "FT").all(): stop("non-FT fixture present")
    if df.home_goals.isna().any(): stop("missing outcome label")
    u = df.unix.to_numpy()
    if not np.all(u[:-1] <= u[1:]): stop("frozen list not chronological")
    if set(ids) & set(old50): stop("overlap with the frozen 50")
    if set(ids) & set(old50[:20]): stop("overlap with the frozen 20")

    y = np.where(df.home_goals > df.away_goals, "H",
                 np.where(df.home_goals == df.away_goals, "D", "A"))
    comp = dict(Counter(df.competition_name))
    print(f"  100 fixtures, 0 duplicates, all FT, chronological")
    print(f"  overlap with frozen 50: 0   with frozen 20: 0")
    print(f"  composition: " + " · ".join(
        f"{k} {v}" for k, v in sorted(comp.items(), key=lambda kv: -kv[1])))
    print(f"  date span : {df.date.iloc[0][:10]} .. {df.date.iloc[-1][:10]}")
    print(f"  outcomes  : H={int((y=='H').sum())} D={int((y=='D').sum())} "
          f"A={int((y=='A').sum())}")
    print(f"\n  {'#':>4} {'fixture_id':>11} {'date':<11} {'league':<15} "
          f"{'home':<20} {'away':<20} {'res':>5}")
    print("  " + "-" * 94)
    for i, r in df.iterrows():
        print(f"  {i+1:>4} {r.fixture_id:>11} {str(r.date)[:10]:<11} "
              f"{r.competition_name[:14]:<15} {r.home_name[:19]:<20} "
              f"{r.away_name[:19]:<20} "
              f"{int(r.home_goals)}-{int(r.away_goals)} {y[i]}")

    # ---- PHASE 3 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 3 — FRESH MARKET REFERENCE")
    print("=" * 78)
    conn = sqlite3.connect(f"file:{FRESH_MARKET_DB}?mode=ro", uri=True)
    mk = pd.read_sql_query(
        """SELECT fixture_id, p_home, p_draw, p_away, bookmaker_name,
                  price_class, data_class, market_key FROM fresh_100_market""",
        conn)
    mcols = [r[1] for r in conn.execute("PRAGMA table_info(fresh_100_market)")]
    conn.close()
    if len(mk) != 100: stop(f"market rows = {len(mk)}")
    if set(mk.fixture_id) != set(ids): stop("market fixture mismatch")
    if set(mk.bookmaker_name) != {"Pinnacle"}: stop("market not Pinnacle-only")
    if set(mk.price_class) != {"closing"}: stop("market not closing-only")
    if set(mk.market_key) != {"ft_result"}: stop("market not 1X2")
    if set(mk.data_class) != {"promotion-validation-only"}:
        stop("market data_class wrong")
    if {"home_goals", "away_goals", "label_result"} & set(mcols):
        stop("market table has an outcome column")
    mk = mk.set_index("fixture_id").loc[ids].reset_index()
    PM = mk[["p_home", "p_draw", "p_away"]].to_numpy(float)
    if not np.allclose(PM.sum(axis=1), 1.0, atol=1e-9):
        stop("market probabilities do not sum to 1")
    print(f"  market coverage = 100/100")
    print(f"  Pinnacle · closing · 1X2 · promotion-validation-only")
    print(f"  no outcome columns ({len(mcols)} columns)")
    print(f"  REFERENCE ONLY — never enters any feature matrix")

    # ---- PHASE 4 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 4 — ARTIFACT VERIFICATION (no retraining)")
    print("=" * 78)
    from models.v3_contract import V3_FEATURE_COLUMNS
    v2, v3 = load_v2(V2_ARTIFACT), load_v3_artifact(V3_ARTIFACT)
    v4 = load_v4_artifact(V4_ARTIFACT)
    if v3.n_features != 87: stop("V3 not 87 features")
    if tuple(v3.feature_columns) != tuple(V3_FEATURE_COLUMNS):
        stop("V3 contract mismatch")
    if v4.n_features != 91: stop("V4 not 91 features")
    if tuple(v4.feature_columns[:87]) != tuple(V3_FEATURE_COLUMNS):
        stop("V4[:87] != V3 contract")
    if tuple(v4.feature_columns[87:]) != ("A_home", "D_home", "A_away", "D_away"):
        stop("V4[87:] != E6 columns")
    if "2025/2026" in v4.training_seasons: stop("V4 trained on 2025/26")
    if v4.holdout_used_in_training is not False: stop("V4 holdout flag wrong")
    print(f"  V2: {v2.model_version}, {len(v2.feature_columns)} features")
    print(f"  V3: {v3.model_version}, 87 features, contract verified")
    print(f"  V4: {v4.model_version}, 91 features, contract verified")
    print(f"  V4 training seasons: {list(v4.training_seasons)}")
    print(f"  V4 holdout_used_in_training: {v4.holdout_used_in_training}")

    # ---- PHASE 5 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 5 — CAUSAL FEATURE GENERATION")
    print("=" * 78)
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])
    if X[list(ELO_COLUMNS)].isna().any().any(): stop("NaN in Elo")

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    phc = ",".join("?" * len(TARGET_COMPS))
    fx = pd.read_sql_query(
        f"""SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                   status, season, season_id, competition_id
            FROM fixtures WHERE competition_id IN ({phc})""",
        conn, params=list(TARGET_COMPS))
    conn.close()
    w = set()
    for s in FINAL_TRAIN_SEASONS: w |= set(SEASON_NAME_TO_IDS[s])
    hist = fx[fx.season_id.isin(w) & fx.home_goals.notna()
              & fx.status.isin(["FT", "AWARDED"])]
    if "2025/2026" in set(hist.season): stop("2025/26 in A/D baseline")
    base = fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    states = compute_ad_states(fx, E6_LR, base).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(states[c])
    if X[list(AD_COLUMNS)].isna().any().any(): stop("NaN in A/D")

    idx = [int(np.where(meta.fixture_id == f)[0][0]) for f in ids]
    X100 = X.iloc[idx].reset_index(drop=True)
    print(f"  Elo 100/100 · A/D 100/100")
    print(f"  E6 baseline: mu_home={base.mu_home:.6f} mu_away={base.mu_away:.6f} "
          f"n={base.n_train} (pre-2025/26 only)")
    print(f"  design matrix: {X100.shape}")

    # ---- PHASE 6 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 6 — MULTI-CUTOFF CAUSAL ADVERSARIAL")
    print("=" * 78)
    unix_map = {int(r.fixture_id): int(r.unix) for r in fx.itertuples()}
    cols = list(AD_COLUMNS)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    causal_ok, worst = True, 0.0
    for k in CUTPOINTS:
        tgt = ids[k - 1]; cut = unix_map[tgt]
        earlier = [i for i in ids if unix_map[i] < cut] + [tgt]
        adv = fx.copy()
        m = (adv.season == "2025/2026") & (adv.unix >= cut)
        adv.loc[m, "home_goals"] = rng.integers(0, 9, int(m.sum())).astype(float)
        adv.loc[m, "away_goals"] = rng.integers(0, 9, int(m.sum())).astype(float)
        a = compute_ad_states(adv, E6_LR, base).set_index("fixture_id")
        d = float(np.max(np.abs(states.loc[earlier, cols].to_numpy()
                                - a.loc[earlier, cols].to_numpy())))
        worst = max(worst, d); causal_ok &= (d == 0.0)
        print(f"  #{k:>3} {tgt}: rewrote {int(m.sum()):>4} at/after -> "
              f"{len(earlier)} protected fixtures, max diff {d:.3e}")
    if not causal_ok: stop("causality violated")
    print(f"  CAUSALITY: PASS (max diff {worst:.3e})")

    # ---- PHASE 7 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 7 — MODEL PREDICTIONS")
    print("=" * 78)

    def run(art, name):
        E = art.preprocessor.transform(X100[list(art.feature_columns)])
        lh, la = art.model_home_goals.predict(E), art.model_away_goals.predict(E)
        if not (np.all(np.isfinite(lh)) and np.all(np.isfinite(la))):
            stop(f"{name}: non-finite lambdas")
        if not (np.all(lh > 0) and np.all(la > 0)):
            stop(f"{name}: non-positive lambdas")
        pr = predict_poisson(lh, la, list(art.class_order))
        P = np.array([[p.probabilities["H"], p.probabilities["D"],
                       p.probabilities["A"]] for p in pr])
        if not np.all(np.isfinite(P)): stop(f"{name}: non-finite probabilities")
        if np.any(P < 0) or np.any(P > 1): stop(f"{name}: probs out of range")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-9):
            stop(f"{name}: rows do not sum to 1")
        if P.shape != (100, 3): stop(f"{name}: wrong shape {P.shape}")
        print(f"  {name}: 100 predictions, mean lambda "
              f"home={lh.mean():.3f} away={la.mean():.3f}")
        return P, lh, la

    P2, lh2, la2 = run(v2, "V2")
    P3, lh3, la3 = run(v3, "V3")
    P4, lh4, la4 = run(v4, "V4")

    # ---- PHASE 8 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 8 — PREDICTION TABLE")
    print("=" * 78)
    p2, p3, p4, pm = preds_of(P2), preds_of(P3), preds_of(P4), preds_of(PM)
    print(f"  {'#':>4} {'league':<15} {'match':<40} {'act':>4}"
          f"{'V2':>10}{'V3':>10}{'V4':>10}{'MKT':>10}")
    print("  " + "-" * 105)
    rows = []
    for i in range(100):
        r = df.iloc[i]
        match = f"{r.home_name[:17]} v {r.away_name[:17]}"
        flag = " <" if len({p2[i], p3[i], p4[i], pm[i]}) > 1 else ""
        print(f"  {i+1:>4} {r.competition_name[:14]:<15} {match:<40} {y[i]:>4}"
              f"{p2[i]:>6}({P2[i].max():.2f}){p3[i]:>4}({P3[i].max():.2f})"
              f"{p4[i]:>4}({P4[i].max():.2f}){pm[i]:>4}({PM[i].max():.2f}){flag}")
        rows.append({
            "n": i + 1, "fixture_id": int(r.fixture_id), "date": str(r.date)[:10],
            "league": r.competition_name, "home": r.home_name,
            "away": r.away_name,
            "score": f"{int(r.home_goals)}-{int(r.away_goals)}", "actual": y[i],
            **{k: {"p_home": round(float(P[i][0]), 6),
                   "p_draw": round(float(P[i][1]), 6),
                   "p_away": round(float(P[i][2]), 6),
                   "pred": pp[i], "confidence": round(float(P[i].max()), 6),
                   "correct": bool(pp[i] == y[i])}
               for k, P, pp in (("V2", P2, p2), ("V3", P3, p3),
                                ("V4", P4, p4), ("MARKET", PM, pm))},
            "V2_lambda": [round(float(lh2[i]), 6), round(float(la2[i]), 6)],
            "V3_lambda": [round(float(lh3[i]), 6), round(float(la3[i]), 6)],
            "V4_lambda": [round(float(lh4[i]), 6), round(float(la4[i]), 6)],
        })

    # ---- PHASE 9 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 9 — 100-MATCH SCORECARD")
    print("=" * 78)
    M = {"V2": metrics(y, P2), "V3": metrics(y, P3),
         "V4": metrics(y, P4), "MARKET": metrics(y, PM)}
    print("  direction: ACC up · LOGLOSS down · BRIER down · RPS down · "
          "DRAW_RECALL up · ECE down")
    print(f"\n  {'MODEL':<8}{'CORRECT':>10}{'ACC':>9}{'LOGLOSS':>10}{'BRIER':>9}"
          f"{'RPS':>9}{'DRAW_REC':>10}{'ECE':>9}")
    print("  " + "-" * 74)
    for k in ("V2", "V3", "V4", "MARKET"):
        m = M[k]
        print(f"  {k:<8}{m['correct']:>7}/100{m['accuracy']:>9.4f}"
              f"{m['log_loss']:>10.6f}{m['brier']:>9.6f}{m['rps']:>9.6f}"
              f"{m['draw_recall']:>10.4f}{m['ece']:>9.6f}")

    # ---- PHASE 10 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 10 — PAIRWISE COMPARISONS (positive = V4 better)")
    print("=" * 78)
    pair = {}
    for o, Po, po in (("V2", P2, p2), ("V3", P3, p3), ("MARKET", PM, pm)):
        mo, m4 = M[o], M["V4"]
        pair[f"V4_vs_{o}"] = {
            "log_loss_delta": round(mo["log_loss"] - m4["log_loss"], 6),
            "brier_delta": round(mo["brier"] - m4["brier"], 6),
            "rps_delta": round(mo["rps"] - m4["rps"], 6),
            "ece_delta": round(mo["ece"] - m4["ece"], 6),
            "accuracy_delta": round(m4["accuracy"] - mo["accuracy"], 6),
            "correct_delta": m4["correct"] - mo["correct"],
            "top_class_changes": int(np.sum(po != p4))}
        d = pair[f"V4_vs_{o}"]
        print(f"\n  V4 vs {o}:")
        print(f"    LogLoss {d['log_loss_delta']:+.6f}   "
              f"Brier {d['brier_delta']:+.6f}   RPS {d['rps_delta']:+.6f}")
        print(f"    ECE {d['ece_delta']:+.6f}   "
              f"Accuracy {d['accuracy_delta']:+.4f}   "
              f"Correct {d['correct_delta']:+d}")
        print(f"    top-class differences: {d['top_class_changes']}/100")

    # ---- PHASE 11 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 11 — CHRONOLOGICAL BUCKETS")
    print("=" * 78)
    buckets = {"1-20": slice(0, 20), "21-40": slice(20, 40),
               "41-60": slice(40, 60), "61-80": slice(60, 80),
               "81-100": slice(80, 100)}
    bres = {}
    print(f"  {'bucket':<9}{'n':>4}" + "".join(
        f"{k:>18}" for k in ("V2", "V3", "V4", "MARKET")))
    print("  " + "-" * 85)
    for nm, sl in buckets.items():
        yy = y[sl]
        e = {k: metrics(yy, P[sl]) for k, P in
             (("V2", P2), ("V3", P3), ("V4", P4), ("MARKET", PM))}
        bres[nm] = e
        print(f"  {nm:<9}{len(yy):>4}" + "".join(
            f"{e[k]['correct']:>6}/{len(yy):<3}{e[k]['log_loss']:>8.4f}"
            for k in ("V2", "V3", "V4", "MARKET")))

    # ---- PHASE 12 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 12 — LEAGUE BREAKDOWN (DESCRIPTIVE ONLY)")
    print("=" * 78)
    lg = {}
    print(f"  {'league':<17}{'n':>4}{'V2acc':>8}{'V3acc':>8}{'V4acc':>8}"
          f"{'MKTacc':>8}{'V4 LL':>10}{'MKT LL':>10}")
    print("  " + "-" * 73)
    for name in sorted(comp, key=lambda k: -comp[k]):
        m = (df.competition_name == name).to_numpy()
        e = {k: metrics(y[m], P[m]) for k, P in
             (("V2", P2), ("V3", P3), ("V4", P4), ("MARKET", PM))}
        lg[name] = {"n": int(m.sum()), **e}
        print(f"  {name:<17}{int(m.sum()):>4}{e['V2']['accuracy']:>8.3f}"
              f"{e['V3']['accuracy']:>8.3f}{e['V4']['accuracy']:>8.3f}"
              f"{e['MARKET']['accuracy']:>8.3f}{e['V4']['log_loss']:>10.5f}"
              f"{e['MARKET']['log_loss']:>10.5f}")
    print(f"\n  All five leagues have n>={min(comp.values())} — better balanced")
    print("  than the 50-match sample, but still DESCRIPTIVE only.")

    # ---- PHASE 13 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 13 — 20 vs 50 vs FRESH 100")
    print("=" * 78)
    hist_cmp = {"fresh_100": {k: M[k] for k in ("V2", "V3", "V4", "MARKET")}}
    for tag, path in (("twenty", RESULTS_20), ("fifty", RESULTS_50)):
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            sc = d.get("scorecard") or d.get("metrics") or d.get("summary") or {}
            hist_cmp[tag] = sc
    print(f"  {'model':<8}{'20 acc':>9}{'20 LL':>10}{'50 acc':>9}{'50 LL':>10}"
          f"{'100 acc':>10}{'100 LL':>10}")
    print("  " + "-" * 66)
    for k in ("V2", "V3", "V4", "MARKET"):
        t = hist_cmp.get("twenty", {}).get(k, {})
        f_ = hist_cmp.get("fifty", {}).get(k, {})
        h = M[k]
        def g(d, key):
            v = d.get(key) if isinstance(d, dict) else None
            return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
        print(f"  {k:<8}{g(t,'accuracy'):>9}{g(t,'log_loss'):>10}"
              f"{g(f_,'accuracy'):>9}{g(f_,'log_loss'):>10}"
              f"{h['accuracy']:>10.4f}{h['log_loss']:>10.6f}")

    # ---- PHASE 14 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 14 — V4 vs V3 PROBABILITY STABILITY")
    print("=" * 78)
    dP = np.abs(P4 - P3)
    change = {"mean_abs_prob_change": round(float(dP.mean()), 8),
              "max_prob_change": round(float(dP.max()), 8),
              "pct_materially_changed": round(
                  100 * float(np.mean(dP.max(axis=1) > MATERIAL)), 4),
              "top_class_changes": int(np.sum(p3 != p4)),
              "mean_abs_lambda_home_change": round(
                  float(np.mean(np.abs(lh4 - lh3))), 8),
              "mean_abs_lambda_away_change": round(
                  float(np.mean(np.abs(la4 - la3))), 8),
              "material_threshold": MATERIAL}
    for k, v in change.items(): print(f"  {k}: {v}")

    # ---- PHASE 15 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 15 — PAIRED BOOTSTRAP (log loss)")
    print("=" * 78)
    boot = {"V4_minus_V2": bootstrap(y, P4, P2),
            "V4_minus_V3": bootstrap(y, P4, P3),
            "V4_minus_MARKET": bootstrap(y, P4, PM)}
    for k, v in boot.items():
        verdict = ("NOT statistically distinguishable" if v["ci_crosses_zero"]
                   else ("V4 better" if v["mean_delta"] < 0 else "V4 worse"))
        print(f"  {k}: {v['mean_delta']:+.6f}  95% CI "
              f"[{v['ci_lower_2.5']:+.6f}, {v['ci_upper_97.5']:+.6f}]  "
              f"favouring V4 {v['pct_favouring_first']:.1f}%  -> {verdict}")
    print("  (negative = V4 better; CI crossing 0 = not distinguishable)")

    # ---- PHASE 17 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 17 — DETERMINISM (full re-run)")
    print("=" * 78)
    P2b, lh2b, la2b = run(v2, "V2")
    P3b, lh3b, la3b = run(v3, "V3")
    P4b, lh4b, la4b = run(v4, "V4")
    conn = sqlite3.connect(f"file:{FRESH_MARKET_DB}?mode=ro", uri=True)
    PMb = pd.read_sql_query(
        "SELECT fixture_id,p_home,p_draw,p_away FROM fresh_100_market",
        conn).set_index("fixture_id").loc[ids][
            ["p_home", "p_draw", "p_away"]].to_numpy(float)
    conn.close()
    dd = {"V2": float(np.max(np.abs(P2 - P2b))),
          "V3": float(np.max(np.abs(P3 - P3b))),
          "V4": float(np.max(np.abs(P4 - P4b))),
          "MARKET": float(np.max(np.abs(PM - PMb))),
          "V4_lambda_home": float(np.max(np.abs(lh4 - lh4b))),
          "V4_lambda_away": float(np.max(np.abs(la4 - la4b)))}
    determinism = all(v == 0.0 for v in dd.values())
    for k, v in dd.items(): print(f"  {k} max abs diff: {v:.3e}")
    print(f"  DETERMINISM: {'PASS' if determinism else 'FAIL'}")

    # ---- PHASE 16/19 -----------------------------------------------------
    post = audit("PHASE 19 — protected integrity (POST)")
    frozen_post = {f: md5(HERE / f) for f in frozen_pre}
    frozen_ok = frozen_pre == frozen_post
    old_mkt_ok = (md5(old_mkt) == old_mkt_pre) if old_mkt.exists() else True
    print(f"\n  frozen 20/50 artifacts unchanged: {frozen_ok}")
    print(f"  50-match market DB unchanged     : {old_mkt_ok}")
    if not frozen_ok: stop("frozen 20/50 artifacts modified")
    if not old_mkt_ok: stop("50-match market DB modified")

    integrity_ok = all(v["status"] == "identical" for v in post.values())
    market_ok = True

    # ---- write outputs ---------------------------------------------------
    now = datetime.now(timezone.utc).isoformat()
    results = {
        "generated_at": now,
        "experiment": "V4 fresh-100 OOS validation",
        "sample_note": ("Fresh 100 = next chronological FT fixtures after the "
                        "frozen 50 (LIMIT 100 OFFSET 50). Zero overlap with "
                        "the frozen 20 or 50. IDs frozen before evaluation."),
        "selection_rule": ("season_id IN (2025/26) AND status='FT' "
                           "ORDER BY unix ASC, fixture_id ASC LIMIT 100 OFFSET 50"),
        "frozen_ids_md5": md5(FROZEN_IDS),
        "fixture_ids": [int(f) for f in ids],
        "composition": comp,
        "date_span": [df.date.iloc[0][:10], df.date.iloc[-1][:10]],
        "outcome_distribution": {"H": int((y == "H").sum()),
                                 "D": int((y == "D").sum()),
                                 "A": int((y == "A").sum())},
        "artifacts": {"V2": md5(V2_ARTIFACT), "V3": md5(V3_ARTIFACT),
                      "V4": md5(V4_ARTIFACT),
                      "V4_training_seasons": list(v4.training_seasons),
                      "V4_holdout_used": False},
        "market": {"source": FRESH_MARKET_DB.name, "md5": md5(FRESH_MARKET_DB),
                   "bookmaker": "Pinnacle", "price": "closing",
                   "coverage": "100/100", "role": "REFERENCE ONLY",
                   "in_feature_matrix": False},
        "per_match": rows, "scorecard": M, "pairwise": pair,
        "buckets": bres, "by_league": lg,
        "history_20_50_100": hist_cmp,
        "v4_vs_v3_change": change, "bootstrap": boot,
        "reliability": {k: reliability(y, P) for k, P in
                        (("V2", P2), ("V3", P3), ("V4", P4), ("MARKET", PM))},
        "causality": {"cutpoints": CUTPOINTS, "max_diff": worst,
                      "passed": causal_ok},
        "determinism": {**dd, "passed": determinism},
        "integrity": {"pre": pre, "post": post, "passed": integrity_ok,
                      "frozen_20_50_unchanged": frozen_ok,
                      "old_market_db_unchanged": old_mkt_ok},
        "promotion": {"promoted": False,
                      "note": "Strictly observational. V4 is NOT promoted."},
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))

    overall = (integrity_ok and frozen_ok and old_mkt_ok
               and causal_ok and determinism and market_ok)

    MANIFEST_JSON.write_text(json.dumps({
        "generated_at": now, "fixtures": 100,
        "results_file": RESULTS_JSON.name, "report_file": REPORT_MD.name,
        "frozen_ids_md5": md5(FROZEN_IDS),
        "market_db_md5": md5(FRESH_MARKET_DB),
        "v4_artifact_md5": md5(V4_ARTIFACT),
        "integrity": integrity_ok, "causality": causal_ok,
        "determinism": determinism, "market_coverage": "100/100",
        "overall": "COMPLETE" if overall else "BLOCKED",
        "v4_promoted": False,
    }, indent=2))

    # ---- report ----------------------------------------------------------
    def row(k):
        m = M[k]
        return (f"| {k} | {m['correct']}/100 | {m['accuracy']:.4f} | "
                f"{m['log_loss']:.6f} | {m['brier']:.6f} | {m['rps']:.6f} | "
                f"{m['draw_recall']:.4f} | {m['ece']:.6f} |")

    lines = [
        "# V4 Fresh-100 OOS Validation Report", "",
        f"**Date:** {now[:10]}",
        "**Type:** Strictly observational out-of-sample validation. "
        "**V4 is NOT promoted.**", "",
        "## Sample", "",
        f"- Selection: `LIMIT 100 OFFSET 50` on the 2025/26 FT fixtures, "
        f"chronological",
        f"- Frozen before evaluation, MD5 `{md5(FROZEN_IDS)}`",
        f"- **Zero overlap** with the frozen 20 or frozen 50",
        f"- Date span: {df.date.iloc[0][:10]} to {df.date.iloc[-1][:10]}",
        f"- Composition: " + ", ".join(
            f"{k} {v}" for k, v in sorted(comp.items(), key=lambda kv: -kv[1])),
        f"- Outcomes: H={int((y=='H').sum())} D={int((y=='D').sum())} "
        f"A={int((y=='A').sum())}", "",
        "## Scorecard", "",
        "Direction: Accuracy ↑ · LogLoss ↓ · Brier ↓ · RPS ↓ · "
        "Draw Recall ↑ · ECE ↓", "",
        "| Model | Correct | Accuracy | LogLoss | Brier | RPS | Draw Recall | ECE |",
        "|---|---|---|---|---|---|---|---|",
        row("V2"), row("V3"), row("V4"), row("MARKET"), "",
        "## V4 Pairwise (positive = V4 better)", "",
        "| Comparison | LogLoss | Brier | RPS | ECE | Accuracy | Correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in ("V2", "V3", "MARKET"):
        d = pair[f"V4_vs_{o}"]
        lines.append(
            f"| V4 vs {o} | {d['log_loss_delta']:+.6f} | "
            f"{d['brier_delta']:+.6f} | {d['rps_delta']:+.6f} | "
            f"{d['ece_delta']:+.6f} | {d['accuracy_delta']:+.4f} | "
            f"{d['correct_delta']:+d} |")
    lines += ["", "## Chronological Buckets", "",
              "| Bucket | n | V2 | V3 | V4 | Market |", "|---|---|---|---|---|---|"]
    for nm, e in bres.items():
        lines.append(f"| {nm} | {e['V2']['n']} | " + " | ".join(
            f"{e[k]['correct']}/{e[k]['n']} ({e[k]['log_loss']:.4f})"
            for k in ("V2", "V3", "V4", "MARKET")) + " |")
    lines += ["", "## Per-League (descriptive only)", "",
              "| League | n | V2 acc | V3 acc | V4 acc | Market acc | V4 LL | Market LL |",
              "|---|---|---|---|---|---|---|---|"]
    for name in sorted(comp, key=lambda k: -comp[k]):
        e = lg[name]
        lines.append(f"| {name} | {e['n']} | {e['V2']['accuracy']:.3f} | "
                     f"{e['V3']['accuracy']:.3f} | {e['V4']['accuracy']:.3f} | "
                     f"{e['MARKET']['accuracy']:.3f} | "
                     f"{e['V4']['log_loss']:.5f} | "
                     f"{e['MARKET']['log_loss']:.5f} |")
    lines += ["", "## Bootstrap (10,000 resamples, seed 20260820)", "",
              "| Comparison | Mean delta | 95% CI | Favouring V4 | Verdict |",
              "|---|---|---|---|---|"]
    for k, v in boot.items():
        verdict = ("not distinguishable" if v["ci_crosses_zero"]
                   else ("V4 better" if v["mean_delta"] < 0 else "V4 worse"))
        lines.append(f"| {k} | {v['mean_delta']:+.6f} | "
                     f"[{v['ci_lower_2.5']:+.6f}, {v['ci_upper_97.5']:+.6f}] | "
                     f"{v['pct_favouring_first']:.1f}% | {verdict} |")
    lines += ["", "## Gates", "",
              f"- INTEGRITY: {'PASS' if integrity_ok and frozen_ok else 'FAIL'}",
              f"- CAUSALITY: {'PASS' if causal_ok else 'FAIL'} "
              f"(max diff {worst:.3e} across cut points {CUTPOINTS})",
              f"- DETERMINISM: {'PASS' if determinism else 'FAIL'}",
              f"- MARKET COVERAGE: PASS (100/100)",
              f"- **OVERALL: {'COMPLETE' if overall else 'BLOCKED'}**", "",
              "**V4 is NOT promoted. No production file was modified.**"]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    # ---- PHASE 20 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("V4 FRESH 100 OOS VALIDATION")
    print("=" * 78)
    for k in ("V2", "V3", "V4", "MARKET"):
        print(f"\n{k}:")
        print(f"  correct  = {M[k]['correct']}/100")
        print(f"  accuracy = {M[k]['accuracy']:.4f}")
    for o in ("V2", "V3", "MARKET"):
        d = pair[f"V4_vs_{o}"]
        print(f"\nV4 vs {o}:")
        print(f"  LogLoss {d['log_loss_delta']:+.6f}  Brier {d['brier_delta']:+.6f}  "
              f"RPS {d['rps_delta']:+.6f}  ECE {d['ece_delta']:+.6f}")
        print(f"  Correct {d['correct_delta']:+d}  "
              f"Accuracy {d['accuracy_delta']:+.4f}")
    print("\n20 vs 50 vs FRESH 100 (accuracy / log loss):")
    for k in ("V2", "V3", "V4", "MARKET"):
        t = hist_cmp.get("twenty", {}).get(k, {})
        f_ = hist_cmp.get("fifty", {}).get(k, {})
        ta = t.get("accuracy"); fa = f_.get("accuracy")
        tl = t.get("log_loss"); fl = f_.get("log_loss")
        print(f"  {k:<7} 20: "
              f"{f'{ta:.4f}' if isinstance(ta,(int,float)) else 'n/a':>7}/"
              f"{f'{tl:.6f}' if isinstance(tl,(int,float)) else 'n/a':>8}   "
              f"50: {f'{fa:.4f}' if isinstance(fa,(int,float)) else 'n/a':>7}/"
              f"{f'{fl:.6f}' if isinstance(fl,(int,float)) else 'n/a':>8}   "
              f"100: {M[k]['accuracy']:.4f}/{M[k]['log_loss']:.6f}")
    print(f"\nINTEGRITY: {'PASS' if integrity_ok and frozen_ok else 'FAIL'}")
    print(f"CAUSALITY: {'PASS' if causal_ok else 'FAIL'}")
    print(f"DETERMINISM: {'PASS' if determinism else 'FAIL'}")
    print(f"MARKET COVERAGE: PASS")
    print(f"\nOVERALL OOS VALIDATION: {'COMPLETE' if overall else 'BLOCKED'}")
    print("\nV4 IS NOT PROMOTED. No production file was modified.")
    print(f"\n  results  -> {RESULTS_JSON.name}")
    print(f"  report   -> {REPORT_MD.name}")
    print(f"  manifest -> {MANIFEST_JSON.name}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
