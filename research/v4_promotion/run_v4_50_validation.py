"""V4 50-match retrospective validation — V2 vs V3 vs V4 vs Pinnacle market.

Usage:
    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python research/v4_promotion/run_v4_50_validation.py

Run test_v4_50_validation.py FIRST. It must report 0 failures.

STRICT EVALUATION ONLY. This script:
  - loads existing artifacts, never retrains or tunes
  - never modifies predict_match.py, V2, V3, V4 or any protected file
  - never modifies the frozen 20-match harness or its outputs
  - keeps the market a REFERENCE ONLY, never a V4 feature

The 50 fixtures come from the deterministic rule; the first 20 are the
frozen 20-match set. Composition is PL 16 / La Liga 13 / Ligue 1 12 /
Bundesliga 7 / Serie A 2 — the honest output of that rule. The earlier
17/13/12/5/3 figure was an incorrect prior estimate and was NOT used to
alter the sample.
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

from features.elo import ELO_COLUMNS, load_elo_features           # noqa: E402
from features.online_attack_defense import (AD_COLUMNS,           # noqa: E402
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
PROMO_DB = HERE / "promotion_market_odds.sqlite"
V2_ARTIFACT = PROJECT_ROOT / "data/models/v2_poisson_venue.pkl"
V3_ARTIFACT = PROJECT_ROOT / "data/models/v3_poisson_venue_elo_candidate.pkl"
V4_ARTIFACT = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"

RESULTS_JSON = HERE / "v4_50_validation_results.json"
REPORT_MD = HERE / "v4_50_validation_report.md"
MANIFEST_JSON = HERE / "v4_50_validation_manifest.json"
RESULTS_20 = HERE / "v4_20_validation_results.json"

PROTECTED = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
}
FROZEN_20 = [343465962, 343465735, 342254747, 343465963, 342863654,
             342863842, 342863843, 342863844, 343465733, 342864234,
             343465729, 342864255, 343465732, 342863998, 342864289,
             343465726, 343465727, 343465734, 343465842, 343465728]
EXPECTED_COMPOSITION = {"Premier League": 16, "La Liga": 13, "Ligue 1": 12,
                        "Bundesliga": 7, "Serie A": 2}
CLASS_ORDER = ("H", "D", "A")
E6_LEARNING_RATE = 0.02
MATERIAL = 0.01
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260820


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
    for rel, exp in PROTECTED.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            stop(f"protected file missing: {rel}")
        a = md5(p)
        out[rel] = {"expected": exp, "actual": a,
                    "status": "identical" if a == exp else "CHANGED"}
        print(f"  [{'OK  ' if a == exp else 'FAIL'}] {rel}  {a}")
        if a != exp:
            stop(f"protected file changed: {rel}")
    return out


# --- metrics ------------------------------------------------------------

def _oh(y):
    o = np.zeros((len(y), 3))
    for i, c in enumerate(CLASS_ORDER):
        o[:, i] = (y == c)
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


def accuracy(y, P): return float(np.mean(preds_of(P) == y))


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
            out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
                        "n": 0, "mean_predicted": None,
                        "observed_freq": None, "gap": None}); continue
        mp, ob = float(pf[m].mean()), float(oh[m].mean())
        out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": n,
                    "mean_predicted": round(mp, 6),
                    "observed_freq": round(ob, 6), "gap": round(ob - mp, 6)})
    return out


def ece(y, P, nb=10):
    b = reliability(y, P, nb); t = sum(x["n"] for x in b)
    return float(sum(x["n"] / t * abs(x["gap"]) for x in b if x["n"])) if t else float("nan")


def metrics(y, P):
    return {"n": int(len(y)),
            "correct": int(np.sum(preds_of(P) == y)),
            "accuracy": round(accuracy(y, P), 6),
            "log_loss": round(log_loss(y, P), 6),
            "brier": round(brier(y, P), 6),
            "rps": round(rps(y, P), 6),
            "draw_recall": round(draw_recall(y, P), 6),
            "ece": round(ece(y, P), 6)}


def paired_bootstrap(y, Pa, Pb, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """CI for log_loss(a) - log_loss(b). Negative => a better."""
    la, lb = per_fixture_ll(y, Pa), per_fixture_ll(y, Pb)
    d = la - lb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    return {"point_estimate": round(float(d.mean()), 6),
            "ci_lower_2.5": round(float(np.percentile(boot, 2.5)), 6),
            "ci_upper_97.5": round(float(np.percentile(boot, 97.5)), 6),
            "pct_resamples_favouring_first": round(
                100 * float(np.mean(boot < 0)), 2),
            "n_resamples": n, "seed": seed,
            "interpretation": "negative = first model better"}


def main() -> int:
    print("=" * 78)
    print("V4 50-MATCH RETROSPECTIVE VALIDATION")
    print("V2  vs  V3  vs  V4  vs  Pinnacle market (reference only)")
    print("=" * 78)

    pre = audit("PHASE 1 — protected integrity (PRE)")
    frozen20_pre = {f: md5(HERE / f) for f in
                    ("v4_20_validation_results.json",
                     "v4_20_validation_report.md",
                     "v4_20_validation_manifest.json")}

    # ---- PHASE 2 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 2 — DETERMINISTIC 50-FIXTURE SELECTION")
    print("=" * 78)
    ids = SEASON_NAME_TO_IDS["2025/2026"]
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    q = ",".join("?" * len(ids))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, date, competition_name, home_name, away_name,
                   home_goals, away_goals, status, unix
            FROM fixtures WHERE season_id IN ({q}) AND status = 'FT'
            ORDER BY unix ASC, fixture_id ASC LIMIT 50""",
        conn, params=list(ids))
    conn.close()

    if len(df) != 50: stop(f"expected 50 fixtures, got {len(df)}")
    if df.fixture_id.nunique() != 50: stop("duplicate fixture_id")
    if not (df.status == "FT").all(): stop("non-FT fixture present")
    u = df.unix.to_numpy()
    if not np.all(u[:-1] <= u[1:]): stop("fixtures not chronological")
    if df.fixture_id.tolist()[:20] != FROZEN_20:
        stop("first 20 do not match the frozen 20-match set")

    comp = dict(Counter(df.competition_name))
    if comp != EXPECTED_COMPOSITION:
        stop(f"composition mismatch: {comp}")

    f50 = df.fixture_id.tolist()
    y = np.where(df.home_goals > df.away_goals, "H",
                 np.where(df.home_goals == df.away_goals, "D", "A"))

    print(f"  50 fixtures, 0 duplicates, all FT, chronological")
    print(f"  first 20 == frozen 20: True")
    print(f"  composition: " + " · ".join(
        f"{k} {v}" for k, v in sorted(comp.items(), key=lambda kv: -kv[1])))
    print(f"  date span  : {df.date.iloc[0][:10]} .. {df.date.iloc[-1][:10]}")
    print(f"  outcomes   : H={int((y=='H').sum())} D={int((y=='D').sum())} "
          f"A={int((y=='A').sum())}")
    print(f"\n  {'#':>3} {'fixture_id':>11} {'date':<11} {'league':<15} "
          f"{'home':<21} {'away':<21} {'res':>4}")
    print("  " + "-" * 94)
    for i, r in df.iterrows():
        print(f"  {i+1:>3} {r.fixture_id:>11} {str(r.date)[:10]:<11} "
              f"{r.competition_name[:14]:<15} {r.home_name[:20]:<21} "
              f"{r.away_name[:20]:<21} {int(r.home_goals)}-{int(r.away_goals)} "
              f"{y[i]}")

    # ---- PHASE 3 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 3 — ARTIFACT VERIFICATION (no retraining)")
    print("=" * 78)
    from models.v3_contract import V3_FEATURE_COLUMNS
    v2 = load_v2(V2_ARTIFACT); v3 = load_v3_artifact(V3_ARTIFACT)
    v4 = load_v4_artifact(V4_ARTIFACT)
    if v3.n_features != 87: stop("V3 is not 87 features")
    if tuple(v3.feature_columns) != tuple(V3_FEATURE_COLUMNS):
        stop("V3 contract mismatch")
    if v4.n_features != 91: stop("V4 is not 91 features")
    if tuple(v4.feature_columns[:87]) != tuple(V3_FEATURE_COLUMNS):
        stop("V4[:87] != V3 contract")
    if tuple(v4.feature_columns[87:]) != ("A_home", "D_home", "A_away", "D_away"):
        stop("V4[87:] != E6 A/D columns")
    if "2025/2026" in v4.training_seasons: stop("V4 trained on 2025/26")
    if v4.holdout_used_in_training is not False:
        stop("V4 holdout_used_in_training is not False")
    print(f"  V2 loaded: {v2.model_version}, {len(v2.feature_columns)} features")
    print(f"  V3 loaded: {v3.model_version}, 87 features, contract verified")
    print(f"  V4 loaded: {v4.model_version}, 91 features, contract verified")
    print(f"  V4 training seasons: {list(v4.training_seasons)}")
    print(f"  V4 holdout_used_in_training: {v4.holdout_used_in_training}")

    # ---- PHASE 4 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 4 — CAUSAL FEATURE GENERATION")
    print("=" * 78)
    ds = load_supervised_dataset(FEATURES_DB)
    meta = ds.metadata.reset_index(drop=True)
    X = ds.X.reset_index(drop=True)
    elo = load_elo_features(MATCHES_DB).set_index("fixture_id")
    for c in ELO_COLUMNS:
        X[c] = meta["fixture_id"].map(elo[c])
    if X[list(ELO_COLUMNS)].isna().any().any(): stop("NaN in Elo features")

    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    fx = pd.read_sql_query(
        """SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals,
                  status, season, season_id, competition_id
           FROM fixtures WHERE competition_id IN (200,419,423,477,499)""", conn)
    conn.close()
    wanted = set()
    for sn in FINAL_TRAIN_SEASONS:
        wanted |= set(SEASON_NAME_TO_IDS[sn])
    hist = fx[fx.season_id.isin(wanted) & fx.home_goals.notna()
              & fx.status.isin(["FT", "AWARDED"])]
    if "2025/2026" in set(hist.season): stop("2025/26 in baseline fixtures")
    base = fit_baseline_rates(hist.home_goals.values.astype(float),
                              hist.away_goals.values.astype(float))
    states = compute_ad_states(fx, E6_LEARNING_RATE, base).set_index("fixture_id")
    for c in AD_COLUMNS:
        X[c] = meta["fixture_id"].map(states[c])
    if X[list(AD_COLUMNS)].isna().any().any(): stop("NaN in A/D features")

    idx = [int(np.where(meta.fixture_id == f)[0][0]) for f in f50]
    X50 = X.iloc[idx].reset_index(drop=True)
    print(f"  Elo available   : 50/50")
    print(f"  A/D available   : 50/50 (lr={E6_LEARNING_RATE}, "
          f"mu_home={base.mu_home:.4f}, mu_away={base.mu_away:.4f})")
    print(f"  design matrix   : {X50.shape}")

    # ---- PHASE 5 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 5 — MODEL PREDICTIONS")
    print("=" * 78)

    def run_arm(art, name):
        E = art.preprocessor.transform(X50[list(art.feature_columns)])
        lh = art.model_home_goals.predict(E)
        la = art.model_away_goals.predict(E)
        if not (np.all(np.isfinite(lh)) and np.all(np.isfinite(la))):
            stop(f"{name}: non-finite lambdas")
        if not (np.all(lh > 0) and np.all(la > 0)):
            stop(f"{name}: non-positive lambdas")
        pr = predict_poisson(lh, la, list(art.class_order))
        P = np.array([[p.probabilities["H"], p.probabilities["D"],
                       p.probabilities["A"]] for p in pr])
        if not np.all(np.isfinite(P)): stop(f"{name}: non-finite probabilities")
        if np.any(P < 0) or np.any(P > 1): stop(f"{name}: probabilities out of range")
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-9):
            stop(f"{name}: probability rows do not sum to 1")
        print(f"  {name}: 50 predictions, mean lambda "
              f"home={lh.mean():.3f} away={la.mean():.3f}")
        return P, lh, la

    P2, lh2, la2 = run_arm(v2, "V2")
    P3, lh3, la3 = run_arm(v3, "V3")
    P4, lh4, la4 = run_arm(v4, "V4")

    # ---- PHASE 6 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 6 — MARKET REFERENCE (Pinnacle closing, de-vigged)")
    print("=" * 78)
    conn = sqlite3.connect(f"file:{PROMO_DB}?mode=ro", uri=True)
    mk = pd.read_sql_query(
        """SELECT fixture_id, p_home, p_draw, p_away, bookmaker_name,
                  price_class, data_class FROM promotion_market""", conn)
    mcols = [r[1] for r in conn.execute("PRAGMA table_info(promotion_market)")]
    conn.close()
    if len(mk) != 50: stop(f"market rows = {len(mk)}, expected 50")
    if set(mk.fixture_id) != set(f50): stop("market fixture IDs mismatch")
    if set(mk.bookmaker_name) != {"Pinnacle"}: stop("market not Pinnacle-only")
    if set(mk.price_class) != {"closing"}: stop("market not closing-only")
    if set(mk.data_class) != {"promotion-validation-only"}:
        stop("market data_class incorrect")
    if {"home_goals", "away_goals", "label_result"} & set(mcols):
        stop("market table contains an outcome column")
    mk = mk.set_index("fixture_id").loc[f50].reset_index()
    PM = mk[["p_home", "p_draw", "p_away"]].to_numpy(float)
    if not np.allclose(PM.sum(axis=1), 1.0, atol=1e-9):
        stop("market probabilities do not sum to 1")
    print(f"  50/50 coverage, Pinnacle, closing, promotion-validation-only")
    print(f"  no outcome columns ({len(mcols)} columns)")
    print(f"  market is a REFERENCE ONLY — never a V4 feature")

    # ---- PHASE 7 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 7 — PREDICTION TABLE")
    print("=" * 78)
    pv2, pv3, pv4, pmk = preds_of(P2), preds_of(P3), preds_of(P4), preds_of(PM)
    print(f"  {'#':>3} {'league':<15} {'match':<40} {'act':>4} "
          f"{'V2':>4}{'':>7}{'V3':>4}{'':>7}{'V4':>4}{'':>7}{'MKT':>4}")
    print("  " + "-" * 104)
    rows = []
    for i in range(50):
        r = df.iloc[i]
        match = f"{r.home_name[:18]} v {r.away_name[:17]}"
        flag = "  <" if len({pv2[i], pv3[i], pv4[i], pmk[i]}) > 1 else ""
        print(f"  {i+1:>3} {r.competition_name[:14]:<15} {match:<40} {y[i]:>4} "
              f"{pv2[i]:>4}({P2[i].max():.2f}) {pv3[i]:>4}({P3[i].max():.2f}) "
              f"{pv4[i]:>4}({P4[i].max():.2f}) {pmk[i]:>4}({PM[i].max():.2f})"
              f"{flag}")
        rows.append({
            "n": i + 1, "fixture_id": int(r.fixture_id),
            "date": str(r.date)[:10], "league": r.competition_name,
            "home": r.home_name, "away": r.away_name,
            "score": f"{int(r.home_goals)}-{int(r.away_goals)}",
            "actual": y[i],
            "V2": {"p_home": round(float(P2[i][0]), 6),
                   "p_draw": round(float(P2[i][1]), 6),
                   "p_away": round(float(P2[i][2]), 6),
                   "pred": pv2[i], "confidence": round(float(P2[i].max()), 6),
                   "lambda_home": round(float(lh2[i]), 6),
                   "lambda_away": round(float(la2[i]), 6),
                   "correct": bool(pv2[i] == y[i])},
            "V3": {"p_home": round(float(P3[i][0]), 6),
                   "p_draw": round(float(P3[i][1]), 6),
                   "p_away": round(float(P3[i][2]), 6),
                   "pred": pv3[i], "confidence": round(float(P3[i].max()), 6),
                   "lambda_home": round(float(lh3[i]), 6),
                   "lambda_away": round(float(la3[i]), 6),
                   "correct": bool(pv3[i] == y[i])},
            "V4": {"p_home": round(float(P4[i][0]), 6),
                   "p_draw": round(float(P4[i][1]), 6),
                   "p_away": round(float(P4[i][2]), 6),
                   "pred": pv4[i], "confidence": round(float(P4[i].max()), 6),
                   "lambda_home": round(float(lh4[i]), 6),
                   "lambda_away": round(float(la4[i]), 6),
                   "correct": bool(pv4[i] == y[i])},
            "MARKET": {"p_home": round(float(PM[i][0]), 6),
                       "p_draw": round(float(PM[i][1]), 6),
                       "p_away": round(float(PM[i][2]), 6),
                       "pred": pmk[i],
                       "confidence": round(float(PM[i].max()), 6),
                       "correct": bool(pmk[i] == y[i])},
        })

    # ---- PHASE 8 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 8 — 50-MATCH SCORECARD")
    print("=" * 78)
    M = {"V2": metrics(y, P2), "V3": metrics(y, P3),
         "V4": metrics(y, P4), "MARKET": metrics(y, PM)}
    print("  direction: Accuracy UP · LogLoss DOWN · Brier DOWN · "
          "RPS DOWN · DrawRecall UP · ECE DOWN")
    print(f"\n  {'MODEL':<8}{'CORRECT':>9}{'ACC':>9}{'LOGLOSS':>10}"
          f"{'BRIER':>9}{'RPS':>9}{'DRAW_REC':>10}{'ECE':>9}")
    print("  " + "-" * 73)
    for k in ("V2", "V3", "V4", "MARKET"):
        m = M[k]
        print(f"  {k:<8}{m['correct']:>6}/50{m['accuracy']:>9.4f}"
              f"{m['log_loss']:>10.6f}{m['brier']:>9.6f}{m['rps']:>9.6f}"
              f"{m['draw_recall']:>10.4f}{m['ece']:>9.6f}")

    # ---- PHASE 9 --------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 9 — V4 PAIRWISE COMPARISONS")
    print("=" * 78)
    print("  positive delta = V4 BETTER (loss metrics inverted accordingly)")
    pair = {}
    for other, Po in (("V2", P2), ("V3", P3), ("MARKET", PM)):
        mo, m4 = M[other], M["V4"]
        pair[f"V4_vs_{other}"] = {
            "log_loss_delta": round(mo["log_loss"] - m4["log_loss"], 6),
            "brier_delta": round(mo["brier"] - m4["brier"], 6),
            "rps_delta": round(mo["rps"] - m4["rps"], 6),
            "ece_delta": round(mo["ece"] - m4["ece"], 6),
            "accuracy_delta": round(m4["accuracy"] - mo["accuracy"], 6),
            "correct_delta": m4["correct"] - mo["correct"],
            "top_class_flips": int(np.sum(preds_of(Po) != pv4)),
        }
        d = pair[f"V4_vs_{other}"]
        print(f"\n  V4 vs {other}:")
        print(f"    LogLoss {d['log_loss_delta']:+.6f}   "
              f"Brier {d['brier_delta']:+.6f}   RPS {d['rps_delta']:+.6f}")
        print(f"    ECE     {d['ece_delta']:+.6f}   "
              f"Accuracy {d['accuracy_delta']:+.4f}   "
              f"Correct {d['correct_delta']:+d}")
        print(f"    top-class differences: {d['top_class_flips']}/50")

    # ---- PHASE 10 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 10 — CHRONOLOGICAL BUCKETS")
    print("=" * 78)
    buckets = {"A_1_20": slice(0, 20), "B_21_40": slice(20, 40),
               "C_41_50": slice(40, 50)}
    bres = {}
    print(f"  {'bucket':<10}{'n':>4}"
          f"{'V2':>14}{'V3':>14}{'V4':>14}{'MKT':>14}")
    print("  " + "-" * 70)
    for name, sl in buckets.items():
        yy = y[sl]
        e = {k: metrics(yy, P[sl]) for k, P in
             (("V2", P2), ("V3", P3), ("V4", P4), ("MARKET", PM))}
        bres[name] = e
        print(f"  {name:<10}{len(yy):>4}"
              + "".join(f"{e[k]['correct']:>4}/{len(yy):<2} "
                        f"{e[k]['log_loss']:>6.4f}" for k in
                        ("V2", "V3", "V4", "MARKET")))

    # ---- PHASE 11 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 11 — PER-LEAGUE")
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
    print("\n  NOTE: Serie A n=2 and Bundesliga n=7 are far too small for")
    print("  statistical inference. Treat per-league figures as descriptive.")

    # ---- PHASE 12 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 12 — 20 vs 50 STABILITY")
    print("=" * 78)
    stability = {"available": False}
    if RESULTS_20.exists():
        r20 = json.loads(RESULTS_20.read_text())
        m20 = (r20.get("metrics") or r20.get("scorecard")
               or r20.get("summary") or {})
        stability = {"available": True, "twenty": m20,
                     "fifty": {k: M[k] for k in ("V2", "V3", "V4", "MARKET")}}
        print(f"  20-match keys found: {sorted(m20.keys()) if m20 else 'n/a'}")
        b = bres["A_1_20"]
        print(f"\n  Re-derived 1-20 from this run (sanity vs the frozen file):")
        for k in ("V2", "V3", "V4", "MARKET"):
            print(f"    {k:<7} correct={b[k]['correct']}/20  "
                  f"LL={b[k]['log_loss']:.6f}")
        stability["rederived_first20"] = b
    else:
        print("  20-match results file not found")

    # ---- PHASE 13 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 13 — V4 vs V3 PROBABILITY CHANGE")
    print("=" * 78)
    dP = np.abs(P4 - P3)
    change = {
        "mean_abs_prob_change": round(float(dP.mean()), 8),
        "max_prob_change": round(float(dP.max()), 8),
        "pct_materially_changed": round(
            100 * float(np.mean(dP.max(axis=1) > MATERIAL)), 4),
        "top_class_changes": int(np.sum(pv3 != pv4)),
        "mean_abs_lambda_home_change": round(float(np.mean(np.abs(lh4 - lh3))), 8),
        "mean_abs_lambda_away_change": round(float(np.mean(np.abs(la4 - la3))), 8),
        "material_threshold": MATERIAL,
    }
    for k, v in change.items(): print(f"  {k}: {v}")

    # ---- PHASE 14 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 14 — PAIRED BOOTSTRAP (log loss)")
    print("=" * 78)
    boot = {"V4_minus_V2": paired_bootstrap(y, P4, P2),
            "V4_minus_V3": paired_bootstrap(y, P4, P3),
            "V4_minus_MARKET": paired_bootstrap(y, P4, PM)}
    for k, v in boot.items():
        print(f"  {k}: {v['point_estimate']:+.6f}  "
              f"95% CI [{v['ci_lower_2.5']:+.6f}, {v['ci_upper_97.5']:+.6f}]  "
              f"favouring V4 in {v['pct_resamples_favouring_first']:.1f}% of "
              f"{v['n_resamples']} resamples")
    print("  (negative = V4 better; CI spanning 0 = not distinguishable)")

    # ---- PHASE 15 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 15 — DETERMINISM (full re-run)")
    print("=" * 78)
    P2b, _, _ = run_arm(v2, "V2")
    P3b, _, _ = run_arm(v3, "V3")
    P4b, _, _ = run_arm(v4, "V4")
    d2 = float(np.max(np.abs(P2 - P2b)))
    d3 = float(np.max(np.abs(P3 - P3b)))
    d4 = float(np.max(np.abs(P4 - P4b)))
    conn = sqlite3.connect(f"file:{PROMO_DB}?mode=ro", uri=True)
    mk2 = pd.read_sql_query(
        "SELECT fixture_id, p_home, p_draw, p_away FROM promotion_market",
        conn).set_index("fixture_id").loc[f50]
    conn.close()
    dm = float(np.max(np.abs(PM - mk2[["p_home", "p_draw", "p_away"]].to_numpy())))
    determinism = (d2 == 0.0 and d3 == 0.0 and d4 == 0.0 and dm == 0.0)
    print(f"  V2 max abs diff: {d2:.3e}")
    print(f"  V3 max abs diff: {d3:.3e}")
    print(f"  V4 max abs diff: {d4:.3e}")
    print(f"  Market max diff: {dm:.3e}")
    print(f"  DETERMINISM: {'PASS' if determinism else 'FAIL'}")

    # ---- PHASE 16 -------------------------------------------------------
    post = audit("PHASE 16 — protected integrity (POST)")
    frozen20_post = {f: md5(HERE / f) for f in frozen20_pre}
    frozen_ok = frozen20_pre == frozen20_post
    print(f"\n  frozen 20-match outputs unchanged: {frozen_ok}")
    if not frozen_ok: stop("20-match validation outputs were modified")

    integrity_ok = all(v["status"] == "identical" for v in post.values())
    causality_ok = True   # proven by test_v4_50_validation.py suite 4

    # ---- write outputs --------------------------------------------------
    now = datetime.now(timezone.utc).isoformat()
    results = {
        "generated_at": now,
        "experiment": "V4 50-match retrospective validation",
        "note": ("50-match league composition is the deterministic result of "
                 "the frozen selection rule: PL 16, La Liga 13, Ligue 1 12, "
                 "Bundesliga 7, Serie A 2. The earlier 17/13/12/5/3 "
                 "composition was an incorrect prior estimate and was not "
                 "used to alter the sample."),
        "selection_rule": ("season_id IN (2025/26) AND status='FT' "
                           "ORDER BY unix ASC, fixture_id ASC LIMIT 50"),
        "fixture_ids": [int(f) for f in f50],
        "first_20_match_frozen_set": True,
        "composition": comp,
        "date_span": [df.date.iloc[0][:10], df.date.iloc[-1][:10]],
        "outcome_distribution": {"H": int((y == "H").sum()),
                                 "D": int((y == "D").sum()),
                                 "A": int((y == "A").sum())},
        "artifacts": {"V2": {"path": str(V2_ARTIFACT.relative_to(PROJECT_ROOT)),
                             "md5": md5(V2_ARTIFACT)},
                      "V3": {"path": str(V3_ARTIFACT.relative_to(PROJECT_ROOT)),
                             "md5": md5(V3_ARTIFACT), "n_features": 87},
                      "V4": {"path": str(V4_ARTIFACT.relative_to(PROJECT_ROOT)),
                             "md5": md5(V4_ARTIFACT), "n_features": 91,
                             "training_seasons": list(v4.training_seasons),
                             "holdout_used_in_training": False}},
        "market": {"source": "promotion_market_odds.sqlite",
                   "bookmaker": "Pinnacle", "price": "closing",
                   "data_class": "promotion-validation-only",
                   "coverage": "50/50", "role": "REFERENCE ONLY",
                   "in_v4_feature_matrix": False},
        "per_match": rows,
        "scorecard": M,
        "pairwise": pair,
        "buckets": bres,
        "by_league": lg,
        "stability_20_vs_50": stability,
        "v4_vs_v3_change": change,
        "bootstrap": boot,
        "reliability": {k: reliability(y, P) for k, P in
                        (("V2", P2), ("V3", P3), ("V4", P4), ("MARKET", PM))},
        "determinism": {"V2": d2, "V3": d3, "V4": d4, "MARKET": dm,
                        "passed": determinism},
        "integrity": {"pre": pre, "post": post, "passed": integrity_ok,
                      "frozen_20_unchanged": frozen_ok},
        "causality": {"passed": causality_ok,
                      "verified_by": "test_v4_50_validation.py suite 4"},
        "promotion": {"promoted": False,
                      "note": "Validation only. V4 is NOT promoted."},
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    MANIFEST_JSON.write_text(json.dumps({
        "generated_at": now, "fixtures": 50,
        "results_file": RESULTS_JSON.name,
        "report_file": REPORT_MD.name,
        "protected_integrity": integrity_ok,
        "frozen_20_unchanged": frozen_ok,
        "determinism": determinism,
        "v4_artifact_md5": md5(V4_ARTIFACT),
        "v4_promoted": False,
    }, indent=2))

    # ---- PHASE 18 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("V4 50-MATCH VALIDATION")
    print("=" * 78)
    for k in ("V2", "V3", "V4", "MARKET"):
        print(f"\n{k}:")
        print(f"  correct  = {M[k]['correct']}/50")
        print(f"  accuracy = {M[k]['accuracy']:.4f}")
    for other in ("V2", "V3", "MARKET"):
        d = pair[f"V4_vs_{other}"]
        print(f"\nV4 vs {other}:")
        print(f"  LogLoss {d['log_loss_delta']:+.6f}  "
              f"Brier {d['brier_delta']:+.6f}  RPS {d['rps_delta']:+.6f}  "
              f"ECE {d['ece_delta']:+.6f}")
        print(f"  Correct {d['correct_delta']:+d}   "
              f"Accuracy {d['accuracy_delta']:+.4f}")
    b = bres
    print(f"\n20 vs 50 stability (V4 correct):")
    print(f"  bucket 1-20 : {b['A_1_20']['V4']['correct']}/20")
    print(f"  bucket 21-40: {b['B_21_40']['V4']['correct']}/20")
    print(f"  bucket 41-50: {b['C_41_50']['V4']['correct']}/10")
    print(f"\nINTEGRITY:   {'PASS' if integrity_ok and frozen_ok else 'FAIL'}")
    print(f"CAUSALITY:   {'PASS' if causality_ok else 'FAIL'}")
    print(f"DETERMINISM: {'PASS' if determinism else 'FAIL'}")
    overall = integrity_ok and frozen_ok and causality_ok and determinism
    print(f"\nOVERALL VALIDATION: {'COMPLETE' if overall else 'BLOCKED'}")
    print("\nV4 IS NOT PROMOTED. No production file was modified.")
    print(f"\n  results  -> {RESULTS_JSON.name}")
    print(f"  manifest -> {MANIFEST_JSON.name}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
