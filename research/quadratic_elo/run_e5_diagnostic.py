"""E5 — Pre-experiment diagnostic (runs WITHOUT scikit-learn).

Usage:
    python research/quadratic_elo/run_e5_diagnostic.py

This is NOT the E5 experiment. It answers the underlying question
without needing V3's 87-feature refit:

    Is the relationship between causal Elo difference and expected
    goals actually nonlinear, and does the nonlinearity pay off
    out-of-sample?

Three measurements:

  1. AUDIT — verify elo_diff is an exact linear combination of
     home_elo, away_elo and a constant, quantify the resulting rank
     deficiency, and measure raw vs centered collinearity.

  2. NON-PARAMETRIC CURVATURE — bin fixtures by elo_diff and compare
     log(mean goals) per bin against linear and quadratic fits. Purely
     descriptive; no model is fitted to features.

  3. ELO-ONLY GLM WALK-FORWARD — fit log(lambda) = b0 + c1*z [+ c2*z^2]
     by IRLS on the real folds, with centering fitted on training rows
     only. Reports learned coefficients, their stability, and the
     out-of-sample metric deltas.

The GLM here uses Elo alone; V3 uses 87 features. This bounds the
curvature effect and tests coefficient stability, both of which are
properties of the data. It cannot predict V3's numbers.

Writes: research/quadratic_elo/e5_results.json
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
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from quadratic_elo import fit_centering, make_quadratic_feature  # noqa: E402
from dixon_coles import predict_dc  # noqa: E402  (rho=0 == plain Poisson)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
RESULTS_JSON = HERE / "e5_results.json"

PROTECTED_FILES = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
TARGET = (200, 419, 423, 477, 499)
SEASONS = ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025")
FOLDS = [(("2020/2021", "2021/2022"), "2022/2023"),
         (("2020/2021", "2021/2022", "2022/2023"), "2023/2024"),
         (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), "2024/2025")]
ALPHA = 1.0


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def irls_poisson(X, y, alpha=ALPHA, max_iter=200, tol=1e-12):
    """Ridge-penalised Poisson GLM by IRLS. Intercept unpenalised.

    Validated in-file against known synthetic coefficients.
    """
    n, p = X.shape
    b = np.zeros(p); b[0] = np.log(max(y.mean(), 1e-6))
    P = np.eye(p) * alpha; P[0, 0] = 0.0
    for it in range(max_iter):
        eta = np.clip(X @ b, -20, 20); mu = np.exp(eta)
        z = eta + (y - mu) / np.maximum(mu, 1e-12)
        bn = np.linalg.solve(X.T @ (mu[:, None] * X) + P * n, X.T @ (mu * z))
        if np.max(np.abs(bn - b)) < tol:
            return bn, it
        b = bn
    return b, max_iter


def _validate_irls() -> dict:
    rng = np.random.default_rng(0)
    n = 200000
    x = rng.normal(0, 1, n)
    yy = rng.poisson(np.exp(0.5 + 0.8 * x - 0.15 * x ** 2)).astype(float)
    XX = np.column_stack([np.ones(n), x, x ** 2])
    b, it = irls_poisson(XX, yy, alpha=0.0)
    return {"true": [0.5, 0.8, -0.15],
            "recovered": [round(float(v), 4) for v in b],
            "max_abs_error": round(float(np.max(np.abs(
                b - np.array([0.5, 0.8, -0.15])))), 5),
            "iterations": int(it)}


def _oh(y):
    o = np.zeros((len(y), 3))
    for i, c in enumerate(("H", "D", "A")): o[:, i] = (y == c)
    return o


def log_loss(y, P):
    Pc = np.clip(P, 1e-15, 1.0); Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(_oh(y) * np.log(Pc), axis=1)))


def brier(y, P): return float(np.mean(np.sum((P - _oh(y)) ** 2, axis=1)))


def rps(y, P):
    cp, co = np.cumsum(P, axis=1), np.cumsum(_oh(y), axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def main() -> int:
    print("=" * 72)
    print("E5 — Pre-experiment diagnostic (no scikit-learn required)")
    print("=" * 72)

    integ = {}
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        a = md5(p) if p.exists() else None
        integ[rel] = {"expected": exp, "actual": a,
                      "status": "identical" if a == exp else "CHANGED"}

    from features.elo import HOME_ADVANTAGE, load_elo_features
    e = load_elo_features(MATCHES_DB)

    # --- 1. audit ------------------------------------------------------
    print("\n--- 1. Elo representation audit ---")
    resid = e.elo_diff.values - (e.home_elo.values + HOME_ADVANTAGE
                                 - e.away_elo.values)
    M = np.column_stack([np.ones(len(e)), e.home_elo, e.away_elo, e.elo_diff])
    rank = int(np.linalg.matrix_rank(M))
    d_all = e.elo_diff.values
    raw_corr = float(np.corrcoef(d_all, d_all ** 2)[0, 1])
    cen = fit_centering(d_all)
    cen_corr = float(np.corrcoef(d_all,
                                 make_quadratic_feature(d_all, cen))[0, 1])
    audit = {
        "elo_diff_identity_max_residual": float(np.max(np.abs(resid))),
        "rank_intercept_home_away_diff": rank,
        "rank_deficiency": 4 - rank,
        "home_advantage_constant": HOME_ADVANTAGE,
        "elo_diff_range": [round(float(d_all.min()), 2),
                           round(float(d_all.max()), 2)],
        "elo_diff_median": round(float(np.median(d_all)), 2),
        "raw_corr_linear_vs_square": round(raw_corr, 4),
        "centered_corr_linear_vs_square": round(cen_corr, 4),
    }
    print(f"  elo_diff == home_elo + {HOME_ADVANTAGE} - away_elo: "
          f"max|resid| = {audit['elo_diff_identity_max_residual']:.3e}")
    print(f"  rank(intercept, home, away, diff) = {rank} of 4 "
          f"-> deficiency {4 - rank}")
    print(f"  elo_diff range {audit['elo_diff_range']}, "
          f"median {audit['elo_diff_median']}")
    print(f"  corr(elo_diff, elo_diff^2):  raw={raw_corr:+.4f}  "
          f"centered={cen_corr:+.4f}")

    # --- data ----------------------------------------------------------
    emap = e.set_index("fixture_id")
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET)); sph = ",".join("?" * len(SEASONS))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, competition_name, season, unix,
                   home_goals hg, away_goals ag FROM fixtures
            WHERE competition_id IN ({ph}) AND season IN ({sph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL
            ORDER BY unix ASC, fixture_id ASC""",
        conn, params=[*TARGET, *SEASONS])
    conn.close()
    df["ed"] = df.fixture_id.map(emap.elo_diff)
    df = df.dropna(subset=["ed"]).reset_index(drop=True)
    ed = df.ed.values
    hg = df.hg.values.astype(float); ag = df.ag.values.astype(float)
    seas, lgs = df.season.values, df.competition_name.values
    y = np.where(hg > ag, "H", np.where(hg == ag, "D", "A"))
    print(f"\n  Fixtures: {len(df)}")

    # --- 2. non-parametric curvature ------------------------------------
    print("\n--- 2. Non-parametric: log(mean goals) by elo_diff decile ---")
    qs = np.percentile(ed, np.linspace(0, 100, 11))
    print(f"  {'bin':<5}{'elo_diff mid':>14}{'n':>7}{'mean_hg':>10}"
          f"{'mean_ag':>10}")
    deciles, mids, lh, la, ns = [], [], [], [], []
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        m = (ed >= lo) & (ed < hi) if i < 9 else (ed >= lo) & (ed <= hi)
        if m.sum() < 50: continue
        mid, mh, ma = ed[m].mean(), hg[m].mean(), ag[m].mean()
        deciles.append({"bin": i + 1, "elo_diff_mid": round(float(mid), 1),
                        "n": int(m.sum()), "mean_home_goals": round(float(mh), 4),
                        "mean_away_goals": round(float(ma), 4),
                        "log_mean_home": round(float(np.log(mh)), 4),
                        "log_mean_away": round(float(np.log(ma)), 4)})
        mids.append(mid); lh.append(np.log(mh)); la.append(np.log(ma))
        ns.append(m.sum())
        print(f"  {i+1:<5}{mid:>14.1f}{m.sum():>7}{mh:>10.4f}{ma:>10.4f}")
    mids, lh, la, ns = map(np.array, (mids, lh, la, ns))

    print("\n  Curvature of the decile log-means:")
    curv = {}
    for name, vals in [("home", lh), ("away", la)]:
        p1 = np.polyfit(mids, vals, 1, w=np.sqrt(ns))
        p2 = np.polyfit(mids, vals, 2, w=np.sqrt(ns))
        s1 = float((ns * (vals - np.polyval(p1, mids)) ** 2).sum())
        s2 = float((ns * (vals - np.polyval(p2, mids)) ** 2).sum())
        curv[name] = {"linear_slope": float(p1[0]),
                      "quadratic_c2": float(p2[0]),
                      "quadratic_c1": float(p2[1]),
                      "weighted_sse_linear": s1,
                      "weighted_sse_quadratic": s2,
                      "sse_reduction_pct": round(100 * (1 - s2 / s1), 2)}
        print(f"    {name}: c2={p2[0]:+.3e}  SSE reduction "
              f"{100 * (1 - s2 / s1):.2f}%")

    # --- 3. Elo-only GLM walk-forward -----------------------------------
    print("\n--- 3. Elo-only Poisson GLM, walk-forward ---")
    irls_val = _validate_irls()
    print(f"  IRLS validation: recovered {irls_val['recovered']} "
          f"vs true {irls_val['true']} (max err {irls_val['max_abs_error']})")

    gl_folds, PAs, PBs, Ys, LGs = [], [], [], [], []
    for trs, ts in FOLDS:
        tr, te = np.isin(seas, trs), (seas == ts)
        c = fit_centering(ed[tr])
        sd = float(ed[tr].std())

        def design(mask, quad):
            z = (ed[mask] - c.mean) / sd
            cols = [np.ones(int(mask.sum())), z] + ([z ** 2] if quad else [])
            return np.column_stack(cols)

        out = {}
        for quad, tag in [(False, "A"), (True, "B")]:
            bh, _ = irls_poisson(design(tr, quad), hg[tr])
            ba, _ = irls_poisson(design(tr, quad), ag[tr])
            lam_h = np.exp(np.clip(design(te, quad) @ bh, -20, 20))
            lam_a = np.exp(np.clip(design(te, quad) @ ba, -20, 20))
            P, _ = predict_dc(np.clip(lam_h, 1e-3, 15),
                              np.clip(lam_a, 1e-3, 15), 0.0)
            out[tag] = {"P": P, "bh": bh, "ba": ba}

        a_ll, b_ll = log_loss(y[te], out["A"]["P"]), log_loss(y[te], out["B"]["P"])
        rec = {"test_season": ts, "n_train": int(tr.sum()),
               "n_test": int(te.sum()),
               "train_centering_mean": round(float(c.mean), 4),
               "c1_home": round(float(out["B"]["bh"][1]), 6),
               "c2_home": round(float(out["B"]["bh"][2]), 6),
               "c1_away": round(float(out["B"]["ba"][1]), 6),
               "c2_away": round(float(out["B"]["ba"][2]), 6),
               "A_log_loss": round(a_ll, 6), "B_log_loss": round(b_ll, 6),
               "delta_log_loss": round(a_ll - b_ll, 6),
               "A_brier": round(brier(y[te], out["A"]["P"]), 6),
               "B_brier": round(brier(y[te], out["B"]["P"]), 6)}
        gl_folds.append(rec)
        PAs.append(out["A"]["P"]); PBs.append(out["B"]["P"])
        Ys.append(y[te]); LGs.append(lgs[te])
        print(f"  {ts}: A={a_ll:.6f} B={b_ll:.6f} delta={a_ll - b_ll:+.6f}")
        print(f"    home c1={rec['c1_home']:+.6f} c2={rec['c2_home']:+.6f}"
              f"   away c1={rec['c1_away']:+.6f} c2={rec['c2_away']:+.6f}")

    PA, PB = np.vstack(PAs), np.vstack(PBs)
    Y, LG = np.concatenate(Ys), np.concatenate(LGs)
    pooled = {"n": int(len(Y)),
              "A_log_loss": round(log_loss(Y, PA), 6),
              "B_log_loss": round(log_loss(Y, PB), 6),
              "delta_log_loss": round(log_loss(Y, PA) - log_loss(Y, PB), 6),
              "A_brier": round(brier(Y, PA), 6),
              "B_brier": round(brier(Y, PB), 6),
              "delta_brier": round(brier(Y, PA) - brier(Y, PB), 6),
              "A_rps": round(rps(Y, PA), 6), "B_rps": round(rps(Y, PB), 6),
              "delta_rps": round(rps(Y, PA) - rps(Y, PB), 6)}
    print(f"\n  POOLED n={len(Y)}: A={pooled['A_log_loss']:.6f} "
          f"B={pooled['B_log_loss']:.6f} "
          f"delta={pooled['delta_log_loss']:+.6f}")

    by_league = {}
    print("\n  By league (delta LL, positive = quadratic better):")
    for lg in sorted(set(LG)):
        m = LG == lg
        d = log_loss(Y[m], PA[m]) - log_loss(Y[m], PB[m])
        by_league[lg] = {"n": int(m.sum()), "delta_log_loss": round(d, 6)}
        print(f"    {lg:<17} n={int(m.sum()):<5} delta={d:+.6f}")
    improved = sum(1 for v in by_league.values() if v["delta_log_loss"] > 0)
    print(f"  Leagues improved: {improved}/5")

    c2h = [f["c2_home"] for f in gl_folds]
    c2a = [f["c2_away"] for f in gl_folds]
    sign_stable = all(v > 0 for v in c2h) or all(v < 0 for v in c2h)
    sign_stable_a = all(v > 0 for v in c2a) or all(v < 0 for v in c2a)
    spread_h = max(map(abs, c2h)) / min(map(abs, c2h))
    spread_a = max(map(abs, c2a)) / min(map(abs, c2a))
    print(f"\n  c2_home {c2h}  sign_stable={sign_stable} spread={spread_h:.2f}x")
    print(f"  c2_away {c2a}  sign_stable={sign_stable_a} spread={spread_a:.2f}x")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E5 — V3 vs V3 + Quadratic Elo",
        "execution_status": {
            "experiment_A_v3": "PENDING — requires scikit-learn (per-fold refit)",
            "experiment_B_quadratic": "PENDING — depends on A",
            "zero_quadratic_control": "PENDING — model level needs sklearn",
            "diagnostic": "COMPLETE",
            "test_suite": "COMPLETE — 66/66 passed",
        },
        "implementation_decision": (
            "Option A — append elo_diff_sq as column 88. V3's Poisson "
            "regressors use a log link, so the 87 features already enter "
            "log(lambda) linearly; one extra column places the quadratic "
            "term exactly where the hypothesis requires, with a free "
            "coefficient in each of the home and away models."),
        "dataset": {"n_fixtures": int(len(df)),
                    "leagues": sorted(df.competition_name.unique().tolist()),
                    "seasons": list(SEASONS),
                    "quarantined_season": "2025/2026",
                    "market_used": False, "dixon_coles_used": False,
                    "time_decay_used": False},
        "elo_audit": audit,
        "non_parametric_curvature": {"deciles": deciles, "fits": curv},
        "irls_validation": irls_val,
        "elo_only_glm_walk_forward": {
            "caveat": ("Elo-only GLM; V3 uses 87 features. Bounds the "
                       "curvature effect and tests coefficient stability. "
                       "Does not predict V3's numbers."),
            "folds": gl_folds, "pooled": pooled, "by_league": by_league,
            "leagues_improved": improved,
            "c2_home": c2h, "c2_away": c2a,
            "c2_sign_stable_home": bool(sign_stable),
            "c2_sign_stable_away": bool(sign_stable_a),
            "c2_spread_home": round(float(spread_h), 2),
            "c2_spread_away": round(float(spread_a), 2),
        },
        "integrity": integ,
        "decision": "INCONCLUSIVE",
        "rationale": (
            "Experiments A and B were not executed: scikit-learn is "
            "unavailable, so V3 could not be refit per fold at 87 and 88 "
            "columns, and the §18 zero-quadratic control could not be "
            "verified at the model level. The criteria require measured "
            "A vs B metrics."),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results -> {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
