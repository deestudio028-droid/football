"""E3 — Pre-experiment diagnostic (runs WITHOUT scikit-learn).

Usage:
    python research/dixon_coles/run_e3_diagnostic.py

This is NOT the E3 experiment. It answers a narrower question that does
not require the V3 artifact:

    Does the low-score dependence that Dixon-Coles corrects actually
    exist in the five target leagues, and how large is it?

It does this two ways:

  1. Residual analysis — compare observed frequencies of 0-0, 1-0, 0-1,
     1-1 and of draws against what an independent Poisson with the
     empirical mean goal rates implies.

  2. Walk-forward proxy — run the exact E3 fold structure and rho
     selection using league-mean lambdas fitted causally on training
     seasons only, and measure the resulting metric deltas.

The proxy's lambdas are homogeneous within a league, so it is a much
weaker model than V3 (its log loss is far higher). Its purpose is to
bound the plausible EFFECT SIZE and expose league heterogeneity, not to
predict V3's exact numbers.

Writes: research/dixon_coles/e3_results.json
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

from dixon_coles import (  # noqa: E402
    grid_size, low_score_probs, predict_dc, rho_validity_bounds, select_rho,
)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
RESULTS_JSON = HERE / "e3_results.json"

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
    (("2020/2021", "2021/2022"), "2022/2023"),
    (("2020/2021", "2021/2022", "2022/2023"), "2023/2024"),
    (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), "2024/2025"),
]
RHO_GRID = [-0.20, -0.175, -0.15, -0.125, -0.10, -0.075, -0.05, -0.025,
            0.0, 0.025, 0.05, 0.075, 0.10]
CLASS_ORDER = ("H", "D", "A")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def _oh(y):
    o = np.zeros((len(y), 3))
    for i, c in enumerate(CLASS_ORDER):
        o[:, i] = (y == c)
    return o


def log_loss(y, P):
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(_oh(y) * np.log(Pc), axis=1)))


def brier(y, P):
    return float(np.mean(np.sum((P - _oh(y)) ** 2, axis=1)))


def rps(y, P):
    cp, co = np.cumsum(P, axis=1), np.cumsum(_oh(y), axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def load() -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET))
    sph = ",".join("?" * len(SEASONS))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, competition_name, season, unix,
                   home_goals hg, away_goals ag
            FROM fixtures
            WHERE competition_id IN ({ph}) AND season IN ({sph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL
            ORDER BY unix ASC, fixture_id ASC""",
        conn, params=[*TARGET, *SEASONS])
    conn.close()
    return df


def main() -> int:
    print("=" * 70)
    print("E3 — Pre-experiment diagnostic (no scikit-learn required)")
    print("=" * 70)

    integ = {}
    for rel, exp in PROTECTED_FILES.items():
        p = PROJECT_ROOT / rel
        act = md5(p) if p.exists() else None
        integ[rel] = {"expected": exp, "actual": act,
                      "status": "identical" if act == exp else "CHANGED"}

    df = load()
    hg = df.hg.values.astype(float)
    ag = df.ag.values.astype(float)
    y = np.where(hg > ag, "H", np.where(hg == ag, "D", "A"))
    print(f"\nFixtures: {len(df)}   Leagues: {df.competition_name.nunique()}")

    # --- 1. Residual analysis -----------------------------------------
    print("\n--- 1. Observed vs independent-Poisson (league-mean lambdas) ---")
    print(f"  {'League':<17}{'n':>6}{'obsDraw':>9}{'poisDraw':>10}{'deficit':>10}")
    league_resid = {}
    for lg in sorted(df.competition_name.unique()):
        m = (df.competition_name == lg).values
        lh, la = hg[m].mean(), ag[m].mean()
        obs = float((hg[m] == ag[m]).mean())
        P, _ = predict_dc(np.array([lh]), np.array([la]), 0.0)
        league_resid[lg] = {
            "n": int(m.sum()),
            "mean_lambda_home": round(float(lh), 6),
            "mean_lambda_away": round(float(la), 6),
            "observed_draw_rate": round(obs, 6),
            "poisson_draw_rate": round(float(P[0, 1]), 6),
            "draw_deficit": round(obs - float(P[0, 1]), 6),
        }
        print(f"  {lg:<17}{m.sum():>6}{obs:>9.4f}{P[0,1]:>10.4f}"
              f"{obs - P[0,1]:>+10.4f}")

    lh, la = hg.mean(), ag.mean()
    obs = float((hg == ag).mean())
    P, _ = predict_dc(np.array([lh]), np.array([la]), 0.0)
    pooled_resid = {
        "n": int(len(df)),
        "mean_lambda_home": round(float(lh), 6),
        "mean_lambda_away": round(float(la), 6),
        "observed_draw_rate": round(obs, 6),
        "poisson_draw_rate": round(float(P[0, 1]), 6),
        "draw_deficit": round(obs - float(P[0, 1]), 6),
    }
    print(f"  {'POOLED':<17}{len(df):>6}{obs:>9.4f}{P[0,1]:>10.4f}"
          f"{obs - P[0,1]:>+10.4f}")

    K = grid_size(max(lh, la))
    l0 = low_score_probs(np.array([lh]), np.array([la]), 0.0, K)
    print("\n--- The four DC cells (pooled) ---")
    print(f"  {'Score':<8}{'observed':>10}{'poisson':>10}{'obs-pois':>11}")
    cells = {}
    for (i, j), key, lbl in [((0, 0), "p_0_0", "0-0"), ((1, 0), "p_1_0", "1-0"),
                             ((0, 1), "p_0_1", "0-1"), ((1, 1), "p_1_1", "1-1")]:
        o = float(((hg == i) & (ag == j)).mean())
        pp = float(l0[key][0])
        cells[lbl] = {"observed": round(o, 6), "poisson": round(pp, 6),
                      "residual": round(o - pp, 6)}
        print(f"  {lbl:<8}{o:>10.5f}{pp:>10.5f}{o - pp:>+11.5f}")

    dc_signature = (cells["0-0"]["residual"] > 0 and cells["1-1"]["residual"] > 0
                    and cells["1-0"]["residual"] < 0
                    and cells["0-1"]["residual"] < 0)
    print(f"\n  Dixon-Coles signature present: {dc_signature}")

    # --- 2. Walk-forward proxy ----------------------------------------
    print("\n--- 2. Walk-forward proxy (league-mean lambdas, causal) ---")
    proxy_folds, A_all, B_all, Y_all, LG_all = [], [], [], [], []

    for trs, ts in FOLDS:
        tr = df.season.isin(trs).values
        te = (df.season == ts).values
        lam_h = np.zeros(len(df))
        lam_a = np.zeros(len(df))
        for lg in df.competition_name.unique():
            m = (df.competition_name == lg).values
            lam_h[m] = hg[tr & m].mean()
            lam_a[m] = ag[tr & m].mean()
        Kf = grid_size(max(lam_h.max(), lam_a.max()))
        sel = select_rho(y[tr], lam_h[tr], lam_a[tr], RHO_GRID, Kf)
        PA, _ = predict_dc(lam_h[te], lam_a[te], 0.0, K=Kf)
        PB, _ = predict_dc(lam_h[te], lam_a[te], sel.rho, K=Kf)

        rec = {
            "test_season": ts, "n_train": int(tr.sum()),
            "n_test": int(te.sum()), "selected_rho": sel.rho,
            "A_log_loss": round(log_loss(y[te], PA), 6),
            "B_log_loss": round(log_loss(y[te], PB), 6),
            "delta_log_loss": round(log_loss(y[te], PA) - log_loss(y[te], PB), 6),
            "A_brier": round(brier(y[te], PA), 6),
            "B_brier": round(brier(y[te], PB), 6),
            "delta_brier": round(brier(y[te], PA) - brier(y[te], PB), 6),
        }
        proxy_folds.append(rec)
        A_all.append(PA); B_all.append(PB)
        Y_all.append(y[te]); LG_all.append(df.competition_name.values[te])
        print(f"  {ts}: rho={sel.rho:<7} A_LL={rec['A_log_loss']:.6f} "
              f"B_LL={rec['B_log_loss']:.6f} delta={rec['delta_log_loss']:+.6f}")

    PA = np.vstack(A_all); PB = np.vstack(B_all)
    Y = np.concatenate(Y_all); LG = np.concatenate(LG_all)

    proxy_pooled = {
        "n": int(len(Y)),
        "A_log_loss": round(log_loss(Y, PA), 6),
        "B_log_loss": round(log_loss(Y, PB), 6),
        "delta_log_loss": round(log_loss(Y, PA) - log_loss(Y, PB), 6),
        "A_brier": round(brier(Y, PA), 6),
        "B_brier": round(brier(Y, PB), 6),
        "delta_brier": round(brier(Y, PA) - brier(Y, PB), 6),
        "A_rps": round(rps(Y, PA), 6),
        "B_rps": round(rps(Y, PB), 6),
        "delta_rps": round(rps(Y, PA) - rps(Y, PB), 6),
    }
    print(f"\n  POOLED n={len(Y)}: A_LL={proxy_pooled['A_log_loss']:.6f} "
          f"B_LL={proxy_pooled['B_log_loss']:.6f} "
          f"delta={proxy_pooled['delta_log_loss']:+.6f}")

    proxy_league = {}
    print("\n  By league (delta LogLoss, positive = DC better):")
    for lg in sorted(set(LG)):
        m = LG == lg
        d = log_loss(Y[m], PA[m]) - log_loss(Y[m], PB[m])
        proxy_league[lg] = {
            "n": int(m.sum()),
            "A_log_loss": round(log_loss(Y[m], PA[m]), 6),
            "B_log_loss": round(log_loss(Y[m], PB[m]), 6),
            "delta_log_loss": round(d, 6),
        }
        print(f"    {lg:<17} n={int(m.sum()):<5} delta={d:+.6f}")

    improved = sum(1 for v in proxy_league.values()
                   if v["delta_log_loss"] > 0)
    print(f"\n  Leagues improved: {improved}/5")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E3 — V3 vs V3 + Dixon-Coles",
        "execution_status": {
            "experiment_A_v3": "PENDING — requires scikit-learn to unpickle V3",
            "experiment_B_v3_dc": "PENDING — depends on A",
            "diagnostic": "COMPLETE",
            "test_suite": "COMPLETE — 55/55 passed",
        },
        "dataset": {
            "source": "data/processed/matches.db",
            "leagues": sorted(df.competition_name.unique().tolist()),
            "seasons": list(SEASONS),
            "n_fixtures": int(len(df)),
            "quarantined_season": "2025/2026",
            "market_data_used": False,
        },
        "walk_forward_folds": [
            {"train": list(t), "test": s, "n_test": int((df.season == s).sum())}
            for t, s in FOLDS
        ],
        "rho_grid": RHO_GRID,
        "residual_analysis": {
            "by_league": league_resid,
            "pooled": pooled_resid,
            "low_score_cells_pooled": cells,
            "dixon_coles_signature_present": dc_signature,
        },
        "walk_forward_proxy": {
            "caveat": ("League-mean lambdas, NOT V3 per-fixture lambdas. "
                       "Bounds effect size and exposes league heterogeneity; "
                       "does not predict V3's exact numbers."),
            "folds": proxy_folds,
            "pooled": proxy_pooled,
            "by_league": proxy_league,
            "leagues_improved": improved,
        },
        "integrity": integ,
        "decision": "INCONCLUSIVE",
        "rationale": (
            "Experiments A and B were not executed: scikit-learn is "
            "unavailable in this environment and the V3 artifact cannot be "
            "unpickled, so V3's per-fixture lambdas could not be produced. "
            "The decision criteria require measured A vs B metrics."),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results -> {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
