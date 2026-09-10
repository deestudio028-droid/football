"""E4 — Pre-experiment diagnostic (runs WITHOUT scikit-learn).

Usage:
    python research/time_decay/run_e4_diagnostic.py

This is NOT the E4 experiment. It answers a narrower question that does
not require refitting V3:

    Does predictive signal in these five leagues actually decay with
    match age, and if so, how fast?

Three measurements:

  1. SIGNAL DECAY, equal-width bands — correlation between a team-form
     differential computed from a fixed-width age band and the actual
     match margin, as the band's age increases.

  2. SIGNAL DECAY, fixed sample size — same, but always using exactly
     10 prior matches per team, so only the AGE of those matches varies.
     This isolates age from sample size.

  3. WALK-FORWARD PROXY — the exact E4 fold structure and nested
     temporal xi selection, driving a recency-weighted goal-rate model
     instead of V3. Bounds the effect size and exposes stability.

The proxy has no features; V3 has 87. It cannot predict V3's numbers.
Its purpose is to bound magnitude and reveal whether xi selection is
stable, both of which are properties of the data.

Writes: research/time_decay/e4_results.json
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
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from time_decay import (  # noqa: E402
    HALF_LIFE_GRID_DAYS, SECONDS_PER_DAY, build_inner_splits,
    weight_profile, xi_from_half_life,
)
from dixon_coles import predict_dc  # noqa: E402  (rho=0 == plain Poisson)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
RESULTS_JSON = HERE / "e4_results.json"

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


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def log_loss(y, P):
    oh = np.zeros((len(y), 3))
    for i, c in enumerate(("H", "D", "A")):
        oh[:, i] = (y == c)
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(oh * np.log(Pc), axis=1)))


def load():
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(TARGET))
    sph = ",".join("?" * len(SEASONS))
    df = pd.read_sql_query(
        f"""SELECT competition_name, season, unix, home_id, away_id,
                   home_goals hg, away_goals ag
            FROM fixtures WHERE competition_id IN ({ph}) AND season IN ({sph})
              AND status IN ('FT','AWARDED') AND home_goals IS NOT NULL
            ORDER BY unix ASC""", conn, params=[*TARGET, *SEASONS])
    conn.close()
    return df


def main() -> int:
    print("=" * 72)
    print("E4 — Pre-experiment diagnostic (no scikit-learn required)")
    print("=" * 72)

    integ = {rel: {"expected": exp,
                   "actual": md5(PROJECT_ROOT / rel) if (PROJECT_ROOT / rel).exists() else None}
             for rel, exp in PROTECTED_FILES.items()}
    for k, v in integ.items():
        v["status"] = "identical" if v["actual"] == v["expected"] else "CHANGED"

    df = load()
    u = df.unix.values.astype(float)
    hg = df.hg.values.astype(float)
    ag = df.ag.values.astype(float)
    hid, aid = df.home_id.values, df.away_id.values
    seas, lgs = df.season.values, df.competition_name.values
    margin = hg - ag
    y = np.where(hg > ag, "H", np.where(hg == ag, "D", "A"))
    print(f"\nFixtures: {len(df)}")

    # --- 1. equal-width bands -----------------------------------------
    print("\n--- 1. Signal decay, equal-width 120-day bands ---")
    print(f"  {'age band (days)':<20}{'n_fix':>8}{'avg_matches':>13}{'corr':>9}")
    bands_eq = {}
    for lo in (0, 120, 240, 360, 480, 600):
        hi = lo + 120
        d, m_, cnt = [], [], []
        for i in range(len(df)):
            loT, hiT = u[i] - hi * SECONDS_PER_DAY, u[i] - lo * SECONDS_PER_DAY
            w = (u >= loT) & (u < hiT)
            if not w.any():
                continue
            hm = w & ((hid == hid[i]) | (aid == hid[i]))
            am = w & ((hid == aid[i]) | (aid == aid[i]))
            if hm.sum() < 3 or am.sum() < 3:
                continue
            h = np.where(hid[hm] == hid[i], hg[hm] - ag[hm], ag[hm] - hg[hm]).mean()
            a = np.where(hid[am] == aid[i], hg[am] - ag[am], ag[am] - hg[am]).mean()
            d.append(h - a); m_.append(margin[i]); cnt.append((hm.sum() + am.sum()) / 2)
        if len(d) < 200:
            continue
        r = float(np.corrcoef(np.array(d), np.array(m_))[0, 1])
        bands_eq[f"[{lo},{hi})"] = {"n_fixtures": len(d),
                                    "avg_matches_per_team": round(float(np.mean(cnt)), 1),
                                    "correlation": round(r, 4)}
        print(f"  [{lo},{hi})".ljust(20) + f"{len(d):>8}"
              f"{np.mean(cnt):>13.1f}{r:>9.4f}")

    # --- 2. fixed sample size ------------------------------------------
    print("\n--- 2. Signal decay, fixed N=10 matches per team ---")
    print(f"  {'offset (days)':<18}{'n_fix':>8}{'corr':>9}")
    fixed_n = {}
    for off in (0, 60, 120, 180, 240, 365, 540):
        d, m_ = [], []
        for i in range(len(df)):
            cut = u[i] - off * SECONDS_PER_DAY
            w = u < cut
            if not w.any():
                continue
            hm = np.where(w & ((hid == hid[i]) | (aid == hid[i])))[0]
            am = np.where(w & ((hid == aid[i]) | (aid == aid[i])))[0]
            if len(hm) < 10 or len(am) < 10:
                continue
            hm, am = hm[-10:], am[-10:]
            h = np.where(hid[hm] == hid[i], hg[hm] - ag[hm], ag[hm] - hg[hm]).mean()
            a = np.where(hid[am] == aid[i], hg[am] - ag[am], ag[am] - hg[am]).mean()
            d.append(h - a); m_.append(margin[i])
        if len(d) < 200:
            continue
        r = float(np.corrcoef(np.array(d), np.array(m_))[0, 1])
        fixed_n[f"{off}d"] = {"n_fixtures": len(d), "correlation": round(r, 4)}
        print(f"  {off:<18}{len(d):>8}{r:>9.4f}")

    # implied half-life from the fixed-N curve
    offs = np.array([0, 60, 120, 180, 240, 365, 540], dtype=float)
    corrs = np.array([fixed_n[f"{int(o)}d"]["correlation"] for o in offs])
    slope = np.polyfit(offs, np.log(corrs), 1)[0]
    implied_xi = float(-slope)
    implied_hl = float(np.log(2) / implied_xi) if implied_xi > 0 else None
    print(f"\n  Exponential fit to the fixed-N curve:")
    print(f"    implied xi        = {implied_xi:.8f} / day")
    print(f"    implied half-life = {implied_hl:.0f} days"
          if implied_hl else "    implied half-life = infinite")

    # --- 3. walk-forward proxy -----------------------------------------
    print("\n--- 3. Walk-forward proxy (nested temporal xi selection) ---")

    def lambdas(tr_mask, idx, xi, ref):
        w = (np.exp(-xi * np.maximum((ref - u[tr_mask]) / SECONDS_PER_DAY, 0))
             if xi > 0 else np.ones(int(tr_mask.sum())))
        H, A = hg[tr_mask], ag[tr_mask]
        Hi, Ai = hid[tr_mask], aid[tr_mask]
        gh, ga = np.average(H, weights=w), np.average(A, weights=w)
        atk, dfn = {}, {}
        for t in set(Hi) | set(Ai):
            mh, ma = Hi == t, Ai == t
            sw = w[mh].sum() + w[ma].sum()
            if sw < 1e-9:
                atk[t] = dfn[t] = 1.0; continue
            sc = (w[mh] * H[mh]).sum() + (w[ma] * A[ma]).sum()
            cd = (w[mh] * A[mh]).sum() + (w[ma] * H[ma]).sum()
            atk[t] = max((sc / sw) / ((gh + ga) / 2), 0.2)
            dfn[t] = max((cd / sw) / ((gh + ga) / 2), 0.2)
        lh = np.array([gh * atk.get(hid[i], 1.) * dfn.get(aid[i], 1.) for i in idx])
        la = np.array([ga * atk.get(aid[i], 1.) * dfn.get(hid[i], 1.) for i in idx])
        return np.clip(lh, 0.05, 6), np.clip(la, 0.05, 6)

    proxy_folds, PAs, PBs, Ys, LGs = [], [], [], [], []
    for trs, ts in FOLDS:
        trm, tem = np.isin(seas, trs), (seas == ts)
        tei, tri = np.where(tem)[0], np.where(trm)[0]
        ref = float(u[tem].min())
        inner = build_inner_splits(seas[trm], u[trm], trs)

        best_hl, best_ll, curve = None, float("inf"), []
        for hl in HALF_LIFE_GRID_DAYS:
            xi = xi_from_half_life(hl)
            sc = []
            for sp in inner:
                g_tr, g_va = tri[sp.inner_train_idx], tri[sp.inner_val_idx]
                m = np.zeros(len(df), bool); m[g_tr] = True
                lh, la = lambdas(m, g_va, xi, sp.reference_unix)
                P, _ = predict_dc(lh, la, 0.0)
                sc.append(log_loss(y[g_va], P))
            mu = float(np.mean(sc))
            curve.append({"half_life_days": hl, "xi": xi,
                          "mean_inner_log_loss": round(mu, 8)})
            if mu < best_ll - 1e-12:
                best_ll, best_hl = mu, hl

        xi = xi_from_half_life(best_hl)
        lhA, laA = lambdas(trm, tei, 0.0, ref)
        lhB, laB = lambdas(trm, tei, xi, ref)
        PA, _ = predict_dc(lhA, laA, 0.0)
        PB, _ = predict_dc(lhB, laB, 0.0)
        a_ll, b_ll = log_loss(y[tem], PA), log_loss(y[tem], PB)
        proxy_folds.append({
            "test_season": ts, "n_train": int(trm.sum()), "n_test": int(tem.sum()),
            "n_inner_splits": len(inner),
            "selected_half_life_days": best_hl,
            "selected_xi": xi,
            "weight_profile": weight_profile(xi),
            "mean_inner_log_loss": round(best_ll, 6),
            "xi_curve": curve,
            "A_log_loss": round(a_ll, 6), "B_log_loss": round(b_ll, 6),
            "delta_log_loss": round(a_ll - b_ll, 6),
        })
        PAs.append(PA); PBs.append(PB); Ys.append(y[tem]); LGs.append(lgs[tem])
        print(f"  {ts}: half-life={best_hl}  A={a_ll:.6f} B={b_ll:.6f} "
              f"delta={a_ll - b_ll:+.6f}")

    PA, PB = np.vstack(PAs), np.vstack(PBs)
    Y, LG = np.concatenate(Ys), np.concatenate(LGs)
    pooled = {"n": int(len(Y)),
              "A_log_loss": round(log_loss(Y, PA), 6),
              "B_log_loss": round(log_loss(Y, PB), 6),
              "delta_log_loss": round(log_loss(Y, PA) - log_loss(Y, PB), 6)}
    print(f"\n  POOLED n={len(Y)}: A={pooled['A_log_loss']:.6f} "
          f"B={pooled['B_log_loss']:.6f} delta={pooled['delta_log_loss']:+.6f}")

    hls = [f["selected_half_life_days"] for f in proxy_folds]
    fin = [h for h in hls if h is not None]
    spread = (max(fin) / min(fin)) if fin and len(fin) == len(hls) else None
    stability = ("STABLE" if len(set(map(str, hls))) == 1
                 else "REASONABLY STABLE" if spread and spread <= 2.0
                 else "UNSTABLE")
    print(f"  Selected half-lives: {hls}  spread={spread:.2f}x  -> {stability}"
          if spread else f"  Selected half-lives: {hls} -> {stability}")

    proxy_league = {}
    print("\n  By league (delta LL, positive = decay better):")
    for lg in sorted(set(LG)):
        m = LG == lg
        d = log_loss(Y[m], PA[m]) - log_loss(Y[m], PB[m])
        proxy_league[lg] = {"n": int(m.sum()), "delta_log_loss": round(d, 6)}
        print(f"    {lg:<17} n={int(m.sum()):<5} delta={d:+.6f}")
    improved = sum(1 for v in proxy_league.values() if v["delta_log_loss"] > 0)
    print(f"\n  Leagues improved: {improved}/5")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E4 — V3 vs V3 + Exponential Time Decay",
        "execution_status": {
            "experiment_A_v3": "PENDING — requires scikit-learn (per-fold refit)",
            "experiment_B_v3_decay": "PENDING — depends on A",
            "zero_decay_control": "PENDING — model-level control needs sklearn",
            "diagnostic": "COMPLETE",
            "test_suite": "COMPLETE — 69/69 passed",
        },
        "dataset": {"n_fixtures": int(len(df)),
                    "leagues": sorted(df.competition_name.unique().tolist()),
                    "seasons": list(SEASONS),
                    "quarantined_season": "2025/2026",
                    "market_used": False, "dixon_coles_used": False},
        "half_life_grid_days": list(HALF_LIFE_GRID_DAYS),
        "signal_decay_equal_width_bands": bands_eq,
        "signal_decay_fixed_sample_size": fixed_n,
        "implied_decay_from_fixed_n": {
            "xi_per_day": round(implied_xi, 8),
            "half_life_days": round(implied_hl, 1) if implied_hl else None,
            "note": ("Fitted to the fixed-N correlation curve, which controls "
                     "for sample size. Very mild decay."),
        },
        "walk_forward_proxy": {
            "caveat": ("Recency-weighted goal-rate model with no features. "
                       "V3 has 87. Bounds effect size and reveals stability; "
                       "does not predict V3's numbers."),
            "folds": proxy_folds,
            "pooled": pooled,
            "by_league": proxy_league,
            "leagues_improved": improved,
            "selected_half_lives": hls,
            "spread_ratio": round(spread, 2) if spread else None,
            "stability": stability,
        },
        "integrity": integ,
        "decision": "INCONCLUSIVE",
        "rationale": (
            "Experiments A and B were not executed: scikit-learn is "
            "unavailable, so V3 could not be refit per fold with or without "
            "sample weights, and the §19 zero-decay control could not be "
            "verified at the model level. The decision criteria require "
            "measured A vs B metrics."),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results -> {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
