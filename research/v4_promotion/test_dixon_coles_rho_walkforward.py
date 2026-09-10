r"""Pre-evaluation behavioural test suite for Phase 2 Dixon-Coles Walk-Forward & Rho Estimation.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python research/v4_promotion/test_dixon_coles_rho_walkforward.py

MUST pass with 0 failures before running dixon_coles_rho_experiment.py.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from dixon_coles import (
    CLASS_ORDER, DixonColesError, grid_size, hda_from_grid, low_score_probs,
    pmf_grid, predict_dc, rho_validity_bounds, score_grid, tau_matrix,
)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
FROZEN_IDS_300 = HERE / "fresh_extended_fixture_ids.json"
EXT_MARKET_DB = HERE / "fresh_extended_market_odds.sqlite"

PINNED_MINIMUM = {
    "data/models/v2_poisson_venue.pkl": "25935b4e93fc4074f67f16e3181ed4df",
    "data/models/v1_logreg.pkl": "5e504427712b35778bb8a62a8496c7cd",
    "data/models/v3_poisson_venue_elo_candidate.pkl": "a2850a7687822a5916663301f5ccc96c",
    "data/models/v4_poisson_venue_elo_online_ad.pkl": "06841f0c03c8597b2b8cd8f8ab064864",
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
    "research/market_odds/odds_history.sqlite": "0be31e8b59d739b72c3fb48e555d9fd8",
    "research/market_odds/research_dataset.sqlite": "bdab370ffdfe5bbf8ff3a8a26e64471c",
    "research/v4_promotion/promotion_market_odds.sqlite": "f8a41b79cd33afb412ccd9ae2892a196",
    "research/v4_promotion/fresh_100_market_odds.sqlite": "2cb80b79d772fbedd4f3707b39a32c13",
    "research/v4_promotion/fresh_100_fixture_ids.json": "761ad5cc571643e6985e671bd9c3d83a",
    "research/v4_promotion/fresh_extended_fixture_ids.json": "0526bfd6980dd51dae59c6f6aadab2f5",
    "research/v4_promotion/fresh_extended_market_odds.sqlite": "b4889d1791723ea653057af51ca00f8e",
}

TARGET_COMPS = (200, 419, 423, 477, 499)
HIST_SEASONS = ("2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025")


class Suite:
    def __init__(self, name: str):
        self.name = name
        self.p, self.f, self.s = 0, 0, 0
        self.lines: list[str] = []

    def ok(self, msg: str, d: str = ""):
        self.p += 1
        self.lines.append(f"  PASS: {msg}" + (f" — {d}" if d else ""))

    def bad(self, msg: str, d: str = ""):
        self.f += 1
        self.lines.append(f"  FAIL: {msg}" + (f" — {d}" if d else ""))

    def skip(self, msg: str, d: str = ""):
        self.s += 1
        self.lines.append(f"  SKIP: {msg}" + (f" — {d}" if d else ""))

    def check(self, cond: bool, msg: str, d: str = ""):
        self.ok(msg, d) if cond else self.bad(msg, d)

    def report(self):
        print(f"\n{'=' * 78}\nSuite: {self.name}\n{'=' * 78}")
        for l in self.lines:
            print(l)
        print(f"\n  Pass: {self.p}  Fail: {self.f}  Skip: {self.s}")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


# --- Suite 1: Protected Assets Integrity ---
def suite1() -> Suite:
    s = Suite("1. Protected Assets & Manifest Integrity")
    for rel, exp in PINNED_MINIMUM.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            s.bad(Path(rel).name, "missing")
            continue
        a = md5(p)
        s.check(a == exp, Path(rel).name, f"MD5={a}" if a == exp else f"got={a} exp={exp}")
    return s


# --- Suite 2: Historical Dataset Integrity ---
def suite2() -> Suite:
    s = Suite("2. Historical Dataset Scope (2020/21–2024/25)")
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    phc = ",".join("?" * len(TARGET_COMPS))
    phs = ",".join("?" * len(HIST_SEASONS))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, date, competition_name, home_name, away_name,
                   home_goals, away_goals, status, unix, season
            FROM fixtures
            WHERE competition_id IN ({phc}) AND season IN ({phs})
            ORDER BY unix ASC, fixture_id ASC""",
        conn, params=list(TARGET_COMPS) + list(HIST_SEASONS)
    )
    # Check for 2025/26 fixtures
    n_2025 = conn.execute(
        f"""SELECT COUNT(*) FROM fixtures
            WHERE competition_id IN ({phc}) AND season = '2025/2026'""",
        list(TARGET_COMPS)
    ).fetchone()[0]
    conn.close()

    s.check(len(df) == 8983, "Historical fixture count is exactly 8,983", f"n={len(df)}")
    s.check(bool((df.status.isin(["FT", "AWARDED"])).all()), "All historical statuses FT or AWARDED")
    s.check(bool(df.home_goals.notna().all() and df.away_goals.notna().all()), "Zero missing goal labels")
    s.check(len(set(df.fixture_id)) == len(df), "Zero duplicate fixture IDs")
    s.check(set(df.season) == set(HIST_SEASONS), "Exactly seasons 2020/21 through 2024/25")
    s.check(n_2025 > 0, "2025/26 season present in DB but excluded from historical scope", f"n_2025={n_2025}")
    
    comp = Counter(df.competition_name)
    s.check(len(comp) == 5, "All 5 target competitions represented", str(dict(comp)))
    return s


# --- Suite 3: Walk-Forward Fold Temporal Purity ---
def suite3() -> Suite:
    s = Suite("3. Walk-Forward Fold Temporal Purity")
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    phc = ",".join("?" * len(TARGET_COMPS))
    df = pd.read_sql_query(
        f"""SELECT fixture_id, unix, season FROM fixtures
            WHERE competition_id IN ({phc})
            ORDER BY unix ASC, fixture_id ASC""",
        conn, params=list(TARGET_COMPS)
    )
    conn.close()

    folds = [
        (("2020/2021",), "2021/2022"),
        (("2020/2021", "2021/2022"), "2022/2023"),
        (("2020/2021", "2021/2022", "2022/2023"), "2023/2024"),
        (("2020/2021", "2021/2022", "2022/2023", "2023/2024"), "2024/2025"),
    ]

    all_pure = True
    for train_s, val_s in folds:
        tr_max = df[df.season.isin(train_s)].unix.max()
        val_min = df[df.season == val_s].unix.min()
        pure = (tr_max < val_min)
        all_pure &= pure
        s.check(pure, f"Fold train {train_s} -> val {val_s} strictly chronological",
                f"tr_max={tr_max} < val_min={val_min}")

    s.check(all_pure, "All 4 historical walk-forward folds strictly chronologically pure")
    return s


# --- Suite 4: Dixon-Coles Score & Fisher Information Math ---
def suite4() -> Suite:
    s = Suite("4. Dixon-Coles Bivariate Score & Fisher Information Math")
    
    # Test analytical likelihood, score and information
    from scipy.optimize import minimize_scalar
    
    # Synthetic rates
    lam_h = np.array([1.5, 1.8, 0.9, 2.1, 1.2])
    lam_a = np.array([1.1, 1.3, 1.4, 0.8, 1.2])
    hg = np.array([0, 1, 0, 1, 2])
    ag = np.array([0, 0, 1, 1, 0])
    w = np.ones(5)

    def loglik(r):
        tau = np.ones(len(hg))
        for i in range(len(hg)):
            if hg[i] == 0 and ag[i] == 0: tau[i] = 1 - lam_h[i]*lam_a[i]*r
            elif hg[i] == 1 and ag[i] == 0: tau[i] = 1 + lam_a[i]*r
            elif hg[i] == 0 and ag[i] == 1: tau[i] = 1 + lam_h[i]*r
            elif hg[i] == 1 and ag[i] == 1: tau[i] = 1 - r
        if np.any(tau <= 1e-9): return -1e9
        return np.sum(w * np.log(tau))

    def score(r):
        s_val = 0.0
        for i in range(len(hg)):
            if hg[i] == 0 and ag[i] == 0: s_val += -w[i] * lam_h[i] * lam_a[i] / (1 - lam_h[i]*lam_a[i]*r)
            elif hg[i] == 1 and ag[i] == 0: s_val += w[i] * lam_a[i] / (1 + lam_a[i]*r)
            elif hg[i] == 0 and ag[i] == 1: s_val += w[i] * lam_h[i] / (1 + lam_h[i]*r)
            elif hg[i] == 1 and ag[i] == 1: s_val += -w[i] / (1 - r)
        return s_val

    def fisher_info(r):
        inf_val = 0.0
        for i in range(len(hg)):
            if hg[i] == 0 and ag[i] == 0: inf_val += w[i] * (lam_h[i]*lam_a[i])**2 / (1 - lam_h[i]*lam_a[i]*r)**2
            elif hg[i] == 1 and ag[i] == 0: inf_val += w[i] * lam_a[i]**2 / (1 + lam_a[i]*r)**2
            elif hg[i] == 0 and ag[i] == 1: inf_val += w[i] * lam_h[i]**2 / (1 + lam_h[i]*r)**2
            elif hg[i] == 1 and ag[i] == 1: inf_val += w[i] / (1 - r)**2
        return inf_val

    # Test numerical gradient vs analytical score
    r_test = -0.05
    eps = 1e-6
    num_grad = (loglik(r_test + eps) - loglik(r_test - eps)) / (2 * eps)
    ana_score = score(r_test)
    s.check(abs(num_grad - ana_score) < 1e-5, "Analytical score matches numerical gradient",
            f"num={num_grad:.6f} ana={ana_score:.6f}")

    # Test numerical second derivative vs analytical Fisher information
    eps_h = 1e-4
    num_hess = -(loglik(r_test + eps_h) - 2 * loglik(r_test) + loglik(r_test - eps_h)) / (eps_h**2)
    ana_info = fisher_info(r_test)
    s.check(abs(num_hess - ana_info) < 1e-3, "Analytical Fisher information matches negative Hessian",
            f"num={num_hess:.6f} ana={ana_info:.6f}")
    s.check(ana_info > 0, "Fisher information strictly positive (strictly convex negative log-likelihood)")

    return s


# --- Suite 5: Empirical Bayes Shrinkage Properties ---
def suite5() -> Suite:
    s = Suite("5. Empirical Bayes Shrinkage Properties (Methods G & H)")
    
    # 5 leagues with sample estimates and variances
    rho_leagues = np.array([-0.11, -0.01, -0.06, 0.01, -0.04])
    var_leagues = np.array([0.0010, 0.0008, 0.0009, 0.0008, 0.0009])
    rho_global = -0.040

    # DerSimonian-Laird tau^2 estimator
    K = len(rho_leagues)
    raw_var = np.var(rho_leagues, ddof=1)
    mean_within_var = np.mean(var_leagues)
    tau_sq = max(1e-6, raw_var - mean_within_var)

    w = tau_sq / (tau_sq + var_leagues)
    rho_shrunk = w * rho_leagues + (1 - w) * rho_global

    s.check(bool(np.all(w >= 0.0) and np.all(w <= 1.0)), "Shrinkage weights bounded in [0, 1]",
            f"weights={list(np.round(w, 4))}")
    s.check(bool(np.all(np.abs(rho_shrunk - rho_global) <= np.abs(rho_leagues - rho_global) + 1e-9)),
            "Shrunk estimates strictly shrunk toward global rho")
    
    # Asymptotic test: infinite sample variance -> full shrinkage to global
    w_inf = tau_sq / (tau_sq + 1e9)
    shrunk_inf = w_inf * rho_leagues[0] + (1 - w_inf) * rho_global
    s.check(abs(shrunk_inf - rho_global) < 1e-6, "Infinite variance shrinks completely to global rho")

    # Asymptotic test: zero sample variance -> no shrinkage
    w_zero = tau_sq / (tau_sq + 0.0)
    shrunk_zero = w_zero * rho_leagues[0] + (1 - w_zero) * rho_global
    s.check(abs(shrunk_zero - rho_leagues[0]) < 1e-6, "Zero variance retains exact league estimate")

    return s


# --- Suite 6: OOS Gate Isolation Safety ---
def suite6() -> Suite:
    s = Suite("6. OOS Gate Isolation Safety")
    s.check(FROZEN_IDS_300.exists(), "300 OOS fixture file exists")
    s.check(EXT_MARKET_DB.exists(), "300 OOS market DB exists")
    
    # Verify that the test runner does NOT access the OOS outcomes
    ids_300 = json.loads(FROZEN_IDS_300.read_text(encoding="utf-8"))["fixture_ids"]
    s.check(len(ids_300) == 300, "300 OOS fixtures confirmed intact", f"n={len(ids_300)}")
    
    # Verify zero overlap between historical training and 300 OOS
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    phc = ",".join("?" * len(TARGET_COMPS))
    phs = ",".join("?" * len(HIST_SEASONS))
    hist_ids = set(r[0] for r in conn.execute(
        f"SELECT fixture_id FROM fixtures WHERE competition_id IN ({phc}) AND season IN ({phs})",
        list(TARGET_COMPS) + list(HIST_SEASONS)
    ).fetchall())
    conn.close()

    overlap = set(ids_300) & hist_ids
    s.check(len(overlap) == 0, "Zero overlap between historical training set and 300 OOS fixtures",
            f"overlap={len(overlap)}")
    return s


def main() -> int:
    print("=" * 78)
    print("PHASE 2 DIXON-COLES WALK-FORWARD & RHO ESTIMATION — PRE-EVALUATION TESTS")
    print("=" * 78)
    suites = [suite1(), suite2(), suite3(), suite4(), suite5(), suite6()]
    for x in suites:
        x.report()

    tp = sum(x.p for x in suites)
    tf = sum(x.f for x in suites)
    ts = sum(x.s for x in suites)

    print("\n" + "=" * 78 + "\nOVERALL PRE-EVALUATION TEST SUMMARY\n" + "=" * 78)
    print(f"  Suites: {len(suites)}\n  Pass:   {tp}\n  Fail:   {tf}\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL PRE-EVALUATION TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    if tf > 0:
        print("\n  STOP — do not run dixon_coles_rho_experiment.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
