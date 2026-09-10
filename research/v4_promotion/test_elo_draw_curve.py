r"""Pre-evaluation behavioural test suite for Phase 3 Elo -> Empirical Draw Curve Research.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python research/v4_promotion/test_elo_draw_curve.py

MUST pass with 0 failures before running elo_draw_curve_experiment.py.
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

from features.elo import (
    ELO_COLUMNS, HOME_ADVANTAGE, INIT_RATING, K_FACTOR,
    compute_elo_features, load_elo_features,
)

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
FROZEN_IDS_300 = HERE / "fresh_extended_fixture_ids.json"
EXT_MARKET_DB = HERE / "fresh_extended_market_odds.sqlite"
DC_FROZEN_METHOD = HERE / "dixon_coles_rho_method_frozen.json"

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
    "research/v4_promotion/dixon_coles_rho_method_frozen.json": "822e742dcc82e5e96445b31c14c0c604",
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


# --- Suite 2: Causal Elo Properties & Invariance ---
def suite2() -> Suite:
    s = Suite("2. Causal Elo Properties & Invariance (features.elo)")
    s.check(INIT_RATING == 1500.0, "Elo init rating is 1500.0", f"val={INIT_RATING}")
    s.check(K_FACTOR == 20.0, "Elo K factor is 20.0", f"val={K_FACTOR}")
    s.check(HOME_ADVANTAGE == 100.0, "Elo home advantage is 100.0", f"val={HOME_ADVANTAGE}")

    elo_df = load_elo_features(MATCHES_DB)
    s.check(len(elo_df) > 9000, "Elo features computed for all DB fixtures", f"n={len(elo_df)}")
    s.check(set(elo_df.columns) == {"fixture_id", "home_elo", "away_elo", "elo_diff"},
            "Exact V3 Elo contract columns present", str(list(elo_df.columns)))
    s.check(bool(elo_df.home_elo.notna().all() and elo_df.away_elo.notna().all()), "Zero nulls in Elo ratings")
    
    # Test elo_diff definition: (home_elo + 100.0) - away_elo
    diff_check = np.isclose(elo_df.elo_diff.values, (elo_df.home_elo.values + 100.0) - elo_df.away_elo.values, atol=1e-9)
    s.check(bool(diff_check.all()), "elo_diff strictly equals (home_elo + 100.0) - away_elo")

    # Adversarial test: changing a fixture's own result does not change its pre-match Elo
    conn = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    df_raw = pd.read_sql_query(
        "SELECT fixture_id, unix, home_id, away_id, home_goals, away_goals, status "
        "FROM fixtures ORDER BY unix ASC, fixture_id ASC LIMIT 500",
        conn
    )
    conn.close()

    base_elo = compute_elo_features(df_raw).set_index("fixture_id")
    
    # Mutate fixture index 250 outcome to 9-0
    df_mut = df_raw.copy()
    target_fid = df_mut.iloc[250].fixture_id
    df_mut.loc[df_mut.fixture_id == target_fid, "home_goals"] = 9
    df_mut.loc[df_mut.fixture_id == target_fid, "away_goals"] = 0
    mut_elo = compute_elo_features(df_mut).set_index("fixture_id")

    own_invariant = (
        np.isclose(base_elo.loc[target_fid, "home_elo"], mut_elo.loc[target_fid, "home_elo"]) and
        np.isclose(base_elo.loc[target_fid, "away_elo"], mut_elo.loc[target_fid, "away_elo"]) and
        np.isclose(base_elo.loc[target_fid, "elo_diff"], mut_elo.loc[target_fid, "elo_diff"])
    )
    s.check(own_invariant, "Target fixture's pre-match Elo is 100% immune to its own outcome rewrite")

    # Prior fixtures invariant
    prior_fids = df_raw.iloc[:250].fixture_id.values
    prior_diff = np.abs(base_elo.loc[prior_fids, "elo_diff"].values - mut_elo.loc[prior_fids, "elo_diff"].values).max()
    s.check(prior_diff < 1e-9, "All strictly prior fixtures immune to subsequent outcome rewrites", f"max_diff={prior_diff}")

    return s


# --- Suite 3: Empirical Binning & Mathematical Properties ---
def suite3() -> Suite:
    s = Suite("3. Empirical Binning & Statistical Calculations")

    # Wilson score interval test
    def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
        if n == 0: return 0.0, 0.0
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        delta = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return max(0.0, center - delta), min(1.0, center + delta)

    w_lo, w_hi = wilson_ci(25, 100)
    s.check(0.15 < w_lo < 0.20 and 0.30 < w_hi < 0.38, "Wilson interval accurately computed for 25/100",
            f"ci=[{w_lo:.4f}, {w_hi:.4f}]")

    # Logistic curve monotonic decrease test
    from scipy.optimize import minimize
    x_synth = np.array([10, 40, 80, 120, 180, 240, 300])
    y_synth = np.array([1, 1, 0, 1, 0, 0, 0])  # draw rate declines with x

    def neg_loglik_logis(params):
        b0, b1 = params
        z = b0 + b1 * x_synth
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return -np.sum(y_synth * np.log(p) + (1 - y_synth) * np.log(1 - p))

    res = minimize(neg_loglik_logis, [0.0, -0.005], method="L-BFGS-B")
    b0_fit, b1_fit = res.x
    s.check(b1_fit < 0, "Logistic regression yields negative coefficient on strength imbalance",
            f"b0={b0_fit:.4f}, b1={b1_fit:.6f}")

    return s


# --- Suite 4: 3-Way Simplex Redistribution & Normalization ---
def suite4() -> Suite:
    s = Suite("4. 3-Way Simplex Redistribution & Normalization")

    # Test proportional redistribution rule:
    # Given V4 P = [p_H, p_D, p_A] and new draw probability p_D_new
    # p_H_new = p_H * (1 - p_D_new) / (1 - p_D)
    # p_A_new = p_A * (1 - p_D_new) / (1 - p_D)
    p_orig = np.array([[0.45, 0.22, 0.33], [0.60, 0.20, 0.20]])
    p_d_new = np.array([0.27, 0.25])

    ratio = (1.0 - p_d_new) / (1.0 - p_orig[:, 1])
    p_h_new = p_orig[:, 0] * ratio
    p_a_new = p_orig[:, 2] * ratio
    p_new = np.column_stack([p_h_new, p_d_new, p_a_new])

    s.check(bool((p_new >= 0).all()), "All redistributed probabilities non-negative")
    s.check(bool(np.isclose(p_new.sum(axis=1), 1.0, atol=1e-12).all()), "Redistributed rows sum exactly to 1.0",
            f"max_dev={np.abs(p_new.sum(axis=1) - 1.0).max():.2e}")
    
    # Conditional odds preservation: p_H / p_A must be invariant
    orig_odds = p_orig[:, 0] / p_orig[:, 2]
    new_odds = p_new[:, 0] / p_new[:, 2]
    s.check(bool(np.isclose(orig_odds, new_odds, atol=1e-12).all()),
            "Conditional Home/Away relative odds perfectly preserved under draw redistribution")

    return s


# --- Suite 5: OOS Gate Isolation & Frozen Artifact Integrity ---
def suite5() -> Suite:
    s = Suite("5. OOS Gate Isolation & Frozen Artifact Integrity")
    s.check(FROZEN_IDS_300.exists(), "Frozen 300 OOS fixture file exists")
    s.check(EXT_MARKET_DB.exists(), "Frozen 300 OOS market DB exists")
    s.check(DC_FROZEN_METHOD.exists(), "Phase 2 Dixon-Coles frozen methodology exists")
    
    ids_300 = json.loads(FROZEN_IDS_300.read_text(encoding="utf-8"))["fixture_ids"]
    s.check(len(ids_300) == 300, "300 OOS fixtures confirmed", f"n={len(ids_300)}")
    return s


def main() -> int:
    print("=" * 78)
    print("PHASE 3 ELO -> EMPIRICAL DRAW CURVE — PRE-EVALUATION TEST SUITE")
    print("=" * 78)
    suites = [suite1(), suite2(), suite3(), suite4(), suite5()]
    for x in suites:
        x.report()

    tp = sum(x.p for x in suites)
    tf = sum(x.f for x in suites)
    ts = sum(x.s for x in suites)

    print("\n" + "=" * 78 + "\nOVERALL PRE-EVALUATION TEST SUMMARY\n" + "=" * 78)
    print(f"  Suites: {len(suites)}\n  Pass:   {tp}\n  Fail:   {tf}\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL PRE-EVALUATION TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    if tf > 0:
        print("\n  STOP — do not run elo_draw_curve_experiment.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
