r"""Pre-evaluation behavioural test suite for Phase 5 Market & Probability Calibration Research.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python research/v4_promotion/test_market_calibration.py

MUST pass with 0 failures before running market_calibration_experiment.py.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from dixon_coles import CLASS_ORDER, predict_dc

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
V4_ARTIFACT = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
FROZEN_IDS_300 = HERE / "fresh_extended_fixture_ids.json"
EXT_MARKET_DB = HERE / "fresh_extended_market_odds.sqlite"
DC_FROZEN_METHOD = HERE / "dixon_coles_rho_method_frozen.json"
ELO_FROZEN_METHOD = HERE / "elo_draw_curve_method_frozen.json"
MATRIX_FROZEN_METHOD = HERE / "full_score_matrix_method_frozen.json"

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
    "research/v4_promotion/elo_draw_curve_method_frozen.json": "65dc2cf762f3d78abcf1a617ef23fe00",
    "research/v4_promotion/full_score_matrix_method_frozen.json": "cd44e1da88a50ac45e8383557ad5271f",
}


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


# --- Suite 2: Temperature Scaling & Softmax Math ---
def suite2() -> Suite:
    s = Suite("2. Temperature Scaling & Softmax Math")
    P = np.array([[0.60, 0.25, 0.15], [0.35, 0.35, 0.30]])
    logits = np.log(P)

    # T = 1.0 -> Exact Identity
    T_1 = 1.0
    P_t1 = np.exp(logits / T_1) / np.sum(np.exp(logits / T_1), axis=1, keepdims=True)
    dev_1 = np.abs(P_t1 - P).max()
    s.check(dev_1 < 1e-12, "T=1.0 preserves exact input probabilities", f"max_dev={dev_1:.2e}")

    # T -> Large (e.g. 1000) -> Uniform Distribution (1/3, 1/3, 1/3)
    T_inf = 1000.0
    P_inf = np.exp(logits / T_inf) / np.sum(np.exp(logits / T_inf), axis=1, keepdims=True)
    dev_inf = np.abs(P_inf - (1.0 / 3.0)).max()
    s.check(dev_inf < 1e-3, "T -> inf approaches uniform distribution", f"max_dev={dev_inf:.2e}")

    # All sum to 1.0 and non-negative
    s.check(bool((P_t1 >= 0).all()), "Probabilities non-negative")
    s.check(bool((P_t1 <= 1).all()), "Probabilities bounded <= 1.0")
    s.check(np.abs(P_t1.sum(axis=1) - 1.0).max() < 1e-12, "Rows sum strictly to 1.0")
    return s


# --- Suite 3: Draw-Only Calibration & Proportional Simplex Invariance ---
def suite3() -> Suite:
    s = Suite("3. Draw-Only Calibration & Proportional Simplex Invariance")
    P_orig = np.array([[0.55, 0.25, 0.20], [0.30, 0.20, 0.50], [0.45, 0.30, 0.25]])
    p_d_new = np.array([0.28, 0.24, 0.32])

    from elo_draw_curve_experiment import redistribute_draw_mass
    P_cal = redistribute_draw_mass(P_orig, p_d_new)

    # 1. Simplex Sum to 1.0
    sums = P_cal.sum(axis=1)
    s.check(np.abs(sums - 1.0).max() < 1e-12, "Redistributed rows sum exactly to 1.0")

    # 2. Exact P(D) match
    dev_pd = np.abs(P_cal[:, 1] - p_d_new).max()
    s.check(dev_pd < 1e-12, "Calibrated P(D) matches target exactly", f"max_dev={dev_pd:.2e}")

    # 3. Invariance of conditional relative odds P(H)/P(A)
    odds_orig = P_orig[:, 0] / P_orig[:, 2]
    odds_cal = P_cal[:, 0] / P_cal[:, 2]
    dev_odds = np.abs(odds_orig - odds_cal).max()
    s.check(dev_odds < 1e-12, "Home/Away relative odds strictly invariant under draw calibration", f"max_dev={dev_odds:.2e}")

    return s


# --- Suite 4: Market Reference Isolation & OOS Gate Safety ---
def suite4() -> Suite:
    s = Suite("4. Market Reference Isolation & OOS Gate Safety")
    s.check(EXT_MARKET_DB.exists(), "Frozen 300 market database exists")
    s.check(FROZEN_IDS_300.exists(), "Frozen 300 fixture list exists")

    conn = sqlite3.connect(f"file:{EXT_MARKET_DB}?mode=ro", uri=True)
    df_mkt = pd.read_sql_query("SELECT * FROM fresh_extended_market", conn)
    conn.close()

    s.check(len(df_mkt) == 300, "Market reference contains exactly 300 rows", f"n={len(df_mkt)}")
    s.check("outcome" not in df_mkt.columns, "Market table carries zero outcome columns")
    s.check(bool((df_mkt[["p_home", "p_draw", "p_away"]].values >= 0).all()), "Market probabilities non-negative")
    s.check(np.abs(df_mkt[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1.0).max() < 1e-12, "Market probabilities sum to 1.0")

    # Check prior frozen methodologies
    s.check(DC_FROZEN_METHOD.exists(), "Phase 2 DC frozen methodology exists")
    s.check(ELO_FROZEN_METHOD.exists(), "Phase 3 Elo frozen methodology exists")
    s.check(MATRIX_FROZEN_METHOD.exists(), "Phase 4 Score-Matrix frozen methodology exists")

    return s


def main() -> int:
    print("=" * 78)
    print("PHASE 5 MARKET / PROBABILITY CALIBRATION — PRE-EVALUATION TEST SUITE")
    print("=" * 78)
    suites = [suite1(), suite2(), suite3(), suite4()]
    for x in suites:
        x.report()

    tp = sum(x.p for x in suites)
    tf = sum(x.f for x in suites)
    ts = sum(x.s for x in suites)

    print("\n" + "=" * 78 + "\nOVERALL PRE-EVALUATION TEST SUMMARY\n" + "=" * 78)
    print(f"  Suites: {len(suites)}\n  Pass:   {tp}\n  Fail:   {tf}\n  Skip:   {ts}\n  Total:  {tp + tf + ts}")
    print(f"\n  VERDICT: {'ALL PRE-EVALUATION TESTS PASSED' if tf == 0 else 'TESTS FAILED'}")
    if tf > 0:
        print("\n  STOP — do not run market_calibration_experiment.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
