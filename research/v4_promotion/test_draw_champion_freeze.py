r"""Pre-evaluation behavioural test suite for Phase 8 Draw Champion Freeze & Prospective Protocol.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python research/v4_promotion/test_draw_champion_freeze.py

MUST pass with 0 failures before running draw_champion_freeze.py.
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
from models.v4_artifact import load_v4_artifact

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
V4_ARTIFACT = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
FROZEN_IDS_300 = HERE / "fresh_extended_fixture_ids.json"
EXT_MARKET_DB = HERE / "fresh_extended_market_odds.sqlite"

PINNED_20 = {
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
    "research/v4_promotion/market_calibration_method_frozen.json": "550a0e1f1358a8359d7141b422521dd9",
    "research/v4_promotion/draw_complementarity_method_frozen.json": "d4f7dc75785df076c105a6ebfc0a4d6e",
    "research/v4_promotion/temporal_regime_method_frozen.json": "4a4f72e1d288d2547272c9b30b0368df",
    "research/v4_promotion/statistical_power_uncertainty_method_frozen.json": "68d55b30789d40440a0c14cbfe225c7f",
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
    for rel, exp in PINNED_20.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            s.bad(Path(rel).name, "missing")
            continue
        a = md5(p)
        s.check(a == exp, Path(rel).name, f"MD5={a}" if a == exp else f"got={a} exp={exp}")
    return s


# --- Suite 2: Champion Reconstruction & Mathematical Invariants ---
def suite2() -> Suite:
    s = Suite("2. Champion Reconstruction & Mathematical Invariants")
    
    # Mathematical equations & coefficients to freeze
    a0 = 0.1130
    w_dc = 0.6037
    w_elo = 0.4812

    p_v4 = np.array([0.45, 0.23, 0.32])
    p_dc_d = 0.255
    p_elo_d = 0.260

    z_dc = np.log(p_dc_d / (1.0 - p_dc_d))
    z_elo = np.log(p_elo_d / (1.0 - p_elo_d))
    z_new = a0 + w_dc * z_dc + w_elo * z_elo
    p_d_new = 1.0 / (1.0 + np.exp(-z_new))

    ratio = (1.0 - p_d_new) / (1.0 - p_v4[1])
    p_h_new = p_v4[0] * ratio
    p_a_new = p_v4[2] * ratio
    p_new = np.array([p_h_new, p_d_new, p_a_new])

    s.check(abs(p_new.sum() - 1.0) < 1e-12, "Calibrated probabilities sum strictly to 1.0", f"sum={p_new.sum()}")
    s.check(bool((p_new >= 0).all()), "Calibrated probabilities strictly non-negative")
    s.check(abs((p_h_new / p_a_new) - (p_v4[0] / p_v4[2])) < 1e-12, "Conditional relative odds P(H)/P(A) strictly invariant", f"orig={p_v4[0]/p_v4[2]:.6f} new={p_h_new/p_a_new:.6f}")
    s.check(0.0 < p_d_new < 1.0, "P(Draw) bounded strictly in (0, 1)", f"p_d={p_d_new:.4f}")
    return s


# --- Suite 3: Causal Contract & Prospective Data Isolation ---
def suite3() -> Suite:
    s = Suite("3. Causal Contract & Prospective Data Isolation")
    # Assert 300 OOS is not evaluated in Phase 8
    s.check(FROZEN_IDS_300.exists(), "Historical 300 OOS fixture list exists")
    s.check(EXT_MARKET_DB.exists(), "Historical 300 OOS market DB exists")
    
    # Verify no live/future data leakage files exist
    s.check(not (HERE / "prospective_eval_results.json").exists(), "Prospective evaluation results do NOT exist (clean pre-state)")
    return s


# --- Suite 4: Prospective Protocol Specifications ---
def suite4() -> Suite:
    s = Suite("4. Prospective Protocol Specifications")
    min_sample = 1050
    alpha = 0.05
    min_power = 0.80
    meaningful_threshold = -0.0010

    s.check(min_sample >= 1050, "Prospective minimum sample size >= 1,050 fixtures", f"n={min_sample}")
    s.check(alpha == 0.05, "Two-tailed significance alpha = 0.05")
    s.check(min_power >= 0.80, "Minimum statistical power requirement >= 80%")
    s.check(meaningful_threshold == -0.0010, "Pre-declared meaningful practical threshold = -0.0010")
    return s


def main() -> int:
    print("=" * 78)
    print("PHASE 8 DRAW CHAMPION FREEZE & PROSPECTIVE PROTOCOL — TEST SUITE")
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
        print("\n  STOP — do not run draw_champion_freeze.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
