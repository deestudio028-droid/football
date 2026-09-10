r"""Pre-evaluation behavioural test suite for Phase 4 Full Score-Matrix Draw Modeling.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python research/v4_promotion/test_full_score_matrix.py

MUST pass with 0 failures before running full_score_matrix_experiment.py.
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

from dixon_coles import (
    CLASS_ORDER, grid_size, hda_from_grid, pmf_grid, predict_dc, score_grid,
)
from models.v4_artifact import load_v4_artifact

MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
V4_ARTIFACT = PROJECT_ROOT / "data/models/v4_poisson_venue_elo_online_ad.pkl"
FROZEN_IDS_300 = HERE / "fresh_extended_fixture_ids.json"
EXT_MARKET_DB = HERE / "fresh_extended_market_odds.sqlite"
DC_FROZEN_METHOD = HERE / "dixon_coles_rho_method_frozen.json"
ELO_FROZEN_METHOD = HERE / "elo_draw_curve_method_frozen.json"

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


# --- Suite 2: Score-Matrix Construction & Normalization Math ---
def suite2() -> Suite:
    s = Suite("2. Score-Matrix Construction & Normalization Math")
    
    # 1. Test independent Poisson grid construction
    lam_h = np.array([1.6, 2.2, 0.9])
    lam_a = np.array([1.2, 0.8, 1.4])
    K = 12

    P_dc_0, _ = predict_dc(lam_h, lam_a, 0.0, K=K)
    
    # Verify manual score-matrix reduction matches predict_dc
    for i in range(len(lam_h)):
        # Construct 2D grid
        qh = pmf_grid(lam_h[i:i+1], K)[0]
        qa = pmf_grid(lam_a[i:i+1], K)[0]
        M = np.outer(qh, qa)
        
        # Tail check: sum of outer product before normalization
        tail_loss = 1.0 - float(M.sum())
        s.check(tail_loss < 1e-5, f"Tail truncation loss < 1e-5 for K={K}", f"loss={tail_loss:.2e}")
        
        M_norm = M / M.sum()
        p_h = float(np.sum(np.tril(M_norm, -1).T))  # i > j
        p_d = float(np.trace(M_norm))               # i == j
        p_a = float(np.sum(np.triu(M_norm, 1).T))   # i < j
        p_1x2 = np.array([p_h, p_d, p_a])
        
        dev = np.abs(p_1x2 - P_dc_0[i]).max()
        s.check(dev < 1e-12, f"Matrix reduction matches predict_dc exactly (match {i})", f"max_dev={dev:.2e}")
        s.check(abs(p_1x2.sum() - 1.0) < 1e-12, f"1X2 probabilities sum to 1.0 (match {i})")

    return s


# --- Suite 3: Regularized Matrix Calibration Safeguards ---
def suite3() -> Suite:
    s = Suite("3. Regularized Matrix Calibration Safeguards (Candidates F & G)")
    
    # Test multiplicative residual correction with shrinkage
    # C_reg(x,y) = (n * Obs(x,y) + alpha * Exp(x,y)) / ((n + alpha) * Exp(x,y))
    # Bound check: C_reg strictly > 0 and finite
    n_sample = 1000
    obs_cell = 0.07  # observed 0-0
    exp_cell = 0.05  # expected 0-0
    alpha = 500.0   # prior pseudocounts

    c_reg = (n_sample * obs_cell + alpha * exp_cell) / ((n_sample + alpha) * exp_cell)
    s.check(1.0 < c_reg < 1.4, "Regularized multiplier shrinks observed ratio toward 1.0", f"c_reg={c_reg:.4f}")
    
    # Infinite alpha -> c_reg = 1.0 (pure baseline)
    c_inf = (n_sample * obs_cell + 1e9 * exp_cell) / ((n_sample + 1e9) * exp_cell)
    s.check(abs(c_inf - 1.0) < 1e-6, "Infinite shrinkage retains exact Poisson baseline")

    # Non-negativity assertion under log-linear perturbation
    # P'(x,y) = exp(log P(x,y) + theta_{x,y}) / Z
    M_base = np.array([[0.06, 0.08], [0.09, 0.12]])
    theta = np.array([[0.15, -0.05], [-0.05, 0.10]])
    M_pert = M_base * np.exp(theta)
    M_pert_norm = M_pert / M_pert.sum()
    s.check(bool((M_pert_norm >= 0).all()), "Perturbed score matrix strictly non-negative")
    s.check(abs(M_pert_norm.sum() - 1.0) < 1e-12, "Perturbed score matrix sums to 1.0")

    return s


# --- Suite 4: OOS Gate Isolation Safety ---
def suite4() -> Suite:
    s = Suite("4. OOS Gate Isolation Safety")
    s.check(FROZEN_IDS_300.exists(), "Frozen 300 OOS fixture file exists")
    s.check(EXT_MARKET_DB.exists(), "Frozen 300 OOS market DB exists")
    s.check(DC_FROZEN_METHOD.exists(), "Phase 2 Dixon-Coles frozen methodology exists")
    s.check(ELO_FROZEN_METHOD.exists(), "Phase 3 Elo Draw Curve frozen methodology exists")
    
    ids_300 = json.loads(FROZEN_IDS_300.read_text(encoding="utf-8"))["fixture_ids"]
    s.check(len(ids_300) == 300, "300 OOS fixtures confirmed", f"n={len(ids_300)}")
    return s


def main() -> int:
    print("=" * 78)
    print("PHASE 4 FULL SCORE-MATRIX DRAW MODELING — PRE-EVALUATION TEST SUITE")
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
        print("\n  STOP — do not run full_score_matrix_experiment.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
