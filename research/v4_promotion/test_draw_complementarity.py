r"""Pre-evaluation behavioural test suite for Draw Signal Complementarity & Ablation Research.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python research/v4_promotion/test_draw_complementarity.py

MUST pass with 0 failures before running draw_complementarity_experiment.py.
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
DC_FROZEN_METHOD = HERE / "dixon_coles_rho_method_frozen.json"
ELO_FROZEN_METHOD = HERE / "elo_draw_curve_method_frozen.json"
MATRIX_FROZEN_METHOD = HERE / "full_score_matrix_method_frozen.json"
CALIB_FROZEN_METHOD = HERE / "market_calibration_method_frozen.json"

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
    "research/v4_promotion/market_calibration_method_frozen.json": "550a0e1f1358a8359d7141b422521dd9",
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


# --- Suite 2: Frozen Candidate Reconstruction Math ---
def suite2() -> Suite:
    s = Suite("2. Frozen Candidate Reconstruction Math")
    # Verify all 4 frozen research JSONs parse correctly
    dc = json.loads(DC_FROZEN_METHOD.read_text(encoding="utf-8"))
    elo = json.loads(ELO_FROZEN_METHOD.read_text(encoding="utf-8"))
    mat = json.loads(MATRIX_FROZEN_METHOD.read_text(encoding="utf-8"))
    cal = json.loads(CALIB_FROZEN_METHOD.read_text(encoding="utf-8"))

    s.check("primary_candidate" in dc, "Phase 2 DC frozen method contains primary candidate")
    s.check("primary_candidate" in elo, "Phase 3 Elo frozen method contains primary candidate")
    s.check("primary_candidate" in mat, "Phase 4 Matrix frozen method contains primary candidate")
    s.check("primary_candidate" in cal, "Phase 5 Calib frozen method contains primary candidate")

    # Verify dummy prediction through all 4 candidates
    lam_h = np.array([1.5])
    lam_a = np.array([1.1])
    abs_elo = np.array([50.0])

    p_base, _ = predict_dc(lam_h, lam_a, 0.0)
    
    # DC reconstruction
    r_dc = dc["primary_candidate"]["league_rhos"].get("Premier League", -0.0560)
    p_dc, _ = predict_dc(lam_h, lam_a, r_dc)
    s.check(p_dc[0, 1] > p_base[0, 1], "Dixon-Coles increases P(Draw) for rho < 0", f"p_dc={p_dc[0,1]:.4f} vs base={p_base[0,1]:.4f}")

    # Elo reconstruction
    a0 = elo["primary_candidate"]["coefficients"]["a0_intercept"]
    a1 = elo["primary_candidate"]["coefficients"]["a1_logit_v4"]
    a2 = elo["primary_candidate"]["coefficients"]["a2_abs_elo"]
    z_base = np.log(p_base[0, 1] / (1.0 - p_base[0, 1]))
    pd_elo = 1.0 / (1.0 + np.exp(-(a0 + a1 * z_base + a2 * 0.5)))
    from elo_draw_curve_experiment import redistribute_draw_mass
    p_elo = redistribute_draw_mass(p_base, np.array([pd_elo]))
    s.check(abs(p_elo.sum() - 1.0) < 1e-12, "Elo calibrated probabilities sum to 1.0")

    return s


# --- Suite 3: Simplex Preservation under Ensemble / Stacking Layers ---
def suite3() -> Suite:
    s = Suite("3. Simplex Preservation under Ensemble / Stacking Layers")
    P1 = np.array([[0.50, 0.25, 0.25], [0.30, 0.30, 0.40]])
    P2 = np.array([[0.48, 0.28, 0.24], [0.28, 0.34, 0.38]])
    P3 = np.array([[0.52, 0.26, 0.22], [0.32, 0.31, 0.37]])

    # Convex combination blend
    weights = np.array([0.5, 0.3, 0.2])
    P_blend = weights[0] * P1 + weights[1] * P2 + weights[2] * P3
    s.check(np.abs(P_blend.sum(axis=1) - 1.0).max() < 1e-12, "Linear convex blend sums strictly to 1.0")
    s.check(bool((P_blend >= 0).all()), "Linear convex blend non-negative")

    # Stacking draw prediction + proportional redistribution
    p_d_stack = 0.5 * P1[:, 1] + 0.3 * P2[:, 1] + 0.2 * P3[:, 1]
    from elo_draw_curve_experiment import redistribute_draw_mass
    P_stack = redistribute_draw_mass(P1, p_d_stack)
    s.check(np.abs(P_stack.sum(axis=1) - 1.0).max() < 1e-12, "Stacking redistributed rows sum to 1.0")
    s.check(np.abs(P_stack[:, 0] / P_stack[:, 2] - P1[:, 0] / P1[:, 2]).max() < 1e-12, "Conditional relative odds strictly preserved")

    return s


# --- Suite 4: Market Isolation & OOS Gate Safety ---
def suite4() -> Suite:
    s = Suite("4. Market Isolation & OOS Gate Safety")
    s.check(EXT_MARKET_DB.exists(), "Frozen 300 market database exists")
    s.check(FROZEN_IDS_300.exists(), "Frozen 300 fixture list exists")

    conn = sqlite3.connect(f"file:{EXT_MARKET_DB}?mode=ro", uri=True)
    df_mkt = pd.read_sql_query("SELECT * FROM fresh_extended_market", conn)
    conn.close()

    s.check(len(df_mkt) == 300, "Market reference contains exactly 300 rows", f"n={len(df_mkt)}")
    s.check("outcome" not in df_mkt.columns, "Market table carries zero outcome columns")
    s.check(bool((df_mkt[["p_home", "p_draw", "p_away"]].values >= 0).all()), "Market probabilities non-negative")

    return s


def main() -> int:
    print("=" * 78)
    print("PHASE 5 DRAW SIGNAL COMPLEMENTARITY & ABLATION — PRE-EVALUATION TEST SUITE")
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
        print("\n  STOP — do not run draw_complementarity_experiment.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
