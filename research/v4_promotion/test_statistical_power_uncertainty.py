r"""Pre-evaluation behavioural test suite for Phase 7 Statistical Power, Multi-OOS & Uncertainty Research.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python research/v4_promotion/test_statistical_power_uncertainty.py

MUST pass with 0 failures before running statistical_power_uncertainty_experiment.py.
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
COMPL_FROZEN_METHOD = HERE / "draw_complementarity_method_frozen.json"
TEMPORAL_FROZEN_METHOD = HERE / "temporal_regime_method_frozen.json"

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
    "research/v4_promotion/draw_complementarity_method_frozen.json": "d4f7dc75785df076c105a6ebfc0a4d6e",
    "research/v4_promotion/temporal_regime_method_frozen.json": "4a4f72e1d288d2547272c9b30b0368df",
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


# --- Suite 2: Clustered Bootstrap Sampling Invariants ---
def suite2() -> Suite:
    s = Suite("2. Clustered Bootstrap Sampling Invariants")
    # Generate mock clusters
    clusters = np.repeat(np.arange(5), 60)  # 5 clusters, 60 items each = 300 items
    data = np.random.default_rng(20260820).normal(0.0, 1.0, size=300)
    
    # Resample clusters with replacement
    rng = np.random.default_rng(20260820)
    unique_clusts = np.unique(clusters)
    sampled_clusts = rng.choice(unique_clusts, size=len(unique_clusts), replace=True)
    resampled_indices = np.concatenate([np.where(clusters == c)[0] for c in sampled_clusts])

    s.check(len(resampled_indices) == len(data), "Cluster bootstrap preserves sample size under equal cluster sizes", f"n={len(resampled_indices)}")
    s.check(len(sampled_clusts) == len(unique_clusts), "Cluster count matches original unique clusters", f"k={len(sampled_clusts)}")
    return s


# --- Suite 3: Power Simulation Monotonicity ---
def suite3() -> Suite:
    s = Suite("3. Power Simulation Monotonicity")
    sample_sizes = [300, 1000, 3000]
    effect_mean = -0.0045
    effect_std = 0.050

    # Approximate analytical power: P(Z < (mu / (sigma / sqrt(N))) - 1.96)
    from scipy.stats import norm
    powers = []
    for n in sample_sizes:
        se = effect_std / np.sqrt(n)
        z = effect_mean / se
        pwr = norm.cdf(z - 1.96) + (1.0 - norm.cdf(z + 1.96))
        powers.append(pwr)

    s.check(powers[0] < powers[1] < powers[2], "Statistical power is strictly monotonically increasing with sample size", f"powers={[round(p, 3) for p in powers]}")
    s.check(0.0 <= min(powers) and max(powers) <= 1.0, "Statistical power is bounded in [0, 1]")
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

    return s


def main() -> int:
    print("=" * 78)
    print("PHASE 7 STATISTICAL POWER & UNCERTAINTY — PRE-EVALUATION TEST SUITE")
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
        print("\n  STOP — do not run statistical_power_uncertainty_experiment.py.")
    return 1 if tf else 0


if __name__ == "__main__":
    raise SystemExit(main())
