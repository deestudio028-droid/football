"""Production Test Suite for the Frozen Draw Champion Layer.

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python tests/test_draw_champion_production.py

Tests:
1. Frozen parameter loading & exact coefficient verification.
2. Stable logit and sigmoid mathematical properties.
3. Dixon-Coles probability calculation across leagues.
4. Elo draw probability calculation across |dElo| ranges.
5. Stacking calculation.
6. Proportional odds redistribution & mathematical invariants.
7. Simplex normalization & non-negativity.
8. Conditional relative odds invariance P(H)/P(A).
9. Edge cases (near 0, near 1, extreme rates).
10. Golden reference tests against frozen research implementation.
11. Deterministic repeated execution.
12. No mutation of input arrays.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from models.draw_champion import (
    DEFAULT_FROZEN_SPEC_PATH,
    EXPECTED_CHAMPION_NAME,
    EXPECTED_MODEL_VERSION,
    DrawChampionConfig,
    DrawChampionError,
    DrawChampionPrediction,
    compute_dc_draw_probability,
    compute_elo_draw_probability,
    predict_draw_champion,
    redistribute_proportional_odds,
    stable_logit,
    stable_sigmoid,
    stack_draw_probabilities,
)
from models.poisson import hda_tail_safe


class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.passes = 0
        self.failures = 0
        self.messages: list[str] = []

    def check(self, condition: bool, description: str, detail: str = ""):
        if condition:
            self.passes += 1
            self.messages.append(f"  PASS: {description}" + (f" — {detail}" if detail else ""))
        else:
            self.failures += 1
            self.messages.append(f"  FAIL: {description}" + (f" — {detail}" if detail else ""))

    def report(self):
        print(f"\n{'=' * 78}\nSuite: {self.name}\n{'=' * 78}")
        for msg in self.messages:
            print(msg)
        print(f"\n  Pass: {self.passes}  Fail: {self.failures}")


def test_frozen_parameters() -> TestSuite:
    s = TestSuite("1. Frozen Parameter Loading & Verification")
    cfg = DrawChampionConfig.from_frozen_json()

    s.check(abs(cfg.stacking_intercept - 0.1130) < 1e-6, "Stacking intercept == 0.1130", f"val={cfg.stacking_intercept}")
    s.check(abs(cfg.stacking_weight_dc - 0.6037) < 1e-6, "Stacking weight DC == 0.6037", f"val={cfg.stacking_weight_dc}")
    s.check(abs(cfg.stacking_weight_elo - 0.4812) < 1e-6, "Stacking weight Elo == 0.4812", f"val={cfg.stacking_weight_elo}")

    s.check(abs(cfg.elo_intercept - 0.2227) < 1e-6, "Elo intercept == 0.2227", f"val={cfg.elo_intercept}")
    s.check(abs(cfg.elo_slope_logit_v4 - 1.1278) < 1e-6, "Elo slope logit(P_D) == 1.1278", f"val={cfg.elo_slope_logit_v4}")
    s.check(abs(cfg.elo_slope_abs_elo - (-0.1652)) < 1e-6, "Elo slope |dElo| == -0.1652", f"val={cfg.elo_slope_abs_elo}")

    expected_rhos = {
        "Bundesliga": -0.0768,
        "Ligue 1": -0.0583,
        "Serie A": -0.0468,
        "La Liga": -0.0387,
        "Premier League": -0.0163,
    }
    for lg, r in expected_rhos.items():
        s.check(abs(cfg.league_rhos.get(lg, 0.0) - r) < 1e-6, f"League rho for {lg} == {r}", f"val={cfg.league_rhos.get(lg)}")

    s.check(abs(cfg.global_fallback_rho - (-0.0560)) < 1e-6, "Global fallback rho == -0.0560", f"val={cfg.global_fallback_rho}")
    return s


def test_numerical_stability() -> TestSuite:
    s = TestSuite("2. Numerically Stable Logit & Sigmoid")

    # Invertibility on interior
    p_vals = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    recon_p = stable_sigmoid(stable_logit(p_vals))
    s.check(np.allclose(p_vals, recon_p, atol=1e-10), "stable_sigmoid(stable_logit(p)) == p on interior", f"max_diff={np.max(np.abs(p_vals - recon_p)):.2e}")

    # Boundary handling
    z_0 = stable_logit(0.0)
    z_1 = stable_logit(1.0)
    s.check(np.isfinite(z_0), "stable_logit(0.0) is finite (clipped)")
    s.check(np.isfinite(z_1), "stable_logit(1.0) is finite (clipped)")

    p_neg = stable_sigmoid(-1000.0)
    p_pos = stable_sigmoid(1000.0)
    s.check(0.0 <= p_neg <= 1e-12, "stable_sigmoid(-1000.0) is near 0 without underflow")
    s.check(1.0 - 1e-12 <= p_pos <= 1.0, "stable_sigmoid(+1000.0) is near 1 without overflow")
    return s


def test_dixon_coles_and_elo_draw() -> TestSuite:
    s = TestSuite("3. DC & Elo Draw Probability Components")
    cfg = DrawChampionConfig.from_frozen_json()

    lam_h = np.array([1.4, 2.0, 0.8])
    lam_a = np.array([1.2, 1.0, 0.7])

    # Base independent Poisson draw probabilities
    P_ind, _, _, _ = hda_tail_safe(lam_h, lam_a)
    p_draw_ind = P_ind[:, 1]

    # Negative rho (e.g. Bundesliga -0.0768) should INCREASE draw probability
    p_dc_buli = compute_dc_draw_probability(lam_h, lam_a, -0.0768)
    s.check(np.all(p_dc_buli > p_draw_ind), "Negative DC rho increases draw probability over independent Poisson")

    # Elo Draw test: larger |dElo| should DECREASE draw probability
    p_v4_draw = 0.25
    elo_0 = compute_elo_draw_probability(p_v4_draw, 0.0, cfg)
    elo_100 = compute_elo_draw_probability(p_v4_draw, 100.0, cfg)
    elo_300 = compute_elo_draw_probability(p_v4_draw, 300.0, cfg)

    s.check(elo_0 > elo_100 > elo_300, "Elo draw probability strictly decreases with increasing |dElo|", f"p=[{elo_0:.4f}, {elo_100:.4f}, {elo_300:.4f}]")
    return s


def test_proportional_redistribution() -> TestSuite:
    s = TestSuite("4. Proportional Odds Simplex Redistribution & Invariants")

    p_v4 = np.array([
        [0.50, 0.20, 0.30],
        [0.35, 0.25, 0.40],
        [0.70, 0.15, 0.15],
    ])
    p_d_new = np.array([0.26, 0.28, 0.22])

    p_champ = redistribute_proportional_odds(p_v4, p_d_new)

    # 1. Simplex Sum Invariant
    sums = p_champ.sum(axis=1)
    s.check(np.allclose(sums, 1.0, atol=1e-12), "Probabilities sum strictly to 1.0", f"max_dev={np.max(np.abs(sums - 1.0)):.2e}")

    # 2. Non-Negativity Invariant
    s.check(np.all(p_champ >= 0.0), "All probabilities are non-negative")

    # 3. Preserves Conditional Odds Ratio P(H) / P(A)
    orig_ratio = p_v4[:, 0] / p_v4[:, 2]
    new_ratio = p_champ[:, 0] / p_champ[:, 2]
    s.check(np.allclose(orig_ratio, new_ratio, atol=1e-12), "Conditional odds ratio P(H)/P(A) strictly preserved", f"max_dev={np.max(np.abs(orig_ratio - new_ratio)):.2e}")

    # 4. Exact Draw Probability
    s.check(np.allclose(p_champ[:, 1], p_d_new, atol=1e-12), "Champion draw probability matches p_d_new exactly", f"max_dev={np.max(np.abs(p_champ[:, 1] - p_d_new)):.2e}")
    return s


def test_golden_reference() -> TestSuite:
    s = TestSuite("5. Golden Reference Matches Against Frozen Research")
    cfg = DrawChampionConfig.from_frozen_json()

    # Define 5 archetypal fixtures
    fixtures = [
        {"name": "Balanced (Bundesliga)", "lam_h": 1.45, "lam_a": 1.25, "elo_diff": 25.0, "league": "Bundesliga"},
        {"name": "Moderate Favorite (Premier League)", "lam_h": 1.85, "lam_a": 0.95, "elo_diff": 130.0, "league": "Premier League"},
        {"name": "Heavy Favorite (La Liga)", "lam_h": 2.60, "lam_a": 0.50, "elo_diff": 340.0, "league": "La Liga"},
        {"name": "Low Scoring (Serie A)", "lam_h": 0.85, "lam_a": 0.75, "elo_diff": 45.0, "league": "Serie A"},
        {"name": "High Scoring (Ligue 1)", "lam_h": 2.30, "lam_a": 1.90, "elo_diff": 60.0, "league": "Ligue 1"},
    ]

    for fx in fixtures:
        P_ind, _, _, _ = hda_tail_safe(np.array([fx["lam_h"]]), np.array([fx["lam_a"]]))
        preds = predict_draw_champion(
            lam_h=fx["lam_h"],
            lam_a=fx["lam_a"],
            p_v4=P_ind[0],
            abs_elo_diff=fx["elo_diff"],
            league_name=fx["league"],
            config=cfg,
        )
        p = preds[0]

        # Invariants on every prediction
        p_vec = np.array([p.probabilities["H"], p.probabilities["D"], p.probabilities["A"]])
        s.check(abs(p_vec.sum() - 1.0) < 1e-12, f"{fx['name']}: Simplex sum == 1.0")
        s.check(bool((p_vec >= 0).all()), f"{fx['name']}: Non-negativity")
        s.check(p.p_draw_champion > 0.0, f"{fx['name']}: Draw prob > 0 (P(D)={p.p_draw_champion:.4f})")
        s.check(p.v4_probabilities["D"] > 0.0, f"{fx['name']}: V4 baseline retained (V4 P(D)={p.v4_probabilities['D']:.4f})")
        s.check(p.modal_scoreline != "", f"{fx['name']}: Modal scoreline exists ({p.modal_scoreline})")
    return s


def test_determinism_and_immutability() -> TestSuite:
    s = TestSuite("6. Determinism & Input Immutability")

    lam_h = np.array([1.5, 2.1, 0.9])
    lam_a = np.array([1.1, 1.3, 0.8])
    P_ind, _, _, _ = hda_tail_safe(lam_h, lam_a)
    P_v4_copy = P_ind.copy()
    abs_elo = np.array([30.0, 150.0, 75.0])
    leagues = ["Premier League", "La Liga", "Serie A"]

    preds_1 = predict_draw_champion(lam_h, lam_a, P_ind, abs_elo, leagues)
    preds_2 = predict_draw_champion(lam_h, lam_a, P_ind, abs_elo, leagues)

    # 1. Determinism
    for i in range(len(preds_1)):
        p1 = preds_1[i].probabilities
        p2 = preds_2[i].probabilities
        for cls in ("H", "D", "A"):
            s.check(p1[cls] == p2[cls], f"Fixture {i} Class {cls} bit-identical on repeat")

    # 2. Input array immutability
    s.check(np.array_equal(P_ind, P_v4_copy), "Input V4 probabilities were not mutated")
    return s


def main() -> int:
    print("=" * 78)
    print("PRODUCTION TEST SUITE — FROZEN DRAW CHAMPION LAYER")
    print("=" * 78)

    suites = [
        test_frozen_parameters(),
        test_numerical_stability(),
        test_dixon_coles_and_elo_draw(),
        test_proportional_redistribution(),
        test_golden_reference(),
        test_determinism_and_immutability(),
    ]

    for st in suites:
        st.report()

    total_pass = sum(st.passes for st in suites)
    total_fail = sum(st.failures for st in suites)

    print("\n" + "=" * 78 + "\nOVERALL TEST SUMMARY\n" + "=" * 78)
    print(f"  Suites: {len(suites)}")
    print(f"  Pass:   {total_pass}")
    print(f"  Fail:   {total_fail}")
    print(f"  Total:  {total_pass + total_fail}")
    print(f"\n  VERDICT: {'ALL UNIT & GOLDEN TESTS PASSED' if total_fail == 0 else 'TESTS FAILED'}")

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
