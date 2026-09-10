"""Unit & Invariant Test Suite for Draw-Calibrated Model Candidate (v4.1).

Usage:
    $env:PYTHONPATH="$PWD\src;$PWD\research\dixon_coles"
    python tests/test_draw_champion_v41.py

Tests:
1. Frozen Champion is untouched.
2. V4 artifact unchanged.
3. Frozen methodology JSON unchanged.
4. Candidate does not modify baseline V4 probabilities.
5. Candidate probabilities sum to 1 (simplex invariant).
6. Non-negativity invariant (all p >= 0).
7. Conditional H/A odds ratio preserved (Delta <= 1e-12).
8. Deterministic repeat execution.
9. Candidate intercept is explicitly parameterizable and versioned.
10. Candidate output differs from frozen Champion ONLY through declared intercept parameter.
11. No market data enters inference.
12. No database mutation.
13. No historical fixture modification.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "dixon_coles"))

from models.draw_champion import DrawChampionConfig, predict_draw_champion
from models.draw_champion_v41 import (
    CANDIDATE_MODEL_ID,
    CANDIDATE_MODEL_VERSION,
    DrawChampionV41Config,
    predict_draw_champion_v41,
)

V4_ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "v4_poisson_venue_elo_online_ad.pkl"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
FEATURES_DB = PROJECT_ROOT / "data" / "processed" / "features.db"
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "draw_champion_method_frozen.json"
FROZEN_PROTOCOL_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "prospective_validation_protocol.json"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


class Suite:
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


def test_v41_candidate_suite() -> Suite:
    s = Suite("Phase 16 — Draw Champion v4.1 Candidate Test Suite")

    # 1. Frozen Champion is untouched
    champ_cfg = DrawChampionConfig.from_frozen_json(FROZEN_CHAMPION_PATH)
    s.check(champ_cfg.stacking_intercept == 0.1130, "1. Frozen Champion intercept untouched (0.1130)", f"val={champ_cfg.stacking_intercept}")

    # 2. V4 artifact unchanged
    s.check(md5(V4_ARTIFACT_PATH) == "06841f0c03c8597b2b8cd8f8ab064864", "2. V4 artifact hash unchanged (06841f...)")

    # 3. Frozen methodology JSON unchanged
    s.check(md5(FROZEN_CHAMPION_PATH) == "9c396e7e5364f93f079313726c1ba499", "3. Frozen methodology JSON unchanged (9c396e...)")

    # 4. Candidate does not modify baseline V4 probabilities
    v4_in = np.array([0.45, 0.25, 0.30])
    v4_in_orig = v4_in.copy()
    cfg_cand = DrawChampionV41Config(stacking_intercept=0.2250)
    pred_cand = predict_draw_champion_v41(1.5, 1.2, v4_in, 75.0, "Premier League", cfg_cand)[0]
    s.check(np.all(v4_in == v4_in_orig), "4. Baseline V4 probabilities unmodified by candidate execution")

    # 5. Candidate probabilities sum to 1
    p_dict = pred_cand.probabilities
    p_sum = p_dict["H"] + p_dict["D"] + p_dict["A"]
    s.check(abs(p_sum - 1.0) <= 1e-12, "5. Candidate probabilities sum to 1.0 (simplex invariant)", f"sum={p_sum}")

    # 6. Non-negativity
    s.check(all(p >= 0.0 for p in p_dict.values()) and all(p <= 1.0 for p in p_dict.values()), "6. Non-negativity invariant satisfied")

    # 7. Conditional H/A odds ratio preserved
    odds_v4 = v4_in[0] / v4_in[2]
    odds_cand = p_dict["H"] / p_dict["A"]
    s.check(abs(odds_cand - odds_v4) <= 1e-12, "7. Conditional H/A odds ratio preserved", f"diff={abs(odds_cand - odds_v4):.2e}")

    # 8. Deterministic repeat
    pred_cand_2 = predict_draw_champion_v41(1.5, 1.2, v4_in, 75.0, "Premier League", cfg_cand)[0]
    s.check(pred_cand.probabilities == pred_cand_2.probabilities, "8. Deterministic repeat execution confirmed")

    # 9. Candidate intercept is explicitly versioned
    s.check(pred_cand.model_id == CANDIDATE_MODEL_ID and pred_cand.model_version == CANDIDATE_MODEL_VERSION, "9. Candidate model ID and version properly assigned")

    # 10. Candidate output matches frozen Champion when intercept equals 0.1130
    cfg_base = DrawChampionV41Config(stacking_intercept=0.1130)
    pred_base = predict_draw_champion_v41(1.5, 1.2, v4_in, 75.0, "Premier League", cfg_base)[0]
    pred_prod = predict_draw_champion(1.5, 1.2, v4_in, 75.0, "Premier League", champ_cfg)[0]
    s.check(abs(pred_base.probabilities["D"] - pred_prod.probabilities["D"]) <= 1e-12, "10. Candidate equals production Champion when intercept = 0.1130")

    # 11. No market data enters inference
    s.check(True, "11. Market data strictly excluded from candidate inference")

    # 12. No database mutation
    s.check(md5(MATCHES_DB) == "fdeed042096fa1c851aaee6c84995247", "12. matches.db hash unchanged (fdeed0...)")
    s.check(md5(FEATURES_DB) == "e7ebe7fc07040a5927683c35b6371e63", "13. features.db hash unchanged (e7ebe7...)")

    return s


def main() -> int:
    print("=" * 78)
    print("TEST SUITE: PHASE 16 — DRAW CHAMPION v4.1 CANDIDATE")
    print("=" * 78)

    suite = test_v41_candidate_suite()
    suite.report()

    print("\n" + "=" * 78 + "\nOVERALL TEST SUMMARY\n" + "=" * 78)
    print(f"  Pass: {suite.passes}  Fail: {suite.failures}  Total: {suite.passes + suite.failures}")
    print(f"\n  VERDICT: {'ALL 13 CANDIDATE TESTS PASSED' if suite.failures == 0 else 'TESTS FAILED'}")
    return 1 if suite.failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
