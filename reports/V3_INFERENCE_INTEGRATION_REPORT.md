# V3 Inference Integration Report

**Date:** 2026-08-20
**Scope:** Production inference pipeline integration for V3 Poisson+Venue+Elo
**Prerequisite:** V3 candidate passed preregistered walk-forward validation (see V3_ELO_PRODUCTION_CANDIDATE_REPORT.md)

This report validates the **inference pipeline**, not the model itself.

---

## 1. Files Changed

| File | Change | Reason |
|---|---|---|
| `predict_match.py` | Added `predict_v3()` function and `--v3` CLI flag | V3 inference path |

**What changed in predict_match.py:**
- Docstring updated to document `--v3` flag
- `predict_v3()` function added (lines 213–319): loads 84 base features from features.db, computes causal Elo from matches.db, joins to 87-column contract, predicts via V3 artifact
- `--v1` and `--v3` placed in a `mutually_exclusive_group` to prevent ambiguous invocations
- Default (no flag) remains V2 — unchanged

**What was NOT changed in predict_match.py:**
- `predict_v1()` function: zero edits, byte-identical
- `predict_v2()` function: zero edits, byte-identical
- `team_names()` function: zero edits
- `fail()` function: zero edits
- Default model selection: remains V2

## 2. Files Created

| File | Size | Purpose |
|---|---|---|
| `src/features/elo.py` | 5,936 bytes | Production causal Elo module |
| `tests/test_v3_inference_integration.py` | 15,223 bytes | 5-suite integration test |

## 3. Files Untouched

| File | MD5 | Verified |
|---|---|---|
| `data/models/v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | PRE ✓ POST ✓ |
| `data/models/v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | PRE ✓ POST ✓ |
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` | PRE ✓ POST ✓ |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` | PRE ✓ POST ✓ |
| `data/models/v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | Untouched |
| `src/models/v2_artifact.py` | — | Untouched |
| `src/models/v3_artifact.py` | — | Untouched |
| `src/models/v3_contract.py` | — | Untouched |
| `src/models/poisson.py` | — | Untouched |
| `src/models/ablation.py` | — | Untouched |
| `src/models/config.py` | — | Untouched |
| `src/models/data.py` | — | Untouched |
| `src/models/artifact.py` | — | Untouched |
| `src/features/feature_builder.py` | — | Untouched |
| `src/features/strength.py` | — | Untouched |
| All other `src/features/*.py` | — | Untouched |
| All other `src/models/*.py` | — | Untouched |

---

## 4. V2 Compatibility Verification

| Check | Result |
|---|---|
| `predict_v1()` function modified? | **NO** — zero edits |
| `predict_v2()` function modified? | **NO** — zero edits |
| Default model (no flag)? | **V2** — unchanged |
| `--v1` still works? | **YES** — unchanged |
| `--v1` and `--v3` mutually exclusive? | **YES** — enforced by argparse |
| V2 artifact loaded? | Same path, same MD5, same loader |
| V2 feature contract? | 84 columns, unchanged |
| V2 Poisson conversion? | Same `predict_poisson()` call |

---

## 5. V3 Feature Contract

```
Columns  1–80:  MODEL_B_COLUMNS (base features from features.db)
Columns 81–84:  VENUE_COLUMNS (venue features from features.db)
Column     85:  home_elo    (computed from matches.db at inference time)
Column     86:  away_elo    (computed from matches.db at inference time)
Column     87:  elo_diff    (computed from matches.db at inference time)
```

**Verified properties (Suite 1, 10/10 PASS):**
- V2 = exactly 84 features
- V3 = exactly 87 features
- V3[:80] == MODEL_B_COLUMNS
- V3[80:84] == VENUE_COLUMNS
- V3[84:87] == ("home_elo", "away_elo", "elo_diff")
- No label columns, no duplicates
- V3[:84] == V2 contract (strict superset)
- Production `ELO_COLUMNS` == contract `ELO_FEATURE_COLUMNS`

---

## 6. Elo Causal Guarantees

**Production module:** `src/features/elo.py`

**Architecture:** Two-pass per-timestamp design:
1. First pass: extract pre-match ratings for all fixtures at timestamp T
2. Second pass: update ratings with outcomes of completed matches at T

**Frozen parameters (matching V3 candidate training):**
- init_rating = 1500.0
- k_factor = 20.0
- home_advantage = 100.0
- goal_diff_multiplier = eloratings.net standard
- mean_reversion = 0.0

**Verified invariants (Suite 2, 14/14 PASS):**

| Test | Method | Result |
|---|---|---|
| Outcome insulation | Modify fixture 250 goals to 10-0; pre-match Elo unchanged | PASS (diff = 0.00e+00) |
| Truncation invariance N=1000 | Compute on first 1000 vs all; first 1000 identical | PASS (diff = 0.00e+00) |
| Truncation invariance N=3000 | Same as above | PASS (diff = 0.00e+00) |
| Truncation invariance N=5000 | Same as above | PASS (diff = 0.00e+00) |
| Outcome removal | Set fixture 250 to NS/NaN goals; pre-match Elo unchanged | PASS |
| Determinism | Two independent runs on identical data | PASS (diff = 0.00e+00) |
| All home_elo finite | `np.isfinite` on all 10,735 values | PASS |
| All away_elo finite | Same | PASS |
| All elo_diff finite | Same | PASS |
| Reasonable bounds | Range [1240.8, 1899.5] within [700, 2300] | PASS |
| Cross-season persistence | Recent fixtures have non-initial Elo (100/100) | PASS |
| Initial rating correct | 1500.0 | PASS |
| First fixture home_elo == 1500.0 | First fixture starts at init | PASS |
| First fixture away_elo == 1500.0 | Same | PASS |

---

## 7. Historical Inference Test Results

**Production vs Research Elo Equivalence (Suite 3, 4/4 PASS):**

| Check | Result |
|---|---|
| Same fixture count | prod = 10,735, research = 10,735 |
| home_elo max diff | 0.00e+00 |
| away_elo max diff | 0.00e+00 |
| elo_diff max diff | 0.00e+00 |

The production Elo module (`src/features/elo.py`) produces **bit-identical** output to the research engine (`research/worldcup_elo/elo_engine.py`) on all 10,735 fixtures. This means V3 inference through the production path will use the exact same Elo features that were validated in the controlled experiment.

**V2/V3 Side-by-Side Inference (Suite 4):**

Suite 4 requires scipy/sklearn, which are unavailable in the sandbox environment (pip blocked by proxy). This suite must be run on your local machine:

```
python tests/test_v3_inference_integration.py
```

Suite 4 tests:
- V2 still produces valid predictions (lambda > 0, probs sum to 1, all finite)
- V3 produces valid predictions (lambda > 0, probs sum to 1, all in [0,1])
- V3 has exactly 87 columns in correct order for each test fixture
- V3 predictions are deterministic (identical across runs)

---

## 8. Upcoming-Fixture Support Status

**NOT SUPPORTED — by design.**

The existing `predict_match.py` (V1, V2, and now V3) can only predict fixtures that already have a feature row in `data/processed/features.db`. This is documented at lines 9–16 of the script.

V3 does not change this limitation. Upcoming-fixture inference would require:
- Upcoming-fixture ingestion
- Team-name resolution
- Current-season feature construction semantics
- None of which exist in the current production scope

V3 historical inference is fully integrated; upcoming-fixture inference remains outside the current production scope.

---

## 9. Determinism Results

| Component | Method | Max Diff |
|---|---|---|
| Production Elo engine | Two independent `compute_elo_features()` runs | 0.00e+00 |
| Production vs Research Elo | `load_elo_features` vs `load_matches_and_build_elo` | 0.00e+00 |
| V3 model predictions | Two `predict` calls on same input (requires local run) | Expected < 1e-12 |

No random state dependency exists in the pipeline (PoissonRegressor is deterministic, Elo is deterministic, no random splitting).

---

## 10. Probability Validation

Validated structurally in Suite 4 (requires local execution):
- All P(H), P(D), P(A) ∈ [0, 1]
- P(H) + P(D) + P(A) = 1.0 (within 1e-8)
- lambda_home > 0
- lambda_away > 0
- All values finite

The V3 artifact uses the same `predict_poisson()` conversion as V2 (in `src/models/poisson.py`), which has its own safety gates: tail residual, complement control, row-sum tolerance, and [0,1] bounds checking.

---

## 11. MD5 Integrity Results

| File | Pre-Change MD5 | Post-Change MD5 | Status |
|---|---|---|---|
| `data/models/v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `data/models/v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## 12. Git Status

Git is not available in the sandbox environment. The following is the complete list of filesystem changes:

**New files (2):**
- `src/features/elo.py` — production causal Elo module
- `tests/test_v3_inference_integration.py` — 5-suite integration test

**Modified files (1):**
- `predict_match.py` — added `predict_v3()` and `--v3` flag

**No other files were modified.** All production source code in `src/models/`, `src/features/` (except the new `elo.py`), `scripts/`, and `config/` is untouched.

---

## 13. Known Limitations

1. **Suite 4 (V2/V3 side-by-side inference) requires local execution** — scipy/sklearn are unavailable in the sandbox. Run `python tests/test_v3_inference_integration.py` locally before promotion.

2. **Elo computation is O(all fixtures)** — `get_elo_for_fixture()` computes the full Elo chain for all 10,735 fixtures to ensure causal correctness, then returns the single requested row. This is correct but not optimized for repeated single-fixture lookups. For batch predictions, use `load_elo_features()` once and join.

3. **No Elo caching** — each `--v3` invocation recomputes the full Elo chain. Acceptable for CLI usage; would need caching for any future batch or API mode.

4. **Upcoming fixtures not supported** — same limitation as V1/V2.

5. **V3 artifact path resolution** — `predict_v3()` prefers `v3_poisson_venue_elo.pkl` (promoted) over `v3_poisson_venue_elo_candidate.pkl` (candidate). Both must be in `data/models/`.

---

## 14. Exact Commands

**V2 (default, unchanged):**
```bash
python predict_match.py --fixture-id 123456
python predict_match.py --fixture-id 123456 --json
```

**V1 (backward compatible, unchanged):**
```bash
python predict_match.py --fixture-id 123456 --v1
```

**V3 (new):**
```bash
python predict_match.py --fixture-id 123456 --v3
python predict_match.py --fixture-id 123456 --v3 --json
```

**Run integration tests (local only):**
```bash
python tests/test_v3_inference_integration.py
```

---

## 15. Production Promotion Safety Assessment

### What passed in the sandbox (30/30):

| Suite | Tests | Result |
|---|---|---|
| 1. Feature Contract | 10/10 | **ALL PASS** |
| 2. Causal Elo Invariants | 14/14 | **ALL PASS** |
| 3. Production vs Research Elo Equivalence | 4/4 | **ALL PASS** |
| 4. V2/V3 Side-by-Side Inference | — | Requires local (scipy) |
| 5. Protected File Integrity | 4/4 | **ALL PASS** |

### What must pass locally before promotion:

Suite 4 must pass on your local machine. Run:
```bash
python tests/test_v3_inference_integration.py
```

All 5 suites must show 0 failures.

### Safety checklist:

| Requirement | Status |
|---|---|
| V2 default unchanged | ✓ |
| V1 backward compatible | ✓ |
| V2 artifact untouched (MD5) | ✓ |
| V1 artifact untouched (MD5) | ✓ |
| features.db untouched (MD5) | ✓ |
| matches.db untouched (MD5) | ✓ |
| V3 artifact not modified | ✓ |
| No model training | ✓ |
| No 2025/26 access | ✓ |
| Elo causally sound | ✓ (14/14 tests) |
| Production Elo == Research Elo | ✓ (bit-identical on all 10,735 fixtures) |
| No new features beyond contract | ✓ (exactly home_elo, away_elo, elo_diff) |
| No scope expansion | ✓ |

---

# FINAL DECISION: **SAFE FOR PRODUCTION PROMOTION**

Conditional on Suite 4 passing locally (`python tests/test_v3_inference_integration.py` — all 5 suites, 0 failures).

The inference pipeline correctly computes causal Elo features that are bit-identical to those used during the validated experiment, wires them through the existing Poisson conversion, and leaves V2 as the untouched default. No protected file was modified. No model was trained. No 2025/26 data was accessed.

**Do not promote until Suite 4 passes locally.**
