# E8 — Steps 1–4 Audit Report (pre-implementation)

**Date:** 2026-08-20
**Status:** Audit only. No implementation code written. No production file touched.
**Per the execution order:** steps 1–4 (reproduction audit, calibration-infrastructure audit, base selection, grid freeze) are completed and reported before step 5.

---

## HEADLINE

**The E7 reproduction "failure" is not a bug. It is E7 correctly refusing to inherit an in-sample number.**

The V3 artifact was fitted on 2020/21–2024/25 — which includes **both** of E2's out-of-sample test seasons. E2's V3 figure of 0.981357 was therefore produced by a model that had already seen those fixtures' outcomes. E7 refit V3 per fold and got 0.990435, a genuine out-of-sample number.

**Classification per §25: CASE B — discrepancy explained and proven harmless to E8**, with one important correction to the record: E2's V3 and Blend numbers are optimistic and should not be used as reference values going forward.

---

## Step 1 — E7/E2 reproduction discrepancy

### The observations

| Quantity | E2 | E7 | Diff | How it was produced |
|---|---|---|---|---|
| **Market** | 0.955905 | 0.955905 | **0.000000** | Read from a stored file — no fitting |
| **V3** | 0.981357 | 0.990435 | **+0.009078** | Artifact vs per-fold refit |
| **Blend** | 0.957130 | 0.967628 | **+0.010498** | Inherits V3 |

Market reproduced to **exactly zero difference**. That is the tell: the component involving no model fitting matched perfectly, and only the fitted components diverged.

### Root cause — located precisely

`research/market_odds/run_e2_experiment.py:128`

```python
def load_v3_predictions(fixture_ids):
    """Uses the frozen V3 candidate artifact and the causal Elo module."""
    art = load_v3_artifact(path)          # <-- the SHIPPED artifact
```

E2 loaded `v3_poisson_venue_elo_candidate.pkl` and scored every fixture with it. E7 instead refit V3 inside each fold.

Verified from `src/models/config.py`:

```
V3 artifact FINAL_TRAIN_SEASONS: ('2020/2021','2021/2022','2022/2023','2023/2024','2024/2025')
E2 OOS test seasons:             2023/2024, 2024/2025
OVERLAP:                         ['2023/2024', '2024/2025']
```

**The artifact was fitted on 100% of the fixtures E2 evaluated it on.**

### Ruling out a bug

Three independent checks:

1. **Direction.** Both fitted quantities moved the *same* way — E7 worse. An indexing, alignment or ordering bug has no reason to be uniformly one-directional; in-sample optimism predicts exactly this sign.
2. **Magnitude.** +0.009 on log loss is the right order for in-sample optimism on an 87-feature ridge model with ~3.5k training rows.
3. **The blend gap follows arithmetically.** E2 fold 2 and E7 fold 2 both learned `w = 1.00`, meaning the blend *is* pure market there and V3 quality is irrelevant. The blend gap therefore comes almost entirely from fold 1, where `w < 1` and E7's V3 is worse. E2 fold 1 `w = 0.85`, E7 fold 1 `w = 0.30` — E7's optimiser correctly leaned harder on the market once V3 stopped being flattered by leakage.

Each check independently points to leakage-in-E2, not error-in-E7. Going through the §2 checklist: fixture selection, ordering, fold construction, probability conversion and label alignment are all identical between the two harnesses (E7 reproduced E2's fixture counts and fold boundaries exactly, and market matched to 0.0). The only differing element is the V3 fitting path.

### Classification

**CASE B — discrepancy explained and proven harmless**, with a correction attached:

- **E7's numbers are the correct ones.** Arm A = 0.990435 is V3's honest out-of-sample performance on this fixture set.
- **E2's V3 (0.981357) and Blend (0.957130) are in-sample-contaminated** and must not be used as reference values.
- **E2's Market (0.955905) is unaffected** and remains valid — it involves no fitting.
- **E2's PASS verdict is unaffected in direction.** E2 concluded the market signal beats V3. Correcting V3 *downward* to 0.990435 makes that conclusion stronger, not weaker.

I am not modifying `e2_results.json`. The historical record stands; this report documents the correction.

### Note on E7's FAIL

E7's FAIL was triggered by this reproduction gate. Given the above, the gate fired on a comparison that could never have succeeded — E7 was being asked to reproduce a leaked number. E7's *substantive* result is still a genuine FAIL on its own merits, independently of the gate: `w_D` reached 1.00, Arm D (0.970214) lost to market alone (0.955905), and criterion 5 is mandatory. **The FAIL verdict stands; only its stated reason needs amending.**

---

## Step 2 — Calibration infrastructure audit

`src/models/calibration.py` exists and contains:

| Component | Present |
|---|---|
| `compute_ece(y, P, n_bins)` | Yes |
| `compute_reliability_bins` | Yes |
| `IdentityCalibrator` | Yes |
| `MulticlassPlattCalibrator` (`_fit_platt_1d`, `_sigmoid`) | Yes |
| `MulticlassIsotonicCalibrator` (`_pava_weighted`) | Yes |
| **Temperature scaling** | **No — does not exist** |

The files named in §3 (`run_calibrated.py`, `run_matrix_calibration.py`, `calibrate_temperature.py`) **do not exist** in this repository.

**Consequences:**

1. Temperature scaling must be implemented from scratch in `research/temperature_scaling/`.
2. Platt and isotonic are explicitly excluded by §1, so the existing calibrators are not used.
3. `compute_ece` exists but lives in production code; E8 will implement its own ECE inside the research directory rather than import from `src/`, keeping the isolation rule intact and avoiding any temptation to modify a shared utility.

**Leakage review of existing calibrators:** `fit(P, y)` takes labels, so any use must be training-only. E8 does not use them, so this is moot — but noted, since §3 asked.

**Probability zeros:** V3/market probabilities come from `predict_poisson` and power de-vig, both of which normalise to sum 1 with strictly positive entries. Exact zeros are not expected, but `log(0)` must still be handled — see step 4.

---

## Step 3 — Primary base selection

§4 requires the strongest **proven and reproducible** incumbent.

Full E7 pooled OOS on the common universe (n = 3,479):

| Arm | Log Loss |
|---|---|
| A — V3 | 0.990435 |
| B — V3 + Market (blend) | 0.967628 |
| C — V3 + Online A/D | 0.991780 |
| D — V3 + A/D + Market | 0.970214 |
| **MARKET alone** | **0.955905** |

Market alone is best by a clear margin, and it is the **only** quantity that reproduced exactly between E2 and E7.

### Decision

> **PRIMARY E8 BASE = MARKET ALONE** (de-vigged Pinnacle closing 1X2).

Reasons:

1. Strongest incumbent — beats every fitted arm.
2. **Only** perfectly reproducible quantity (diff 0.000000).
3. Immune to the leakage that contaminated E2's V3 and Blend figures.
4. Requires no model fitting, so E8 tests temperature scaling in isolation with nothing else varying.

**Secondary diagnostic:** V3 (Arm A refit, 0.990435) will be temperature-scaled and reported, but it **cannot alter the primary decision**. Arm C is not used — it is a refit quantity with no E2 counterpart and adds nothing to a calibration question.

### A caution I want on record

Market alone already has the best calibration measured anywhere in this project: **ECE 0.007783** in E2, against V3's 0.013069. Temperature scaling exists to fix miscalibration. Applying it to the best-calibrated distribution available means **there may be very little headroom**, and `T ≈ 1.00` is a plausible — and entirely legitimate — outcome. §30 states this explicitly: if T = 1 repeatedly, calibration adds nothing. I expect that possibility and will not treat it as a failure of the experiment.

---

## Step 4 — Frozen temperature grid and formula

### Formula

```
P'_k = P_k^(1/T) / Σ_j P_j^(1/T)
```

implemented in log space for stability:

```
z_k  = log(P_k) / T
P'_k = softmax(z)          with max-subtraction before exp
```

`T < 1` sharpens · `T = 1` identity · `T > 1` softens.

### Grid — frozen now, before any outer-test result is seen

```
T ∈ {0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20}
```

Nine values, symmetric about 1.00, which is included so the procedure can decline to act. **The grid will not be expanded after seeing results.** If T repeatedly lands on 0.80 or 1.20 I will report the boundary problem rather than widen the grid.

### Numerical handling — documented epsilon

`log(0)` is `-inf`. Probabilities are clipped to `EPS = 1e-300` before the log — chosen because it is the smallest value keeping `log` finite in float64 while being ~290 orders of magnitude below any probability this pipeline actually produces (observed minimum across the market dataset is ~1e-3).

**The identity control is unaffected**: at `T = 1.0` the implementation returns the input array unchanged by an explicit short-circuit, so no epsilon, log or exp is applied and `max|P − P_T1| = 0` exactly rather than merely within tolerance.

### Validation protocol

E2's 2-fold structure, matching the base:

| Fold | Train | Test | n_test |
|---|---|---|---|
| 1 | 2022/23 | 2023/24 | 1,727 |
| 2 | 2022/23, 2023/24 | 2024/25 | 1,752 |

Eligible OOS: **n = 3,479**. 2025/26 quarantined.

**Inner-split limitation, stated now:** fold 1 trains on a single season, so no nested chronological inner split is constructible — the same structural issue flagged in E7. For fold 1 the temperature will be selected on the **outer training period directly** (2022/23 outcomes only, never test outcomes). This is training-only and therefore not leakage, but it is *not* nested validation, and I will report it as a limitation rather than describe fold 1 as nested.

---

## Protected file integrity

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## Summary of audit decisions

| Item | Decision |
|---|---|
| E7 reproduction discrepancy | **CASE B** — E2's V3/Blend were in-sample; E7's are correct |
| E2 numbers to trust | Market (0.955905) only |
| E7 FAIL verdict | **Stands** — Arm D lost to market alone on its own merits |
| Existing temperature code | **None** — implement fresh in research/ |
| **Primary E8 base** | **Market alone** |
| Secondary diagnostic | V3 refit (0.990435), non-decisive |
| Grid | 9 values, 0.80–1.20, **frozen** |
| Folds | E2's 2-fold, n = 3,479 |
| Known limitation | Fold 1 cannot use nested inner validation |
| Expected risk | Market is already well calibrated (ECE 0.0078) — T ≈ 1 is plausible |

Nothing blocks E8. Proceeding to step 5 (implementation).
