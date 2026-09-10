# FINAL MODEL AUDIT / PHASE CLOSURE REPORT

**Document type:** phase closure. Read-only audit. No experiment, no fit, no code or data change.

**Date:** 2026-08-18

**Evidence-class convention used throughout:** **PROVEN/MEASURED** (directly computed) · **REPLICATED** (holds in ≥2 of 3 folds, same direction) · **INFERENCE** (follows from multiple diagnostics, not directly measured) · **SUGGESTION** (consistent with evidence, not established) · **NOT JUSTIFIED** (evidence does not support) · **NOT VERIFIED** (not recoverable from project evidence).

---

## 1. Executive Verdict

| Item | Value |
|---|---|
| Current model version | **`MODEL_VERSION = v1.0`** |
| Frozen H1 regularisation | **C = 0.0005** (`max_iter=2000`, `random_state=0`) |
| Validation folds | **3** walk-forward folds (2022/23, 2023/24, 2024/25) |
| Mean H1 validation log loss | **0.9915427871369671** |
| H1 reproduction | **EXACT** — reproduced to <1e-9 in every diagnostic that required it |
| Improvement cycle | **CLOSED** |

> ## STOP — no further experiment on the existing information contract is authorized.

Every mechanism proposed during this cycle was tested and rejected on pre-2025/26 validation evidence. The remaining error is dominated by weak information magnitude, not by a correctable model defect.

---

## 2. Frozen Baseline

| Item | Value | Class |
|---|---|---|
| `MODEL_VERSION` | `v1.0` | PROVEN |
| Estimator | `LogisticRegression(C=0.0005, max_iter=2000, random_state=0)` | PROVEN |
| Contract | **76 columns** = 75 numeric + `competition_id` | PROVEN |
| fold_1 log loss | **0.9979365509246185** | PROVEN |
| fold_2 log loss | **0.9846529437399034** | PROVEN |
| fold_3 log loss | **0.9920388667463793** | PROVEN |
| Mean log loss | **0.9915427871369671** | PROVEN |
| Mean Brier | **0.5913870749997768** | PROVEN |
| Per-fold accuracy / Brier | **NOT VERIFIED** — not recorded in the evidence available to this audit | — |
| Pinned baselines | **13/13 byte-identical** | PROVEN |
| Reproduction | **exact, <1e-9, deterministic repeat PASS** | PROVEN |

**Locked 2025/26 final-test record (untouched, quoted for the record only):** V1 C=1.0 — 902/1751 correct, accuracy 0.5151342090234152, log loss 0.9960487649063601, Brier 0.5947186313129067. E-1 C=0.0005 — 901/1751 correct, accuracy 0.5145631067961165, log loss 0.9948612098135484, Brier 0.5940695427374098.

---

## 3. Experiment / Diagnostic History

| Experiment / Diagnostic | Purpose | Result | Status | Decision |
|---|---|---|---|---|
| **E-1** extended-C validation | test C below the Phase 4B grid edge | C=0.0005 lowest mean log loss, improved in **all 3 folds**, Brier improved, deterministic | Qualified under the pre-registered rule | Frozen as the H1 candidate; final test spent |
| **H1** de-duplication (80→76) | remove 4 exact alias pairs | mean log-loss gain **−2.915e-05**; at `tol=1e-10` **−2.341e-05** | Mechanism confirmed (P1/P2/P4), effect **empirically inert** — below the solver `tol` of 1e-4 | Adopted as the contract for diagnostics; **not promoted** |
| **H2-C** recency reparameterisation | `[season, last10−season, last5−last10]`, impute-first | mean log loss **+1.461489e-03**, **all 3 folds worse**, Brier +1.106970e-03, accuracy −8.888e-04 | **WORSENED** | **REJECTED** |
| **H3 / H3-A** family penalty imbalance | test whether large families gain disproportionate influence | see §4 | **FALSIFIED** | **REJECTED** |
| **H4** balance × attacking-output interaction | test a nonlinear interaction | fold_1 ρ=+1.000, fold_2 ≈0, **fold_3 reversed** (−0.20, −0.50); Bonferroni-adjusted best p = 0.080 | **NOT REPLICATED** | **REJECTED** |
| **Calibration** (Phase 3 Step 4) | Platt / isotonic under nested expanding-window | uncalibrated 1.0006233634; Platt **+0.004744**; isotonic **+0.049713** | **WORSENED** | **REJECTED** |
| **Model-ceiling diagnostic** | locate the remaining error | see §5 | Errors dominated by low confidence | Supports **STOP** |
| **Draw-separation diagnostic** | is there exploitable draw signal? | see §6 | Signal real, replicated, very weak | Supports **STOP** |
| **Structural investigation** | is there a correctable structural limitation? | see §7 | Draw never modal in any decile | Supports **STOP** |

---

## 4. H3 Family-Coefficient Audit

**Families (H1 contract):** `goals_core` 18 · `form` **16** · `strength` 5 · `competition_id` **1 raw / 5 encoded** · `shots_core` 18 · `shots_on_core` 18. **PROVEN.**

**Signal measure** (mean |Cohen's d|, predicted-H subset, pooled): strength **0.2450**, shots_on_core 0.2171, goals_core 0.2013, shots_core 0.1994, form **0.1575**. `competition_id`: **NOT APPLICABLE** — Cohen's *d* is undefined for a categorical, and its one-hot block is appended *after* standardisation, so its coefficients are in different units and are never compared against numeric families. **PROVEN.**

**H3 prediction:** influence per unit of measured signal rises with family size.

**Outcome:** **6 positive / 3 negative** fold×class cells (as recorded). Mean ρ and median ρ: **NOT VERIFIED** — not supplied to this audit. Under the pre-registered rule, a sign that is not consistent across all 9 cells is not replication.

> ### H3-A PREMISE = FALSIFIED

**Why a larger total coefficient mass is not evidence.** Under ordinary L2, a family with more columns is *expected* to accumulate more total |coefficient| simply because it has more coefficients — that is the null expectation, not a finding. The only admissible diagnostic is influence **after** normalising for size and signal (`L1_per_dimension / mean_abs_d`), and that ratio did not rise consistently with family size. Reporting total mass alone would have manufactured a result.

---

## 5. Error-Ceiling Evidence

| Finding | Value | Class |
|---|---|---|
| Wrong predictions with confidence < 0.60 | **0.8850 / 0.8799 / 0.8408** | PROVEN, REPLICATED |
| Confident-wrong share of total NLL | **0.1007 / 0.1032 / 0.1354** | PROVEN, REPLICATED |
| Wrong-only NLL share of fold total | **NOT VERIFIED** | — |
| Temporal drift | **NOT ESTABLISHED** — three folds cannot support a trend; a monotone ordering of three points arises by chance with probability 1/3 | PROVEN (structural) |
| Feature-regime results | **NOT VERIFIED** in numeric detail | — |
| Strength / match-balance regimes | **NOT VERIFIED** in numeric detail from the ceiling run; the draw-specific balance result is in §6 | — |
| Counterfactual distance (draws) | median **14.8128 / 12.3164 / 11.7219 SD**; within 1 SD = **0% / 0% / 0.92%** | PROVEN |
| Recurring counterfactual feature | `home_shots_for_per_match_last10` in all 3 folds | PROVEN |

**Why this does not justify another existing-feature experiment.** 84–89% of errors are low-confidence, and confident-wrong rows carry only 10.1–13.5% of total NLL — so even eliminating every confident error entirely would leave the great majority of the loss untouched **(INFERENCE)**. Counterfactual distances of 11.7–14.8 SD are an explicit **upper bound** computed by holding all other features fixed; because the contract is highly correlated, real movement is never single-feature and the true distances are larger still. Recurrence of a feature in counterfactual attribution reflects a large |coefficient| difference — a property of the fit — and is **not** evidence of exploitable missing information.

---

## 6. Draw Limitation

| Measure | fold_1 | fold_2 | fold_3 | Class |
|---|---:|---:|---:|---|
| Draw prevalence | 0.242474 | 0.264269 | 0.249429 | PROVEN |
| H1 `p_draw` ROC-AUC → Draw | **0.546704** | **0.564668** | **0.526820** | PROVEN, REPLICATED |
| Mean `p_draw`, true-D | 0.254864 | 0.255122 | 0.257595 | PROVEN |
| Mean `p_draw`, non-D | 0.248021 | 0.244764 | 0.251460 | PROVEN |
| Separation | +0.006843 | +0.010358 | +0.006135 | PROVEN, REPLICATED |
| True-D ranked 1st | **0.0000** | **0.0000** | **0.0000** | PROVEN |
| D-in-top-2 lift | −0.002409 | −0.032015 | −0.031629 | PROVEN |
| Balanced draw rate | 0.243289 | 0.282895 | 0.283828 | PROVEN |
| Lopsided draw rate | 0.232406 | 0.221667 | 0.225524 | PROVEN |
| Balanced > lopsided | ✓ | ✓ | ✓ | **REPLICATED 3/3** |
| Max single-feature \|AUC−0.5\| | 0.05475 | 0.05434 | 0.03674 | PROVEN |

**Consistently-directed features: 29 of 75** across all three folds; expected under pure direction noise **18.8**; **P(≥29 | noise) = 0.0062** — above chance. **PROVEN.**

**Family-level draw signal** (mean |AUC−0.5|): goals_core 0.02428 · shots_on_core 0.02215 · strength 0.02066 · shots_core 0.01991 · form 0.01789. No family stands out. **PROVEN.**

**Central conclusion.** The data contains **reproducible** draw information — above chance, coherently directed (higher home attacking output ⇒ lower draw likelihood), and replicated in the balance structure. Its **magnitude is weak**: the best single feature reaches AUC ≈ 0.445, and H1's own `p_draw` AUC deviations (0.047 / 0.065 / 0.027) are comparable to a *selection-inflated maximum* over 75 features (0.055 / 0.054 / 0.037). **INFERENCE:** H1 is already extracting approximately the available draw signal; there is no large unexploited pool.

**The model is not perfect.** Draw recall is near zero and `p_draw` ranking is only slightly better than chance. This section does not claim otherwise.

---

## 7. Structural Investigation

**Load-bearing finding — model-free, computed from `features.db` labels alone:**

| Fold | Max empirical P(D) across deciles | Deciles where Draw is modal | Min gap `max(P_H,P_A) − P_D` |
|---|---:|---:|---:|
| fold_1 | **0.2842** | **0 of 10** | +0.1695 |
| fold_2 | **0.3295** | **0 of 10** | +0.0568 |
| fold_3 | **0.2965** | **0 of 10** | +0.0632 |

**PROVEN, REPLICATED 3/3.**

This is a property of the **observed data distribution**, not something H1 imposes. Across every `strength_diff` decile of every validation fold, at least one of Home or Away is empirically more likely than Draw. **INFERENCE:** an oracle calibrated to this observed distribution would likewise not select Draw as argmax in these regions, and the negative D-in-top-2 lift is a consequence of correct probability ordering — draws concentrate in balanced fixtures, where `p_H` and `p_A` are both moderate and both exceed `p_D`.

**Explicitly NOT claimed:** that Draw can never be the correct prediction. Draws occur in 24–26% of matches and are frequently the realised outcome. The finding is about *conditional modal outcome within the observed feature space at decile resolution* — nothing more. A finer partition, a different feature space, or different data could behave differently.

---

## 8. Why Proposed Fixes Were Rejected

| Candidate | Evidence | Verdict |
|---|---|---|
| **1. Family scaling / 1/√n (H3-A)** | Premise falsified (§4); size-normalised influence did not rise with family size across all cells | **NOT JUSTIFIED** |
| **2. Nonlinear model, same 76 columns** | The peaked draw structure is *already expressed* by the softmax (model `p_draw` 0.2679 vs actual 0.2682 in the closest-balance quartile); counterfactual distances 11.7–14.8 SD contradict a nearby better surface | **NOT JUSTIFIED** |
| **3. Interaction terms from existing features** | This is H4 — tested, fold_3 reversed direction, Bonferroni-adjusted best p = 0.080 | **NOT JUSTIFIED** (already tested) |
| **4. Calibration** | Platt **+0.004744** and isotonic **+0.049713** worse on nested validation; confident-wrong carries only 10.1–13.5% of NLL | **NOT JUSTIFIED** (worsened) |
| **5. Draw-specific threshold / decision rule** | Draw is modal in **0 of 30** fold-deciles; forcing draws would necessarily worsen log loss, the primary metric | **CONTRAINDICATED** |
| **6. Existing-feature addition / removal** | Signal is broadly distributed across correlated encodings of one construct; no family stands out; H1 dedup was real but inert | **NOT JUSTIFIED** |
| **7. Temporal explanation** | Three folds cannot establish a trend (chance probability 1/3) | **NOT ESTABLISHED** |
| **8. Counterfactual repair** | Distances are an explicit **upper bound**; 0%/0%/0.92% within 1 SD; recurrence reflects coefficient geometry, not missing signal | **NOT JUSTIFIED** |

---

## 9. Model Ceiling Classification

**PRIMARY LIMITATION — A: weak information magnitude.** The contract contains reproducible signal (P = 0.0062 above chance) but at a magnitude too small to change class assignment: best single-feature AUC ≈ 0.445, family effects 0.018–0.024, `p_draw` AUC 0.527–0.565. 84–89% of errors are low-confidence.

**SECONDARY STRUCTURAL EFFECT — D: draw suppression.** Draw is never the empirical modal outcome in any observed decile, so no calibrated model should select it as argmax there. This is a property of the label distribution in this feature space, not a model defect — hence *structural effect*, not *limitation to be fixed*.

**REJECTED ALTERNATIVES:** **B** (model-form) — falsified on three independent grounds (§8.2); **C** (calibration) — tested, worsened; **E** (temporal) — not establishable from three folds; **F** (no actionable limit) — rejected, because a real limitation *was* identified, just not a correctable one.

**Summary in one line:** *weak information magnitude, plus structural Draw suppression arising from the observed label distribution.*

---

## 10. What We Actually Know

### PROVEN / REPLICATED
H1 reproduces exactly (<1e-9, deterministic). Mean validation log loss **0.9915427871369671**, Brier **0.5913870749997768**. Calibration worsened validation (+0.004744 / +0.049713). H2-C worsened all three folds (+1.461489e-03). H1's dedup mechanism is real (P1/P2/P4) but its effect (−2.915e-05) is below solver tolerance. 29/75 features consistently directed (P = 0.0062). Balanced fixtures draw more often, 3/3 folds. Draw is modal in 0 of 30 fold-deciles. 84–89% of errors are low-confidence. Counterfactual distances 11.7–14.8 SD. 13/13 pinned baselines identical.

### STRONG INFERENCE
H1 already extracts approximately the available draw signal. The negative D-in-top-2 lift reflects correct ordering rather than a defect. Eliminating every confident error would leave the large majority of NLL untouched. The draw signal is one weak construct measured many correlated ways, not an isolated missing variable. Counterfactual distances understate the true difficulty because they hold correlated features fixed.

### UNKNOWN
Whether a materially better model exists for this problem given *different* information. Whether finer-resolution or multivariate partitions would reveal regions where Draw is modal. Whether a nonlinear model would help if given genuinely new inputs. The true irreducible Bayes error for football match outcomes. Whether the rejected mechanisms would behave differently under a different validation protocol or more folds. Per-fold H1 accuracy and several ceiling-diagnostic sub-results (**NOT VERIFIED** here).

**This project stopped because further experimentation on the existing contract was no longer justified — not because the absence of a better model was proven.** Those are different claims, and only the first is supported.

---

## 11. What Would Justify Reopening?

A future modelling cycle may be authorized only if **at least one** becomes true:

1. **Genuinely new predictive information** becomes available that is not a transformation of the existing 76 columns.
2. **A new data source** provides event-level or other constructs absent from the current provider, with auditable provenance.
3. **A previously untested construct** is demonstrably available, auditable, and temporally safe under the existing leakage rules.
4. **A new modelling hypothesis** is supported by evidence *independent of the current error* — error-driven hypothesis generation has now been tried four times (H1–H4) and produced one inert result and three rejections.
5. **A validation-protocol change** is required for a clearly stated reason, pre-registered before execution.

**No paid API is assumed necessary. No provider is recommended.** Any purchase requires explicit approval under the standing project rules.

---

## 12. Data Acquisition Boundary

From `DATA_AUDIT.md` (verbatim): event-level data — goal minute, card minute, substitution minute — is **"NOT AVAILABLE (as discrete events)"**. A `coverage.goal_timings` flag exists and goals are split 1H/2H, but *"no endpoint in the 44-endpoint collection exposes a per-event list (no `/fixtures/:id/events`, no minute-by-minute goal/card log)."* The audit's instruction is to *"treat as NOT AVAILABLE for event-level feature engineering; re-open with OddAlerts support if finer granularity is later required."*

**Present:** fixtures, results, goals, team statistics, shots, shots on target, possession, corners, fouls, cards, xG (post-cutover), historical odds. **Absent:** discrete match events, minute-level timing, and any draw-specific construct beyond what the current aggregates already encode.

**Conclusion — supported by the evidence:** *further progress appears to be a data-acquisition question rather than an existing-model tuning question.* Whether that acquisition is possible, necessary, or affordable is **not** established here and is explicitly out of scope.

---

## 13. Final Authorization State

```
MODEL:                  FROZEN
FEATURE CONTRACT:       FROZEN
H1:                     FROZEN BASELINE
NEW EXPERIMENT:         NOT AUTHORIZED
H5:                     NOT AUTHORIZED
2025/26:                UNTOUCHED
PRODUCTION:             UNMODIFIED
features.db:            READ-ONLY
TUNING:                 NONE
CALIBRATION:            NONE
THRESHOLD CHANGE:       NONE
FEATURE SELECTION:      NONE
ARTIFACTS:              NONE
ESTIMATOR PERSISTENCE:  NONE
```

---

## 14. Final Phase Closure Statement

This cycle tested four mechanisms against pre-registered criteria on pre-2025/26 validation data only. One (H1 de-duplication) was mathematically confirmed but empirically inert at a magnitude below the solver's own convergence tolerance. Three (H2-C, H3, H4) were rejected — one because it worsened every fold, one because its premise was falsified, one because it failed to replicate and reversed direction in a fold. Calibration had already been tested and had worsened validation. The remaining error is dominated by low-confidence predictions, and the draw limitation traces to a property of the observed label distribution rather than to a correctable model defect.

**The cycle is closed because the available evidence no longer justified another experiment on the existing information contract — not because a better model was shown not to exist.** That distinction is preserved deliberately: the evidence supports stopping, and does not support any stronger claim.

---

## 15. Next Phase

> ### PHASE D — DATA / INFORMATION AUDIT

**Purpose:** determine whether genuinely new predictive information exists outside the current 76-column contract, and what evidence would be required to justify reopening modelling.

**Explicitly excluded:** provider purchase, API integration, feature addition, model training, tuning, calibration, and any 2025/26 access.

**Permitted:** read-only inventory of what information could exist, what the current provider does and does not expose, and definition of the evidence bar for reopening.

---

## Integrity Record

| Item | Status |
|---|---|
| Files changed (this closure task) | **1** — this document only |
| Files added across the improvement cycle | 9 experiment/diagnostic scripts (`run_*.py`), all additive, none imported by production |
| Production files modified | **ZERO** (`train.py` `21425459…`, `config.py` `c2ed32cb…`) |
| `features.db` modified | **ZERO** (`e7ebe7fc…`, read-only throughout) |
| 2025/26 accessed | **NO** (beyond the single authorized final BEFORE/AFTER test, which is locked) |
| Experiments run in this task | **ZERO** |
| Estimators persisted | **ZERO** |
| Artifacts created | **ZERO** (audit artifacts unchanged at 38) |
| Tuning / calibration / threshold change / feature selection | **NONE** |
| Pinned baselines | **13/13 byte-identical** |

---

```
FINAL PHASE STATUS:
CLOSED

CURRENT MODEL:
H1 v1.0

CURRENT BASELINE:
0.9915427871369671 mean log loss

NEW EXPERIMENT:
NOT AUTHORIZED

NEXT PHASE:
DATA / INFORMATION AUDIT ONLY
```
