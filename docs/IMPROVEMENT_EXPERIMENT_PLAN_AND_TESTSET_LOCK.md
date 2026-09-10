# Improvement Experiment Plan and 2025/26 Test-Set Lock

**Document type:** governance + experiment design. Planning only — nothing implemented, nothing trained, nothing changed.

**Date:** 2026-08-18

> **2025/26 IS LOCKED AS THE FINAL UNTOUCHED TEST SET.**
> **V1 — FROZEN · MODEL_VERSION v1.0 · CANDIDATE — NOT YET DEFINED, NOT AUTHORIZED**

---

## 0. The BEFORE baseline (recorded, not to be modified)

| Metric | Value |
|---|---|
| Test fixtures | 1,751 |
| Correct / incorrect | 902 / 849 |
| Accuracy | 0.5151342090234152 |
| Log loss | 0.9960487649063601 |
| Brier | 0.5947186313129067 |
| Macro-F1 | 0.3938491505819635 |
| Balanced accuracy | 0.4489548797121166 |

**This baseline is frozen and must not be overwritten.** It is quoted here for the record only. **No number in this document's analysis is derived from it**, and no improvement below was chosen by looking at it.

---

## 1. V1 Diagnostic Analysis — pre-2025/26 evidence only

Every figure in this section comes from the three approved walk-forward validation folds or from recorded Phase 3/4A/4B/5A artifacts computed on those folds. **2025/26 labels, outcomes, and error patterns were not inspected for this analysis.**

### 1.1 Feature-family behaviour

The 80-column Model B contract comprises six families: `goals_core` (18), `form` (20), `strength` (5), `competition_id` (1), `shots_core` (18), `shots_on_core` (18). Recorded per-family maximum null rates on non-test partitions range from 0.0027 (`strength`) to 0.0772 (`shots_core`/`shots_on_core`, fold_1 train), with `competition_id` at 0.0000 throughout. **No family is all-null or constant in any partition.** Missingness is concentrated in early-season rows with insufficient history, not in feed gaps.

### 1.2 Validation-fold performance

Model B, C=1.0, mean validation log loss **0.9993791056968738**; per fold: fold_1 **1.010986668383243**, fold_2 **0.9909502394658608**, fold_3 **0.996200409241518**. Spread across folds ≈ 0.020 — an order of magnitude larger than any candidate improvement discussed below, which is the single most important calibrating fact in this document.

Reference points from the same folds: frequency baseline mean log loss **1.0736742539173472**; HistGradientBoosting **1.08914…** (rejected in Phase 3); Model A (44 columns) **1.004013383939449**.

### 1.3 H/D/A class imbalance behaviour

Validation base rates (n=5,331): H **0.43631588820108796**, D **0.25192271618833240**, A **0.31176139561057964**. Argmax share: H **0.6379666103920465**, D **0.0082536109547927**, A **0.3537797786531607**. The model over-selects Home relative to its base rate and almost never selects Draw.

### 1.4 Draw prediction behaviour

Mean predicted P(draw) **0.2538216306193289** vs empirical **0.2519227161883324** — gap **+0.0018989144309965**. Draw is ranked 1st in **0.83%**, 2nd in **59.03%**, 3rd in **40.14%** of fixtures. The failure is at the **argmax level**, not in the probability mass: P(draw) would have to exceed both P(H) and P(A) simultaneously, and its ceiling sits below that. This is the accepted Condition 2 limitation (`DRAW_LIMITATION_ACCEPTED`, scoped to the probability-output contract).

### 1.5 Probability quality

Top-label ECE (Guo et al. 2017, equal-width bins) on validation folds: **0.005085147705460211** — already small. Mean predicted vs actual by class: H 0.4233 vs 0.4363 (−0.0130), D 0.2538 vs 0.2519 (+0.0019), A 0.3229 vs 0.3118 (+0.0111). **Aggregate calibration is good; there is little headroom here.**

### 1.6 Error patterns from pre-2025/26 validation data

The dominant structural error is Draw's near-zero argmax share (§1.4), which is a known and accepted property rather than a defect to be tuned away. Home over-selection follows from the same mechanism. No further per-fixture error mining was performed, because doing so on validation folds risks the same overfitting-by-inspection that the test-set lock exists to prevent.

### 1.7 Feature redundancy / weak or unstable groups

Phase 4A established Model B (80 cols) beats Model A (44 cols) by **−0.004634278243** mean log loss, and Phase 4B confirmed B beats A at **9 of 9** C values and in **every fold at every C** — an unusually consistent result. Phase 5A tested re-adding `league_home_advantage_season`: mean log loss **0.9993791056968738 → 0.9996821579734537**, delta **+0.000303052276579896** (`CASE_B_worsens_mean_log_loss`). **Condition 1 is closed: that feature stays out.**

### 1.8–1.10 Ablation and robustness evidence

Phase 4B swept **C ∈ {0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0}** on validation folds only. Model B results:

| C | Mean log loss | Brier | Accuracy | Macro-F1 | Balanced acc. |
|---:|---:|---:|---:|---:|---:|
| **0.01** | **0.9957166199** | 0.594245 | 0.520238 | 0.399725 | 0.454188 |
| 0.10 | 0.9989624678 | 0.596281 | 0.520595 | 0.403035 | 0.454829 |
| 0.25 | 0.9993309136 | 0.596492 | 0.519136 | 0.403014 | 0.453796 |
| 0.50 | 0.9994149568 | 0.596524 | 0.518223 | 0.403216 | 0.453113 |
| **1.00 (V1)** | **0.9993791057** | 0.596502 | 0.518414 | 0.403484 | 0.453293 |
| 2.00 | 0.9994021983 | 0.596518 | 0.518778 | 0.403782 | 0.453616 |
| 4.00 | 0.9991150869 | 0.596338 | 0.518945 | 0.404155 | 0.453656 |
| 10.00 | 0.9991081865 | 0.596351 | 0.519326 | 0.404856 | 0.454203 |
| 100.00 | 0.9991863127 | 0.596444 | 0.518937 | 0.405416 | 0.453878 |

C=0.01 is best on log loss by **−0.0036624858** vs C=1.0 (0.3665% lower) and **improves in all three folds individually** (fold_1 1.010987→1.005422, fold_2 0.990950→0.987418, fold_3 0.996200→0.994310). Brier and balanced accuracy also improve; **macro-F1 is slightly worse** (0.399725 vs 0.403484).

Phase 4B recorded two guards that this plan must honour, verbatim:

> `"model_b_best_at_grid_edge": true` — *"A best-C at the grid edge suggests the optimum may lie outside the frozen grid (over- or under-regularization); this is reported, not acted upon."*

> *"Robustness is descriptive only. No statistical test was performed, so no significance claim may be made. No causal claim. This does not select a production model and does not authorize changing V1."*

### 1.11 Calibration evidence

Phase 3 Step 4 compared methods under a **nested expanding-window** protocol (calibrators fit only on strictly earlier folds; 2025/26 never present):

| Method | Mean log loss | Mean Brier | Mean ECE | Δ vs uncalibrated |
|---|---:|---:|---:|---:|
| **uncalibrated** | **1.0006233634** | 0.597828 | 0.011659 | — |
| platt_sigmoid | 1.0053671616 | 0.601091 | 0.023142 | **+0.004744** |
| isotonic | 1.0503368331 | 0.601849 | 0.028463 | **+0.049713** |

**Both calibration methods made validation log loss worse.** Calibration is an evidence-backed dead end, not an untried idea.

### 1.12 Temporal-safety concerns already discovered

The simultaneous-kickoff league-accumulator discrepancy (spec `unix < T.unix` vs implementation `(unix ASC, fixture_id ASC)`) is documented in `docs/GATE10_LEAKAGE_IMPACT_DECISION_RECORD.md`. It affects 2,877 of 10,735 fixtures and four of the 80 contract columns. **Its governance decision is OPEN and it is deliberately excluded from this improvement plan** (see §3.3).

### 1.13 Other opportunities

xG (22 columns) remains structurally unusable: folds 1 and 2 have **100% missing xG in their training partitions**, so Models C/D cannot be ranked on the 3-fold mean and Phase 4A refused to fabricate one.

---

## 2. Eligible improvement opportunities

| # | Opportunity | Evidence (pre-2025/26 only) | Eligible? |
|---|---|---|---|
| E-1 | **Extend the C grid below 0.01** | Phase 4B: best-C at grid edge, flagged for exactly this; improvement consistent across all 3 folds | **YES** |
| E-2 | Class weighting for Draw (`class_weight`) | Never tested; §1.4 shows an argmax-level failure that weighting targets | Eligible to *test*, but see §3.2 |
| E-3 | Strict-temporal feature reconstruction | Leakage decision record | **Eligible only under separate authorization** (§3.3) |

## 3. Ineligible or excluded opportunities

| Opportunity | Status |
|---|---|
| Anything chosen by inspecting 2025/26 errors, per-fixture mistakes, or the 849 incorrect predictions | **NOT ELIGIBLE — requires locked test-set information** |
| Tuning toward the 2025/26 log loss / accuracy figures | **NOT ELIGIBLE — requires locked test-set information** |
| Calibration (Platt / isotonic) | **Excluded on validation evidence** — both worsened log loss (§1.11) |
| Re-adding `league_home_advantage_season` | **Excluded** — Phase 5A, Condition 1 closed |
| Adding xG features | **Excluded** — structural coverage gap in folds 1–2 |
| Switching estimator family (e.g. back to HistGradientBoosting) | **Excluded** — Phase 3 rejected it on validation log loss; also violates "no unnecessary architecture change" |

---

## 4. Recommended smallest improvement experiment

### E-1 — Extend the regularization grid downward, revalidate on folds 1–3 only

**Change:** one hyperparameter, `C`. Nothing else. Same Model B family, same 80 columns and order, same V1 preprocessing, same H/D/A problem definition, same estimator class, no calibration, `max_iter=2000`, `random_state=0`.

**Rationale.** Phase 4B's own recorded flag says the optimum may lie outside the frozen grid. The grid was never extended below 0.01, so the true minimum is unobserved. This is the only improvement in the evidence base that is (a) supported by existing pre-2025/26 measurements, (b) a single-parameter change, and (c) explicitly anticipated by a prior phase.

**Evidence.** §1.10. Improvement at C=0.01 is present in **every fold**, not just the mean — the property most likely to generalise.

**Expected effect.** A modest reduction in mean validation log loss. **Stated honestly: the expected effect is small and may be zero or negative.** The C=1.0 → C=0.01 gain is 0.0037, while the fold-to-fold spread is ≈0.020 — roughly 5× larger. This experiment may well conclude "no change is warranted," and that is a legitimate and useful outcome, not a failure.

**Proposed grid.** `C ∈ {0.0001, 0.0005, 0.001, 0.005}` added to the existing values, so the extended sweep brackets any interior minimum. If the best C is *again* at the new edge, the correct response is to report the edge flag again — **not** to keep extending until something wins.

**Rollback condition.** Retain V1's `C=1.0` if **any** of: (a) the best extended-grid C does not beat C=1.0 on mean validation log loss; (b) it beats the mean but not in all three folds individually; (c) it worsens mean Brier; (d) determinism fails; (e) the best C lands at the new grid edge again.

**Pre-registration.** The grid, the primary metric (mean validation log loss), the all-folds requirement, and the rollback conditions above are fixed **now**, before the experiment runs.

### E-2 — Draw class weighting (proposed but NOT recommended first)

Untested and evidence-motivated, but it targets argmax behaviour that Condition 2 already accepted as out of contract, and class weighting reliably trades log loss (the primary metric) for class-assignment metrics. It also reopens a decision prior governance deliberately declined ("do not tune class weights"). **Recorded as available; not proposed for this round.**

### 3.3 — Simultaneous-kickoff temporal fix: SEPARATE AUTHORIZATION REQUIRED

**A strict-temporal candidate requires separate authorization and must NOT be bundled into E-1.** It would change feature-generation semantics, require regenerating `features.db` (a pinned baseline, `e7ebe7fc07040a5927683c35b6371e63`), break the Gate 9 checksum verification and §11 rollback evidence, and invalidate the reproducibility of every Phase 3/4A/4B/4C/5A artifact. **It is not silently fixed here, and E-1 does not depend on it.** Its decision remains OPEN per the leakage decision record.

**Confounding warning:** running E-1 and the strict-temporal change together would make the result uninterpretable — two variables, one measurement. They must be sequenced, never combined.

---

## 5. Validation protocol

1. **Data:** pre-2025/26 only. The three approved walk-forward folds via the existing `iter_walk_forward_folds`. **2025/26 is never loaded.**
2. **Comparison:** CURRENT FROZEN V1 (Model B, C=1.0) vs CANDIDATE (Model B, extended-grid C), identical in all other respects.
3. **Metrics:** log loss (**primary**), Brier, accuracy, macro-F1, balanced accuracy, plus per-class precision/recall/F1/support — reported per fold and as means.
4. **Selection rule (pre-registered):** lowest mean validation log loss **and** improvement in all three folds individually. Brier must not worsen. Secondary metrics are reported, never decisive.
5. **Determinism:** each configuration fitted twice; predictions must be byte-identical.
6. **No regression:** the candidate must reproduce Phase 4B's recorded value at C=1.0 exactly, proving the harness is faithful before any new C is trusted.
7. **No significance claim.** No statistical test is authorized, so no result may be described as significant.
8. **Freeze:** if the candidate wins under the pre-registered rule, it is frozen — configuration, code path, and checksum — **before** any 2025/26 access is requested.

---

## 6. Risks and governance implications

| Risk | Mitigation |
|---|---|
| Improvement smaller than fold-to-fold noise | Require all-folds improvement; pre-register; make no significance claim |
| Grid-edge chasing | Stop-and-report if the optimum lands at the new edge; do not extend again |
| Confounding with the temporal fix | Strictly sequenced; never combined |
| Scope creep into V2 | E-1 changes one hyperparameter; it does not create V2 and does not change `MODEL_VERSION` |
| Test-set leakage by inspection | §7 lock; candidate frozen before any 2025/26 access |
| Reopening frozen artifacts | E-1 requires **no** change to `features.db`, frozen V1 source, or any pinned artifact |

**Authorization still required before implementing E-1:** an explicit instruction to run a hyperparameter experiment, since standing governance has repeatedly forbidden tuning. E-1 is *designed*, not *approved*, by this document.

---

## 7. 2025/26 Test-Set Lock — governance statement

> **2025/26 is locked as the final untouched test set.**

The candidate must **not** access 2025/26 `y_true`, match outcomes, performance metrics, or error analysis until the candidate is **completely frozen**. No 2025/26 information may inform training, feature engineering, feature selection, hyperparameter tuning, threshold selection, calibration, model selection, error-driven development, or the choice between candidates.

**Only after** the candidate is frozen and all pre-2025/26 validation is complete will **one** final 2025/26 retest be authorized:

- **BEFORE:** frozen V1 → 2025/26
- **AFTER:** frozen candidate → same 2025/26

Same fixtures, same actual outcomes, same evaluation metrics. **One retest, once.** If the retest disappoints, the correct response is to record it — not to iterate against the test set.

**Confirmed for this task:** 2025/26 was not accessed, inspected, or used in any way while producing this analysis. The BEFORE baseline in §0 is quoted from the existing locked record and informed none of the reasoning above.

---

## 8. Exact next implementation step

**Nothing is implemented yet.** The next step requires your explicit authorization of **one** of:

- **(A)** Authorize E-1 — extended-C validation experiment on folds 1–3 only, under the §5 protocol, additive script, no artifact overwrite, no `MODEL_VERSION` change.
- **(B)** Authorize E-2 instead or additionally (class weighting), accepting that it reopens a previously declined decision.
- **(C)** Authorize the strict-temporal candidate first (§3.3), accepting the pinned-artifact and regeneration consequences.
- **(D)** Decline all three and retain frozen V1 as-is.

**No option is selected here, and none is implied by the ordering.**

---

## 9. Status

> **V1 — FROZEN · MODEL_VERSION v1.0**
> **CANDIDATE — DESIGNED, NOT IMPLEMENTED, NOT AUTHORIZED**
> **2025/26 — LOCKED, UNTOUCHED**
> **LEAKAGE GOVERNANCE DECISION — STILL OPEN**
