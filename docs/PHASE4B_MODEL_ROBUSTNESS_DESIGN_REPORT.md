# Phase 4B — Model Robustness / Sensitivity Study: Design & Diagnostic Report

## **Phase 4B training has NOT started.**

No Phase 4B experiment has been run. No Phase 4B result artifact exists. No Phase 4B number appears anywhere in this document, because none has been produced. The runner is implemented and tested but deliberately not executed; running it is a separate, explicit instruction.

---

## 1. Objective

Determine whether Model B's Phase 4A advantage over Model A — mean validation log loss `-0.004634278243` — is robust to reasonable LogisticRegression regularization choices, or whether it depends on the arbitrary default `C=1.0`.

This is **not** a V2 selection exercise, not model selection, and not a production decision. V1 remains frozen and untouched.

## 2. Pre-implementation integrity baseline (recorded before any file was written)

| Artifact | MD5 |
|---|---|
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` |
| `phase3_model_comparison.json` | `616279914b2730749d52eae15b5f96b9` |
| `phase3_calibration_comparison.json` | `d94ed430ab13337797f0592bf82878cb` |
| `phase3_probability_diagnostics.json` | `0a8ee228b394108d338c5495478227b7` |
| `phase3_step3_diagnostic_column_report.json` | `5b5fe15600381106651a13a0f88a53ee` |
| `phase3_baseline_eval.json` | `6b71fdd41128f1a4e75a4b75cf148d45` |
| `phase3_target_audit.json` | `fb7c0b342362d0f60d2f896e1df53ee7` |
| `phase4a_ablation_comparison.json` | `075b0686bce20bfce9f7289fa37076c0` |
| `phase4a_feature_group_diagnostic.json` | `e5b343372feefb62b357ba012febdb8f` |

Phase 4A prediction CSVs (8) and Phase 3 prediction CSVs (6) also checksummed; V1/Phase 4A source files (`config.py`, `train.py`, `run_experiments.py`, `calibration.py`, `ablation.py`, `run_ablation.py`, `splits.py`, `data.py`, `evaluate.py`, `baselines.py`) checksummed. **None of these was modified.** Phase 4B adds only new files.

## 3. Stop-condition diagnostic — all 10 checked

| # | Condition | Result |
|---|---|---|
| 1 | Phase 4A feature definitions ambiguous | **Clear** — A=44, B=80 columns, imported unchanged from `ablation.py`; B = A ∪ shots ∪ shots-on-target verified exactly; no duplicates |
| 2 | V1 LogisticRegression config not reproducible | **Resolvable — see §4** (reported, not worked around silently) |
| 3 | C grid conflicts with a frozen contract | **No conflict** — `config.py` declares no regularization constant |
| 4 | Required feature missing | **None missing** — all 44 and all 80 columns present in the live schema |
| 5 | Unexpected schema/missingness behaviour | **None** — across all three training folds, both models have 0 all-null and 0 constant columns; max null rate 5.4–7.7% (ordinary early-season history gaps) |
| 6 | Final-test data accidentally required | **Not required** — no code path loads it |
| 7 | Requires modifying V1 production code | **Does not** — see §4 |
| 8 | Experiment would require xG | **Does not** — A and B are xG-free; verified disjoint from all 22 xG columns |
| 9 | Unexpected Phase 4A outputs | **None** — only the 6 known Phase 4A paths exist |
| 10 | Integrity checksum changed unexpectedly | **No** — all baselines above verified stable |

## 4. Design constraint requiring explicit disclosure

`train.train_logistic_regression()` hardcodes `LogisticRegression(max_iter=2000, C=1.0, random_state=0)` and **exposes no `C` parameter**. Sweeping C through it would require editing V1 production code, which is forbidden.

**Resolution (no V1 modification):** `robustness.train_logistic_regression_with_C()` reuses the *unmodified* V1 building blocks — `train.LogisticRegressionPreprocessor` (fit on the training partition only) and `train._reorder_proba` (CLASS_ORDER handling) — and constructs the estimator with kwargs identical to V1's except C. Preprocessing, imputation, scaling, `competition_id` one-hot encoding, solver, penalty, tolerance and `random_state` are all V1 behaviour.

This is disclosed rather than assumed safe, and is backed by two mechanical guards:

1. **AST guard (no scikit-learn needed):** the test suite parses `train.py`, extracts the actual `LogisticRegression(...)` kwargs, and asserts `V1_LOGREG_BASE_KWARGS` equals them minus `C`, and that `V1_C_VALUE` equals V1's hardcoded C. If V1's configuration ever changes, this fails loudly instead of the study silently diverging from the model it claims to probe.
2. **Equivalence guard (scikit-learn required, runs at execution time):** `verify_phase4a_equivalence_at_C1()` asserts that at C=1.0 this path reproduces Phase 4A's recorded mean log losses (`1.004013383939449` / `0.9993791056968738`) to 1e-9. A mismatch invalidates the whole comparison and must stop the phase rather than be reinterpreted.

## 5. Frozen experimental design

**C grid (frozen, 9 values, declared before training):** `0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0` — four orders of magnitude around V1's C=1.0, which sits inside the grid rather than at an edge. Not an automated search; `build_logistic_regression()` raises on any off-grid C.

**Models:** Model A (44 columns) and Model B (80 columns), imported unchanged from `ablation.py`. Models C and D are excluded — they require xG, which Phase 4A established cannot be evaluated across the 3-fold protocol.

**Protocol:** 2 models × 9 C values × 3 folds = **54 training runs**. For each cell, fit on that fold's training partition only, evaluate on its validation partition only. Metrics recorded per cell: log loss, Brier, macro-F1, balanced accuracy, accuracy — plus per-fold values, not only means. Primary basis: mean validation log loss across folds 1–3.

**Unchanged from V1/Phase 4A:** feature groups, walk-forward boundaries, preprocessing, imputation, scaling, `competition_id` handling, `CLASS_ORDER`, solver, `random_state`, probability validation. No calibration, no random splitting, no K-fold, no xG, no feature engineering, no feature selection.

## 6. Final-test protection

2025/26 is structurally unreachable from the Phase 4B path, enforced and tested four ways: `splits.final_split` is never imported or called; `FINAL_TEST_SEASONS`/`FINAL_TRAIN_SEASONS` are never imported or referenced; the only split source is `iter_walk_forward_folds` (which structurally yields folds 1–3 and re-verifies temporal safety); and no public function in either Phase 4B module accepts a parameter whose name contains "final" or "test_season" — checked by signature introspection across the whole module.

`analyze_robustness()` accepts exactly one parameter (`summary`), so there is no argument through which final-test data could reach the analysis.

## 7. Pre-declared robustness questions

The runner computes answers to all ten before any interpretation: best C for A and for B; whether B beats A at the same C; at how many C values; per-fold win pattern at each C; the range of B−A deltas; a leave-one-C-out check for whether the conclusion hinges on a single grid point; the delta specifically at C=1.0; directional consistency of Brier/accuracy/macro-F1/balanced accuracy; and flags for a best-C sitting at a grid edge (a signal of over- or under-regularization, reported rather than acted upon).

Every analysis output carries an embedded interpretation guard stating that no statistical test was performed (so no significance claim is permitted), no causal claim is made, no production model is selected, and V1 is not authorized to change.

## 8. Implementation plan

**New files only — nothing existing modified:**
- `src/models/robustness.py` — frozen grid, V1-equivalent estimator construction, feature selection.
- `src/models/run_robustness.py` — runner, summarization, robustness analysis, Phase 4A equivalence check. Writes to `data/audit/phase4b_robustness_comparison.json` and `data/audit/phase4b_predictions/` only.
- `tests/test_model_robustness.py` — 47 tests.

`summarize_by_model_and_C()` raises if any (model, C) cell lacks all 3 folds rather than averaging a partial set — carrying forward Phase 4A's "no fabricated means" rule.

**Execution (separate instruction, not performed):** `python -m models.run_robustness`

## 9. Test results

**Phase 4B targeted:** 47 tests, 47 passed, 0 failed, 2 skipped (scikit-learn-only estimator-construction tests).
**Full suite:** **351 tests, 351 passed, 0 failed, 0 errors, 34 skipped.** Up from 304/32 — the 47 new tests add 2 documented scikit-learn-only skips. No existing test was weakened, removed, skipped or altered.

One defect was found and fixed in my own new code during testing: `build_logistic_regression()` originally imported scikit-learn before validating grid membership, so it could not reject an off-grid C in a scikit-learn-free environment. Validation now runs first. This was a real ordering bug in Phase 4B code, not a test accommodation.

## 10. Integrity confirmation

`features.db` and `matches.db` checksums unchanged. All Phase 3 audit artifacts unchanged. Phase 4A authoritative artifact and all 8 prediction CSVs unchanged. Archived incomplete Phase 4A run untouched. `config.py`, `train.py`, `run_experiments.py`, `calibration.py`, `ablation.py`, `run_ablation.py` unchanged. `MODEL_VERSION` v1.0, `REQUIRED_FEATURE_VERSION` v1.0, 15 approved V1 columns, `CLASS_ORDER` H/D/A, 3 folds, `FINAL_TEST_SEASONS` ('2025/2026',) — all verified intact by test.

## 11. Known limitations of the planned study (stated in advance)

- Three folds is a small basis; a consistent sign across C values is suggestive, not conclusive, and no significance test is planned or claimed.
- The grid is discrete and pre-declared; a best-C at an edge would indicate the optimum may lie outside it, which will be reported rather than chased.
- Only regularization strength is swept. Solver, penalty and class weighting are held at V1 values by design, so this measures sensitivity to C alone.
- Validation-only. Nothing here will be evaluated on 2025/26, so no statement about generalization to the locked test season will be possible.
- Robustness of a difference is not evidence that the difference is *large*. The Phase 4A effect is small in absolute terms and will remain so regardless of this study's outcome.

## 12. Explicit boundary

Not started and out of scope: Phase 4B execution, V2 definition, production deployment, calibration of any tier, xG redesign, final-test evaluation, Poisson/Monte Carlo, betting logic, API, UI.

**Phase 4B training has NOT started.** Awaiting explicit instruction to run `python -m models.run_robustness`.
