# Phase 3 — Step 4: Probability Diagnostics + Calibration

## 1. Objective

Diagnose the probability quality of the already-selected, already-frozen LogisticRegression model (Step 3), investigate the near-absence of predicted Draw outcomes, and determine whether calibration improves validation-fold probability quality — using only out-of-sample walk-forward validation predictions, never the locked 2025/26 test set, for any fitting or selection decision.

## 2. Selected model carried forward

LogisticRegression, approved 15-column feature set (`config.APPROVED_FEATURE_COLUMNS_V1`), selected in Step 3 by lowest mean validation log loss across folds 1–3. Nothing in this step retrains, retunes, or reselects it. Frozen Step 3 numbers (unchanged, reproduced below): mean validation log loss 1.0043186935714623, mean validation Brier 0.6003193827036707, mean validation accuracy 0.5087728341743457; final 2025/26 log loss 1.012643607907126, Brier 0.6059500599589396, accuracy 0.500856653340948.

## 3. Validation prediction methodology

Two data paths were built, both producing the identical Step 1 schema (`fold_name, fixture_id, season_id, y_true, y_pred, p_home, p_draw, p_away, max_probability`):

- `generate_fold_validation_predictions()` — the canonical, reusable mechanism: retrains LogisticRegression on each fold's own training period only (same frozen preprocessing/feature set), predicts on that fold's validation rows. Requires scikit-learn.
- `load_fold_predictions_from_existing_run()` — reads the `logreg_P_H/D/A` columns already present in the fold CSVs `run_experiments.py` produced during Step 3, and joins `season_id` from `features.db` (read-only) by `fixture_id`.

I used the second path to produce the real numbers below, since Step 3's run already exists and is the frozen, approved source. Before trusting it, I verified it's not stale: recomputing log loss/Brier directly from `fold_1/2/3_predictions.csv` and `final_test_predictions.csv` reproduces `data/audit/phase3_model_comparison.json`'s per-fold and final-test numbers exactly (to floating-point precision) — confirmed in `tests/test_model_calibration.py::TestAgainstRealStep3Artifacts`. Row counts also match the known validation-fold sizes exactly (1827 / 1752 / 1752), and every row's `season_id` falls inside that fold's validation season(s) and never its own training seasons (checked directly against `season_ids_for()`).

## 4. Probability diagnostics (folds 1–3 pooled, n=5,331)

| | H | D | A |
|---|---|---|---|
| Mean predicted probability | 0.4233 | 0.2538 | 0.3229 |
| Actual frequency | 0.4363 | 0.2519 | 0.3118 |
| Argmax count / % | 3,401 / 63.8% | 44 / 0.8% | 1,886 / 35.4% |

Mean max probability (confidence): 0.5063. Pooled log loss 1.0044, Brier 0.6004 (both consistent with the Step 3 fold-average numbers, as expected). ECE (top-label, confidence-binned, 10 equal-width bins, definition below): **0.0051** — very low, indicating the model's stated confidence closely tracks its actual accuracy overall.

Mean predicted probability by actual outcome: when the actual result was H, mean predicted P(H)=0.482; when actual was D, mean predicted P(D)=0.259; when actual was A, mean predicted P(A)=0.396. All three are the largest of the three predicted probabilities for their respective actual class, i.e. the model is directionally correct on average even where P(D) trails the other two.

ECE definition used (documented in code): **top-label confidence ECE** (Guo et al., 2017, *On Calibration of Modern Neural Networks*) — bin predictions by the model's own max predicted probability (its confidence in its argmax choice); per bin, compare mean confidence to observed top-1 accuracy; ECE is the size-weighted average of `|mean confidence − accuracy|` across 10 fixed, equal-width bins on [0,1].

## 5. Draw-probability investigation

Mean predicted P(draw) across all validation rows: **0.2538**. Empirical draw frequency: **0.2519**. Gap: **+0.0019** — negligible.

Draw's rank among the three predicted probabilities: ranked 1st (i.e. wins argmax) only **0.8%** of the time, ranked 2nd **59.0%** of the time, ranked 3rd **40.1%** of the time. Draw essentially never has the top probability, but usually has the *second*-highest.

**Verdict: (A), not (B).** The model's Draw probability mass is close to correct on aggregate (0.25 vs. an empirical 0.25 base rate) — this is not a case of the model believing draws are rare when they aren't. Draws simply almost never have the single highest probability among three options, because football's actual outcome distribution rarely makes any one class rise clearly above the others when H and A are both live possibilities. This is expected, structural behavior for a 3-class classifier over an outcome where the "middle" class is systematically the least likely of the three in most fixtures — not evidence of systematic underestimation.

The reliability-bin table (Step 3, below) confirms this rigorously: in the [0.2, 0.3) bin — where **3,865 of 5,331 rows (73%)** of all predicted Draw probabilities land — mean predicted probability is 0.253 and observed frequency is 0.253, an exact match (gap 0.000). The two bins with larger gaps ([0.0,0.1) and [0.4,0.5)) contain only 1 and 2 rows respectively and are not meaningful.

No model change was made based on this finding, per instructions.

## 6. Reliability-bin results

Fixed, equal-width binning (`edges = linspace(0,1,11)`, 10 bins of width 0.1), computed per outcome class. Full tables are in `data/audit/phase3_probability_diagnostics.json`; the load-bearing bins (n ≥ 100):

**Home** — largest gaps are 0.020–0.023 in the 0.1–0.6 range (slight, not alarming); the sparse 0.8–0.9 bin (n=45) shows a 0.109 gap, low-confidence given the small sample.
**Draw** — the dominant bin [0.2,0.3), n=3,865, gap 0.000. Everything else is low-n.
**Away** — gaps mostly 0.005–0.022 through the populated range; the 0.6–0.7 bin (n=226) shows a 0.044 gap, worth watching but not extreme.

Overall: reasonably well-calibrated, with the caveat that Home and Away show mild, consistent *overconfidence* at mid-to-high probability bins (predicted slightly higher than observed in several bins), which is exactly the kind of small, real signal calibration is meant to catch — but see Section 8 for why it didn't help here.

## 7. Calibration methods evaluated

Three methods, compared via a **nested, expanding-window** scheme that respects temporal order and never scores a calibrator on data it was fit on:
- Assessment fold_2: calibrator fit on fold_1's validation predictions only (n=1,827), scored on fold_2 (n=1,752).
- Assessment fold_3: calibrator fit on fold_1+fold_2's validation predictions (n=3,579), scored on fold_3 (n=1,752).

fold_1 is never an assessment target. Methods: **A.** Uncalibrated (identity). **B.** Sigmoid/Platt — one-vs-rest per class, fit via Newton-Raphson logistic regression (pure NumPy, no sklearn dependency — same rationale as `evaluate.py`'s metrics), then renormalized to sum to 1. **C.** Isotonic — one-vs-rest per class via the Pool-Adjacent-Violators Algorithm (pure NumPy), renormalized. Isotonic's minimum-sample guard (≥50 rows/class in the fitting data) was never triggered — both assessment rounds had ample per-class counts (smallest was ~450 draws in the fold_1-only fit).

## 8. Calibration comparison (nested assessment, mean of 2 rounds)

| Method | Mean log loss | Mean Brier | Mean ECE |
|---|---|---|---|
| Uncalibrated | **1.0006** | **0.5978** | 0.0117 |
| Platt (sigmoid) | 1.0054 | 0.6011 | 0.0231 |
| Isotonic | 1.0503 | 0.6018 | 0.0285 |

Both calibrated methods were **worse** than uncalibrated on every primary metric in this nested evaluation, not better. This is consistent with Section 4/6's finding that the uncalibrated model's ECE was already low (0.0051 pooled) — there wasn't much miscalibration for Platt or isotonic to correct, and fitting an extra layer on ~1,800–3,600 rows per round introduced noise rather than removing bias. Isotonic did notably worse than Platt, consistent with it being the more flexible (and more overfitting-prone) of the two on this sample size.

## 9. Selected calibration method

**Uncalibrated — LogisticRegression's raw probabilities, unchanged.** Selection rule (frozen before this decision was made): a candidate must beat uncalibrated mean log loss by more than 0.005 *and* not worsen mean Brier by more than the same margin. Neither Platt (log loss +0.0048, worse not better) nor isotonic (log loss +0.0497) cleared this bar. Calibration was not forced.

## 10. Final-test protection statement

2025/26 was not read, fitted on, or evaluated against at any point before this section. `evaluate_calibration_methods()` and `select_calibration_method()` take no parameter through which final-test data could reach them (verified by signature inspection in tests), and `main()`'s source has `select_calibration_method(...)` textually before `run_final_frozen_evaluation(...)` (also verified by a dedicated test reading the source). The final-test application below is the first and only place 2025/26 predictions are touched in this step, and it runs strictly after the calibration method was already frozen.

## 11. Final 2025/26 results

Since the selected method is "uncalibrated," pre- and post-calibration are identical — no calibration layer was applied.

| | Pre-calibration (= final, since uncalibrated was selected) |
|---|---|
| Log loss | 1.012643607907126 (matches the locked Step 3 number exactly) |
| Brier | 0.6059500599589396 |
| Accuracy | 0.500856653340948 |
| ECE | 0.01732 |

`post_calibration` is `null` in `data/audit/phase3_calibration_comparison.json`, explicitly recorded rather than omitted, so the "no calibration applied" decision is visible in the artifact itself, not just implied by its absence.

## 12. Limitations

- The nested calibration comparison has only 2 assessment rounds (fold_2, fold_3) — a small sample of "did calibration help" trials. A cleaner answer would need more historical seasons, which don't exist yet.
- The top-line draw-probability heuristic (Section 5) uses a fixed ±0.03 threshold I chose and documented, not a statistical test with a significance level — treat it as a sanity check, with the reliability-bin table as the primary evidence.
- Reliability bins with very low n (e.g. Home's 0.8–0.9 bin, n=45; Draw's tail bins, n≤2) produce noisy gap estimates that shouldn't be over-interpreted.
- Calibration was evaluated only for the currently approved 15-feature LogisticRegression; this says nothing about whether HistGradientBoosting or a future xG-inclusive model (C/D ablation) would behave differently under calibration.

## 13. Exact next step

Awaiting your direction. This step's scope explicitly excluded: HGB calibration, xG/shots ablation (Model C/D), Poisson/Monte Carlo simulation, betting logic, hyperparameter tuning, and any production prediction path — none of those were touched.

---

### Test suite results

`python -m unittest discover -s tests -v`: **259 tests, 259 passed, 0 failed, 32 skipped** (all 32 skips require scikit-learn, each names the reason explicitly; up from 217/31 before this step — 42 new tests added in `tests/test_model_calibration.py`, 0 existing tests weakened).

### Artifacts created

- `src/models/calibration.py` (new)
- `tests/test_model_calibration.py` (new, 42 tests)
- `data/audit/phase3_probability_diagnostics.json` — real numbers, produced this session
- `data/audit/phase3_calibration_comparison.json` — real numbers, produced this session
- `data/audit/phase3_predictions/step4_validation_predictions_combined.csv` (5,331 rows) and `step4_final_test_predictions.csv` (1,751 rows)
- `docs/PHASE3_STEP4_PROBABILITY_CALIBRATION_REPORT.md` (this file)

### Data integrity

`data/processed/features.db` and `data/processed/matches.db` MD5 checksums are unchanged from before this step (`features.db`: `e7ebe7fc...`, `matches.db`: `fdeed042...`) — both were only ever opened read-only. No Phase 1/2 artifact, split boundary, or feature definition was touched. 2025/26 remained completely untouched until Section 11 above, which ran strictly after the calibration method was frozen in Section 9.
