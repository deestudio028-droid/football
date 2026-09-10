# Phase 3 — Model Development Report

**Status: implementation complete for V1 (Steps 1–4).** This is the authoritative, consolidated report of what was actually built and actually measured, superseding the earlier `PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md`'s illustrative NumPy sanity-check numbers with real, scikit-learn-trained, test-covered results. It does not change any number reported in the Step 1–4 reports (`PHASE3_STEP1...`, `PHASE3_STEP3_MODEL_COMPARISON_REPORT.md`, `PHASE3_STEP3_DIAGNOSTIC_REPORT.md`, `PHASE3_STEP4_PROBABILITY_CALIBRATION_REPORT.md`) — it consolidates them.

## 1. Target and dataset

**V1 target: 3-class match result, H/D/A**, `label_result` in `feature_rows`. Multiclass classification was chosen over separate binary models, goal-based regression, or a hybrid (full reasoning in the design report §2) because it directly matches the client's stated first requirement and needs one well-calibrated model rather than two.

**10,734 labeled rows** (`SELECT COUNT(*) FROM feature_rows WHERE label_result IS NOT NULL`, confirmed directly against `data/processed/features.db`). Fixture **`420450481`** (Nantes vs Toulouse, Ligue 1 2025/26, ABANDONED — no valid final score) is excluded from every supervised training and evaluation step: it has `label_home_goals`/`label_away_goals`/`label_result` all `NULL` in `feature_rows`, and `models.data.load_supervised_dataset()` filters via `WHERE label_result IS NOT NULL`, then independently re-verifies this fixture never appears in the resulting dataset (`ModelDataError` if it ever does). It is not deleted from `feature_rows` — only excluded from supervised (X, y) pairs.

Class distribution across the 10,734 labeled rows: Home win 42.9%, Away win 31.7%, Draw 25.4% (design report §2) — moderate imbalance, no aggressive resampling used.

## 2. Data leakage prevention

Enforced at both the feature layer (Phase 2, unchanged) and the model layer (Phase 3, new):
- `src/models/data.py`: label columns (`label_home_goals`, `label_away_goals`, `label_result`) and bookkeeping columns (`fixture_id`, `generated_at`, `feature_version`, `season_id`, `unix`, `home_id`, `away_id`) are structurally excluded from the feature matrix `X` via `config.X_EXCLUDED_COLUMNS` — never by convention alone.
- `src/models/splits.py`: every split (`iter_walk_forward_folds`, `final_split`) is defined by season membership, never by row index or a random seed, and every split is passed through `verify_temporal_safety()` before use, which raises `SplitSafetyError` if `max(train.unix) >= min(eval.unix)` or if any `fixture_id` appears on both sides.
- Preprocessing (imputation, scaling, one-hot/categorical encoding, calibration) is fit exclusively on each fold's own training partition — verified mechanically by dedicated tests (`test_model_training.py`, `test_model_leakage.py`, `test_model_calibration.py`), not just documented as a rule.
- 10 mandatory leakage tests (preprocessing-fit isolation, validation-can't-change-statistics, final-test-never-used-for-selection, baseline-probabilities-from-training-labels-only, labels-never-in-X, probabilities-sum-to-1, no-NaN/inf-probabilities, approved-temporal-folds-only, no-random-split-anywhere) are all implemented and passing in `tests/test_model_leakage.py`.

## 3. Temporal validation design

**Random splitting (train_test_split, K-fold, stratified-random) is explicitly rejected** — a walk-forward (rolling-origin) design is used throughout, with 3 validation folds plus one untouched final test season:

| Fold | Train seasons | Validation season | n_train | n_val |
|---|---|---|---|---|
| fold_1 | 2020/21 – 2021/22 | 2022/23 | 3,652 | 1,827 |
| fold_2 | 2020/21 – 2022/23 | 2023/24 | 5,479 | 1,752 |
| fold_3 | 2020/21 – 2023/24 | 2024/25 | 7,231 | 1,752 |
| **Final test (locked)** | 2020/21 – 2024/25 | **2025/26** | 8,983 | 1,751 |

Model selection uses only the mean across folds 1–3. 2025/26 is reserved exclusively for one final, unrepeated evaluation after every other decision (feature set, model choice, calibration method) is already frozen — enforced structurally: `select_model()` (in `run_experiments.py`) and `select_calibration_method()` (in `calibration.py`) each take only a folds-1–3 summary as input, with no parameter through which 2025/26 data could reach them (verified by signature-inspection tests), and both orchestrators (`run_experiments.main()`, `calibration.main()`) call the final-test evaluation strictly after the selection call in source order (verified by source-order tests).

## 4. Feature set — approved V1 subset

**A global, pooled model across all 5 leagues, with `competition_id` included as a categorical feature**, rather than 5 separate per-league models or a hierarchical model — chosen because per-league training would cut each league's training set to roughly a fifth of the pooled size, a real overfitting risk given the feature richness, while the by-league outcome-rate spread (design report §2, ~5 percentage points) isn't large enough to argue the leagues are statistically distinct populations that must never be pooled (design report §7).

**Exactly 15 columns** (`config.APPROVED_FEATURE_COLUMNS_V1`), verified to exist verbatim in `data/processed/features.db` (never invented or substituted):

```
home_goals_for_per_match_season, home_goals_against_per_match_season,
away_goals_for_per_match_season, away_goals_against_per_match_season,
home_goals_for_per_match_last5, away_goals_for_per_match_last5,
home_points_last5, away_points_last5,
home_attack_strength_score, home_defence_strength_score,
away_attack_strength_score, away_defence_strength_score,
strength_diff, league_home_advantage_season,
competition_id
```

**xG, shots, shots-on-target, and every other Phase 2 feature beyond this 15-column set are explicitly NOT in the V1 model.** This was not an oversight corrected mid-project — a real bug did occur (Step 3 diagnostic, see §7 below) where an early implementation of `run_experiments.py` fed the full 264-column `feature_rows` table (including xG) into both candidate models instead of this approved subset, and the fix was to restrict to exactly these 15 columns, not to work around the crash any other way (e.g. blanket all-null-column dropping, zero-filling xG). The diagnostic confirmed none of these 15 approved columns are ever all-null or constant in any fold, in contrast to 22 xG-related columns that are structurally all-null in folds 1–2 (pre-xG-cutover).

## 5. Missing-data handling

No blanket `dropna()` anywhere. Per-model, fold-scoped strategy:
- **LogisticRegression**: `SimpleImputer(strategy="median")` + `StandardScaler`, both fit exclusively on the training fold's own data (`LogisticRegressionPreprocessor` in `train.py`). `competition_id` is one-hot encoded against categories seen in that training fold only.
- **HistGradientBoostingClassifier**: uses scikit-learn's native missing-value handling — no manual imputation. `competition_id`'s real values (`{200, 419, 423, 477, 499}`, not contiguous) are re-coded to a dense `[0, 5)` integer range via `CompetitionCodeEncoder`, fit on the training fold only; an unseen category at transform time maps to `-1`, which HGB's native categorical handling treats as missing (never silently collides with a real category).

## 6. Models evaluated

Exactly two real candidates plus two non-selectable baselines, per the approved scope (no Random Forest, XGBoost, LightGBM, neural networks, Poisson, or Monte Carlo):

| Model | Mean validation log loss (folds 1–3) | Mean validation Brier | Mean validation accuracy |
|---|---|---|---|
| Majority-class baseline | 19.4792 | — | — |
| Frequency baseline | 1.0737 | 0.6498 | — |
| **LogisticRegression** | **1.0043186935714623** | **0.6003193827036707** | **0.5087728341743457** |
| HistGradientBoostingClassifier | 1.0891458453243126 | 0.6446304423510489 | 0.4756097027589706 |

Both real candidates beat the majority-class baseline; only LogisticRegression clearly beat the frequency baseline (HGB did not: `beats_frequency_baseline: {"logistic_regression": true, "hist_gradient_boosting": false}`, recorded in `data/audit/phase3_model_comparison.json`).

**Selection rule (fixed before any candidate was compared): lowest mean validation log loss across folds 1–3, Brier score as tie-breaker when candidates are within 0.005 log loss of each other.** No tie-break was needed (`tie_break_applied: false`) — LogisticRegression's mean log loss was clearly lower.

**Selected: LogisticRegression. HistGradientBoosting was evaluated and is documented, but not selected.**

## 7. Implementation issues found and fixed (Step 3)

A real-machine run surfaced two independent, confirmed bugs, diagnosed before any fix was applied (`docs/PHASE3_STEP3_DIAGNOSTIC_REPORT.md`) and fixed only after explicit review and approval:

1. **Feature-scope bug**: `run_experiments.py` was passing the full 264-column `feature_rows` matrix into both candidate models instead of the approved 15-column subset, letting 22 structurally all-null xG columns (in folds 1–2, pre-cutover) reach `HistGradientBoostingClassifier`'s binning step and crash it (`ValueError: window shape cannot be larger than input array shape`). Fixed by adding `_select_approved_features()`, which restricts `X` to exactly `config.APPROVED_FEATURE_COLUMNS_V1` before either model is touched, and fails loudly (never substitutes) if an approved column is ever missing.
2. **`competition_id` encoding bug**: the real IDs (`{200, 419, 423, 477, 499}`) were passed directly to `HistGradientBoostingClassifier`'s native categorical handling, which requires a dense `[0, n_categories)` integer range — fixed via `CompetitionCodeEncoder` (§5).
3. A third, unrelated bug — `logging_setup.py`'s `setup_logging()` leaking an open `FileHandler` file descriptor across repeated calls, causing `WinError 32` on Windows tempdir cleanup — was diagnosed and fixed separately (`setup_logging()` now closes every existing handler before clearing them), kept deliberately isolated from the ML fix.

No blanket all-null-column dropping, no zero-filling of xG, and no change to `features.db`, `matches.db`, Phase 1/2 artifacts, or split boundaries were made in the course of these fixes.

## 8. Probability diagnostics and the Draw-probability finding (Step 4)

Computed on the pooled folds 1–3 validation predictions (n=5,331), from the frozen LogisticRegression configuration:
- Mean predicted probability: H 0.4233, D 0.2538, A 0.3229 — vs. actual frequency H 0.4363, D 0.2519, A 0.3118.
- Pooled log loss 1.0044, Brier 0.6004 (consistent with the fold-average numbers above).
- **ECE (top-label/confidence, Guo et al. 2017 definition, 10 equal-width bins): 0.0051** — low, indicating the model's stated confidence tracks its actual accuracy closely.

**Draw-probability investigation**: mean predicted P(draw) (0.2538) is within 0.0019 of the empirical draw frequency (0.2519) — essentially exact on aggregate. Draw wins the argmax (i.e. is the model's top prediction) only 0.8% of the time, is ranked 2nd of 3 outcomes 59.0% of the time, and 3rd 40.1% of the time. **Verdict: (A) — a structural argmax phenomenon, not (B) systematic underestimation.** The reliability bin containing 73% of all draw predictions ([0.2, 0.3) predicted probability) has an observed frequency of 0.253 against a mean predicted probability of 0.253 — a calibration gap of exactly 0.000. This finding did not alter the model.

## 9. Calibration (Step 4)

Three methods compared via a nested, expanding-window scheme (fit on strictly-earlier validation folds, scored on a later one — never in-sample, never using 2025/26):

| Method | Mean log loss (nested assessment) | Mean Brier |
|---|---|---|
| Uncalibrated | **1.0006233634207131** | **0.5978281980608899** |
| Platt (sigmoid), pure-NumPy implementation | 1.0053671615638624 | 0.601090766750884 |
| Isotonic (PAVA), pure-NumPy implementation | 1.0503368331234915 | 0.6018488170528399 |

Both calibrated methods performed worse than uncalibrated, consistent with the already-low ECE found in §8. **Decision: raw, uncalibrated LogisticRegression probabilities are used for V1. No Platt or isotonic calibration is applied.** The selection rule (a candidate must improve mean log loss by more than 0.005 without worsening mean Brier by more than the same margin) was fixed before any candidate was compared, and neither candidate cleared it.

## 10. Final, locked 2025/26 test results

Reported once, after every prior decision (feature set, model, calibration method) was already frozen using folds 1–3 only:

- **Log loss: `1.012643607907126`**
- **Brier score: `0.6059500599589396`**
- **Accuracy: `0.500856653340948`**
- Macro-F1: 0.3712753915810083, balanced accuracy: 0.4321797560315324, ECE: 0.01731559581035984
- Confusion matrix (rows = true H/D/A, columns = predicted H/D/A): `[[599, 2, 170], [293, 0, 152], [257, 0, 278]]`

Since the selected calibration method is "uncalibrated," pre- and post-calibration final-test results are identical; `post_calibration` is recorded as `null` in `data/audit/phase3_calibration_comparison.json` explicitly, not omitted.

## 11. Known limitations

- Isotonic and Platt calibration were evaluated with only 2 nested assessment rounds (fold_2, fold_3) — a small sample for concluding "calibration doesn't help," though the result is consistent with the already-low uncalibrated ECE.
- Reliability bins with very low sample counts (e.g. the Home 0.8–0.9 bin, n=45; Draw's tail bins, n≤2) produce noisy gap estimates and should not be over-interpreted.
- xG's missingness is structurally coverage-gated (near-zero pre-2023/24, ~21% in the 2023/24 transition season, ~94.5% in 2024/25–2025/26) — this is why xG is entirely absent from the approved V1 feature set, not merely underweighted.
- Only one season (2025/26) exists as a genuinely untouched final test set — a single realization of "the future," not a large sample; the final-test numbers above should be read with that in mind, not treated as a precise long-run performance guarantee.
- No second-division/promotion data exists (Phase 2 finding, unchanged) — an inherent ceiling on early-season accuracy for newly-promoted teams that V1 cannot address.
- Per-league residual calibration was not separately checked; the global pooled model with `competition_id` as a feature was chosen for sample-size reasons (§4) but has not been audited for whether it under- or over-performs for any single league specifically.

## 12. Explicit Phase 4 exclusions

Not started, and out of scope for everything documented above: xG/shots/full-feature ablation (Model A/B/C/D comparison), hyperparameter tuning, Poisson/Monte Carlo match simulation, betting logic or recommendations, a production prediction API, and any UI. These require an explicit, separate go-ahead before work begins.

## 13. Source artifacts

- `data/audit/phase3_model_comparison.json` — Step 3 fold-by-fold and final-test metrics for all 4 models.
- `data/audit/phase3_step3_diagnostic_column_report.json` — the null/constant column diagnostic behind §7.
- `data/audit/phase3_probability_diagnostics.json`, `data/audit/phase3_calibration_comparison.json` — Step 4 diagnostics and calibration comparison.
- `data/audit/phase3_predictions/` — per-fold and final-test prediction CSVs (H/D/A probabilities per fixture, for every model).
- `docs/PHASE3_STEP3_MODEL_COMPARISON_REPORT.md`, `docs/PHASE3_STEP3_DIAGNOSTIC_REPORT.md`, `docs/PHASE3_STEP4_PROBABILITY_CALIBRATION_REPORT.md` — the step-by-step narrative reports this document consolidates.
- `src/models/` — `config.py`, `data.py`, `splits.py`, `baselines.py`, `train.py`, `evaluate.py`, `run_experiments.py`, `calibration.py`.
- `tests/test_model_*.py` — full test coverage; see the finalization report for the current pass/skip/fail count.
