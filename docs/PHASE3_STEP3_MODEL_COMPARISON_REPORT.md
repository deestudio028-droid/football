# Phase 3 — Step 3: Model Comparison (Experiment Runner)

Status: **runner built and tested; the actual experiment has NOT been executed yet.** This sandbox has no scikit-learn and no PyPI access (same restriction documented in every prior phase). Per your explicit instruction to STOP and report rather than work around a blocker, I did not fabricate or approximate results — everything below is what the code does and what was verified structurally; the real numbers require one run on your machine.

## 1. What was built

`src/models/run_experiments.py` — a single, reproducible entry point:

- `run_walk_forward_experiment(dataset)` — runs Frequency baseline, Majority-class baseline, LogisticRegression, and HistGradientBoosting across the 3 approved walk-forward folds only, using the exact fold boundaries in `config.WALK_FORWARD_FOLDS` (unchanged from Step 1/2). Records per-fold: log loss, Brier, macro-F1, balanced accuracy, accuracy, confusion matrix, `n_train`, `n_val`, and per-column missing-value counts for train and validation separately.
- `summarize_across_folds(fold_results)` — mean/std per model per metric across whichever folds it's given.
- `select_model(summary)` — the model-selection decision. Its function signature only accepts the fold summary dict (verified by a dedicated test that inspects the signature) — there is structurally no argument through which 2025/26 data could reach it. Rule: lowest mean validation log loss between LogisticRegression and HistGradientBoosting; Brier score breaks ties within 0.005 log loss. Baselines are reported but never selectable.
- `run_final_test_evaluation(dataset)` — runs all 4 models once against the untouched 2025/26 test split. `main()` calls this strictly after `select_model()` has already run and been written into the output dict — verified by a source-order test, not just a docstring promise.
- `main()` — orchestrates the above, writes everything to `data/audit/phase3_model_comparison.json`, and (if a predictions directory is given) writes per-fold and final-test CSV prediction files for possible later calibration work.

## 2. Tests

`tests/test_model_experiments.py` — 23 tests total. Split by what they need:

**Run without scikit-learn (11 tests, all passing now):** missingness reporting, `_evaluate_and_record` rejecting invalid probabilities, `summarize_across_folds` hand-computed mean/std, `select_model`'s signature guarantee and tie-break logic, fold-definition chronology checks (no future season in any fold's training seasons, 2025/26 absent from every fold), and a source-order check proving `main()` computes `select_model()` before it ever calls `run_final_test_evaluation()`.

**Require scikit-learn + real `features.db` (12 tests, currently skipped with an explicit reason each):** all 4 models evaluated per fold, deterministic results across repeated runs, no label columns in X, no future season in training data at runtime (not just in the static fold config), and full end-to-end `main()` execution producing the JSON output with the expected keys.

Combined with the existing suite: **200 tests total, 200 passing, 0 failing, 23 skipped** (all 23 skips require scikit-learn; every skip names the reason). No existing test was weakened.

## 3–7. Baseline / LogisticRegression / HistGradientBoosting results, mean/std, model-selection decision

**Not available.** Producing these requires actually running `main()`, which requires scikit-learn. I confirmed the target dataset has exactly **10,734 labeled rows** in `data/processed/features.db` (`SELECT COUNT(*) FROM feature_rows WHERE label_result IS NOT NULL`), matching the number you referenced — so the runner is pointed at the right data, but no model has been fit against it yet in this environment.

## 8. Untouched 2025/26 test results

**Not available**, and by design cannot become available before section 3–7 exists — `run_final_test_evaluation()` is never called until model selection is already frozen.

## Exact next step

Run this on your machine (PowerShell, from the project root, with scikit-learn 1.9.0 installed):

```powershell
cd E:\Football Prediction Project
$env:PYTHONPATH = "$PWD\src"

# 1. Confirm the new code and tests are correct in your real environment first:
python -m unittest discover -s tests -v

# 2. Run the actual experiment on all 10,734 labeled rows:
python -m models.run_experiments
```

This will:
- Write full results to `data/audit/phase3_model_comparison.json` (sklearn version, per-fold metrics for all 4 models, cross-fold mean/std, the frozen model-selection decision, and — only after that decision is recorded — the 2025/26 final-test results for all 4 models).
- Write per-fold and final-test prediction CSVs to `data/audit/phase3_predictions/` (fixture_id, true label, and every model's H/D/A probabilities) — saved for possible calibration work later, **not used for calibration in this step**.

Once you've run it, send me the resulting `phase3_model_comparison.json` (or just paste the `python -m unittest discover -s tests -v` output and the printed "Selected model: ..." line) and I'll write up the actual comparison numbers, sanity-check them for anything suspicious (implausibly high accuracy, unstable probabilities, a model matching or beating the frequency baseline by an implausible margin, etc. — flagged per your standing instruction to stop and report rather than silently proceed), and only then move to whatever you approve as the next step (ablation, calibration, or the final 2025/26 report-out).

## Explicitly not done in this step (per your instructions)

No ablation, no calibration, no hyperparameter tuning, no Poisson/Monte Carlo simulation, no betting recommendations, no future-match predictions. `run_final_test_evaluation()` exists and is wired up correctly, but has never actually executed against real data in this environment — its 2025/26 output described in the spec is entirely prospective pending your run.
