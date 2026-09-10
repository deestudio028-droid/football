# Phase 3 Step 3 — Diagnostic Report (no implementation changes)

Per your instruction: **this is diagnostic-only.** No production code, no `data/processed/*`, no split boundaries, no xG values, and no feature definitions were modified. The only new file written is a read-only audit artifact: `data/audit/phase3_step3_diagnostic_column_report.json` (full per-column, per-fold stats — this is an audit artifact next to the existing Phase 2 audit files, not a production dataset).

## Headline finding, stated plainly

**The crash is not primarily a data problem — it's an implementation gap I introduced in Step 2/3.** `run_experiments.py` passes `train_ds.X` straight into `train_logistic_regression()` / `train_hist_gradient_boosting()`. `train_ds.X` is the **full 264-column feature matrix** from `load_supervised_dataset()`, not the 15-column subset (14 features + `competition_id`) that the Step 2 spec explicitly approved and that I verified existed by name back in Step 2. I verified the columns *existed*; I never actually wrote the code that *restricts* `X` to them before fitting. That gap is what let 54 xG-related columns — including ones that are structurally all-null in early folds — reach both models.

A second, independent bug also exists in `competition_id` handling for HGB (detail in finding 3 below), which would need fixing regardless of the feature-set question.

## 1. Per-fold, per-column diagnostics

Full machine-readable version: `data/audit/phase3_step3_diagnostic_column_report.json` (264 columns × 3 folds, every field you asked for: column, dtype, train non-null count, train null count, train unique-non-null count, all-null flag, constant flag).

**LR input vs HGB input:** currently these are the *same* 264-column `X` for both models (this is itself part of the bug — see headline finding). So the diagnostics below are one table, not two; I've noted where LR's and HGB's *preprocessing* handles them differently.

Summary by fold (training partition only):

| Fold | n_train | Total cols | All-null cols | All xG-related? | Constant cols |
|---|---|---|---|---|---|
| fold_1 (train 2020/21–2021/22) | 3,652 | 264 | **22** | Yes, all 22 | 17 (16 are xG `_coverage_n`=0, 1 is `shrinkage_k`=5) |
| fold_2 (train 2020/21–2022/23) | 5,479 | 264 | **22** | Yes, all 22 | same 17 |
| fold_3 (train 2020/21–2023/24) | 7,231 | 264 | **0** | — | 1 (`shrinkage_k`=5) |

The 22 all-null columns in folds 1 and 2 (identical set both times) are exclusively xG rate/diff features (`home_xg_for_per_match_season`, `away_xg_diff_last5`, `home_xg_diff_form_last10`, etc. — full list in the JSON and reproduced below). This lines up exactly with what `PHASE2_FEATURE_ENGINEERING_REPORT.md` already documented: xG coverage has a season-based cutover, and the walk-forward fold-1/fold-2 training windows (2020/21–2022/23) predate it for a meaningful share of matches — enough that, given the historical form/rolling windows involved, the training partition has literally zero observed xG values in those folds. By fold 3 (train window extends through 2023/24) there's enough post-cutover history that no xG column is all-null anymore.

All 22 all-null columns, fold 1 and fold 2 (identical set):
```
home_xg_for_per_match_last5, home_xg_against_per_match_last5, home_xg_diff_last5,
home_xg_for_per_match_last10, home_xg_against_per_match_last10, home_xg_diff_last10,
home_xg_for_per_match_season, home_xg_against_per_match_season, home_xg_diff_season,
away_xg_for_per_match_last5, away_xg_against_per_match_last5, away_xg_diff_last5,
away_xg_for_per_match_last10, away_xg_against_per_match_last10, away_xg_diff_last10,
away_xg_for_per_match_season, away_xg_against_per_match_season, away_xg_diff_season,
home_xg_diff_form_last5, home_xg_diff_form_last10, away_xg_diff_form_last5, away_xg_diff_form_last10
```
(The corresponding `*_coverage_n` companion columns aren't "all-null" — they're **constant at 0**, which is consistent: they're counting how many of the last N matches had usable xG, and if none did, the count is 0, not missing.)

**Critically: none of the 15 approved Step 2 features (`home_goals_for_per_match_season`, …, `strength_diff`, `league_home_advantage_season`, `competition_id`) are all-null or constant in any fold.** Every one of them has thousands of non-null, non-constant observations in every fold — I checked this explicitly, column by column, and it's in the JSON too. This is the direct evidence that restricting to the approved subset would have avoided the crash entirely.

## 2. Answers to your design questions, grounded in the diagnostics above

**(1) Should all-null features be removed per training fold before fitting?**
Not as a blanket, automatic rule — and I want to flag that doing so unconditionally is close to the "silently drop xG columns globally" behavior you told me not to do. The diagnostics show the actual fix for *this* step doesn't require any per-fold dropping logic at all: the approved 15-feature subset was never all-null in any fold, so simply scoping `X` to that subset (which is what should have happened in Step 2/3 already) resolves the crash with zero new logic. Per-fold automatic null-column-dropping only becomes a real question once a model variant *intentionally* includes xG (i.e. Model C/D in the ablation design) — and that's a future step, not this one.

**(2) If removed per fold, how would validation/test transforms guarantee the same schema the training fold learned?**
This only needs solving once Model C/D is in scope. The design pattern to use is the one `LogisticRegressionPreprocessor` already follows for `competition_id` categories: fit-time state (e.g. a stored `selected_columns_` list, computed once from `X_train` and frozen) must be saved on the fitted preprocessing object, and `.transform()` must always subset any incoming frame (validation or the final test set) to exactly that stored list — never recompute "which columns are non-null" independently against validation/test data. That's a straightforward extension of code that already exists, but I have not written it, since Model C/D isn't approved yet.

**(3) Is `competition_id` currently represented correctly for HGB and LogisticRegression?**
- **LogisticRegression: yes.** `LogisticRegressionPreprocessor` one-hot encodes `competition_id` against categories observed in the training fold only — correct and already tested.
- **HistGradientBoosting: no, this is a real bug independent of the xG issue.** The real `competition_id` values in the dataset are `{200, 419, 423, 477, 499}` (I queried them directly — not sequential, not small integers). `train.py` currently passes `categorical_features=["competition_id"]` straight to `HistGradientBoostingClassifier` with those raw values. scikit-learn's documented contract for `categorical_features` is that categorical columns must be encoded as integers in `[0, n_categories)` — raw arbitrary-magnitude IDs like `200`–`499` violate that contract. I can't execute sklearn in this sandbox to confirm the exact traceback, but this is a documented, independent way to get bin/array-shape errors out of `HistGradientBoostingClassifier`'s binning code, separate from (and in addition to) the all-null-column crash. Both bugs need to be fixed, not just one.

**(4) Does HGB actually need native categorical handling for `competition_id`, or should the strategy change?**
With only 5 distinct leagues, two reasonable options exist, and I'd rather you pick than have me silently choose:
- (a) Keep native categorical support, but fix the encoding: fit a train-only ordinal mapping (the 5 real IDs → `0..4`) before passing to HGB, mirroring the leakage-safe pattern already used elsewhere.
- (b) Drop `categorical_features=` entirely and pass `competition_id` as a plain numeric column. Trees split on thresholds, not enforced ordering, so a tree can still separate leagues effectively on a raw integer id even without semantic ordering — this sidesteps the encoding-range requirement completely, at the cost of being slightly less "native"/explicit about what the column means.

I have not implemented either.

**(5) Does the existing Model A/B/C/D ablation design already solve this more cleanly?**
Yes. This is the cleanest way to see it: Model A/B (no xG) is, by definition, the 15-feature approved subset I verified in Step 2 — and the diagnostics above confirm that subset is never all-null in any fold. The Step 3 experiment was *supposed* to already be scoped to that subset; the crash exists only because the code doesn't actually enforce that scoping. So the ablation design doesn't need to change to fix this — the implementation needs to catch up to a decision that was already made and verified two steps ago. Model C/D (with xG) is where questions 1–2's per-fold schema logic will actually be needed, and that's correctly still out of scope.

## 3. `test_config_and_logging.py` `WinError 32` — diagnosed separately, root cause found

This is unrelated to the ML pipeline. Root cause, from reading `src/ingestion/logging_setup.py`:

```python
def setup_logging(log_dir: Path, name: str = "ingestion", level: int = logging.INFO) -> logging.Logger:
    ...
    logger = logging.getLogger(name)
    ...
    logger.handlers.clear()          # <-- drops handler references, but never calls handler.close()
    ...
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    ...
    logger.addHandler(file_handler)
```

`logging.getLogger(name)` returns a **process-wide singleton** keyed by `name`. `test_logger_file_handler_never_writes_token` calls `setup_logging(tmp_dir / "logs", name="test_logger")`. `logger.handlers.clear()` detaches the *previous* run's `FileHandler` from the logger's handler list, but never calls `.close()` on it — so the underlying OS file descriptor to the old log file stays open. When `tearDown()` calls `self._tmp.cleanup()`, Windows refuses to delete a file that still has an open handle (`WinError 32: The process cannot access the file because it is being used by another process`); POSIX (this sandbox) allows unlinking open files, which is exactly why this never surfaced here — I can see the leaked handler behavior indirectly (a stray `test_logger` log line appears mid-run in this session's own test output, confirming the singleton and its handler persist across runs), but the sandbox's Linux filesystem tolerates it silently.

Fix (not yet applied, flagging for your review since you asked to keep this separate from the ML fix): `setup_logging()` should call `handler.close()` on every handler in `logger.handlers` before `.clear()`, e.g.:
```python
for h in logger.handlers:
    h.close()
logger.handlers.clear()
```
This is a small, self-contained, low-risk fix confined to `logging_setup.py` — it does not touch ingestion data, the model pipeline, or any Phase 1/2/3 artifact.

## What I have NOT done

- No change to `train.py`, `run_experiments.py`, `data.py`, `splits.py`, `config.py`, or `logging_setup.py`.
- No feature-set restriction implemented yet, no per-fold column-dropping logic implemented, no `competition_id` re-encoding implemented, no logging fix applied.
- No new experiment run.
- Only read the real `features.db` (never wrote to it) and ran targeted, read-only Python to produce the diagnostics above, plus the existing test suite (200 pass / 23 skip, unchanged from before — this run was to confirm nothing already broke, not a new experiment).

Waiting on your review of the 5 design-question answers (particularly 3/4, where there's a real choice to make) before I touch any implementation.
