# Phase 4A — Feature Ablation Study: Design/Diagnostic Report

**Status: diagnostic complete, implementation STOPPED at two open decisions before any training.** No ablation model has been trained. V1 (model, feature contract, split boundaries, artifacts, documentation) was not modified — confirmed unchanged (§7).

## 1–4. Real schema audit, column verification, group verification, per-fold coverage audit

Performed directly against `data/processed/features.db` (read-only), not assumed from prior docs. `feature_rows` has 274 total columns; 264 are legal model inputs (`models.data.get_feature_columns()`); of those, 139 are `_n`/`_coverage_n` sample-size/coverage-count companions, 1 (`shrinkage_k`) is a constant hyperparameter record (value `5` on every row — carries zero information for any model), and the remaining **124 are real per-match rate/score/flag columns**, which I organized into 14 schema-verified groups (full lists and verification tests in `src/models/ablation.py` / `tests/test_model_ablation.py::TestFeatureGroupColumnsExistInSchema`):

| Group | n columns | Design-report tier(s) mentioning it |
|---|---|---|
| `goals_core` | 18 | A ("goals + form") |
| `form` | 20 | A ("goals + form") |
| `shots_core` | 18 | B ("+shots") |
| `shots_on_core` | 18 | B ("+shots-on-target") |
| `xg_core` | 22 | C ("+xG") |
| `goals_venue` | 4 | D ("venue-split goals") |
| `tempo` | 6 | D ("tempo... proxies") |
| `discipline` | 4 | D ("discipline proxies") |
| `league_home_advantage` | 1 | D ("league home-advantage") |
| `strength` | 5 | **not mentioned anywhere in the design report's A–D prose** |
| `competition_id` | 1 | **not mentioned anywhere** |
| `league_mean_goals` | 1 | **not mentioned anywhere** |
| `venue_goal_diff` | 2 | **not mentioned anywhere** |
| `history_flags` | 4 | **not mentioned anywhere** |

All 124 + 139 (meta) + 1 (`shrinkage_k`) = 264, confirmed to account for every legal feature column with no gaps and no overlaps (`test_all_264_feature_columns_are_accounted_for`).

**Per-fold null-rate audit** (training partitions only), full numbers in `data/audit/phase4a_feature_group_diagnostic.json`:

| Group | fold_1 train max-null | fold_2 train max-null | fold_3 train max-null | final-test train max-null |
|---|---|---|---|---|
| goals_core / form / shots / shots_on / tempo / discipline / strength / league_context | 0.3%–10.8% | 0.3%–10.8% | 0.3%–10.8% | 0.3%–10.8% |
| **xg_core** | **100.0%** (all 22 columns) | **100.0%** (all 22 columns) | 97.8% (0 columns all-null) | 79.1% (0 columns all-null) |

Every non-xG group has ordinary, modest nulls (early-season "insufficient history" gaps, the same kind V1's approved columns already have) in every fold — no blocker. **xG is categorically different**: every one of its 22 columns has **zero observed values anywhere in fold_1's or fold_2's training partition** — this is the same structural, season-cutover-driven pattern documented in Phase 2 and re-confirmed in the Phase 3 Step 3 diagnostic, now re-verified directly for this study.

## 5. Missing-xG strategy decision — STOP

I did not pick a strategy. Here is the exact contradiction:

- **Forbidden**: zero-filling xG.
- **Forbidden**: silently dropping xG columns because of missingness.
- **Required**: Models C and D must include xG, evaluated across the same 3 walk-forward folds as every other model, using the existing infrastructure unmodified.
- **Fact**: fold_1 and fold_2's training partitions contain **no observed xG values at all** for any of the 22 xG columns. There is no data to impute from, and no data to compute a per-fold-safe fallback from, in either fold.

Any implementation must do one of:
(a) invent a placeholder value for those folds' xG columns — functionally a form of constant-filling, which is exactly what's forbidden, whether the constant is literally 0 or some other invented number;
(b) treat those columns as unusable for those two folds specifically — which is a fold-scoped, fully-documented decision, not a "silent" one, but it does mean Models C/D's xG signal is evaluated on only 1 of 3 folds (fold_3, 97.8% null — still very sparse) plus the untouched final test, undermining "mean log loss across folds 1–3" as an apples-to-apples comparison with Models A/B;
(c) restructure the ablation's fold set specifically for xG-inclusive models (e.g., compare Models C/D only using fold_3 and the coverage-rich final period) — this changes the evaluation protocol for a subset of models, which itself needs sign-off since the task says to use "the existing walk-forward infrastructure" unmodified for all models.

This is precisely the situation the task instructed me to stop on rather than guess. `HistGradientBoostingClassifier` additionally cannot even be *fit* on an all-NaN training column — it crashes (confirmed in Phase 3 Step 3, not theoretical) — so option (a) isn't just philosophically wrong, it's mechanically blocked for one of the two candidate models regardless.

**I need a decision on (a)/(b)/(c) — or another option I haven't considered — before writing any Model C/D training code.**

## Group-composition ambiguity — also flagged (affects all four models)

Independent of the xG question: 14 real schema columns (`strength`, `competition_id`, `league_mean_goals`, `venue_goal_diff`, `history_flags` — see table above) are never assigned to any tier in the design report's prose description of Models A–D. This isn't a missing-data problem, it's an under-specified definition. `src/models/ablation.py` contains `PROPOSED_MODEL_A/B/C/D_COLUMNS` — one concrete, fully-documented reading (competition_id and strength start at Model A since they aren't themselves an ablated dimension and strength is goals-derived; league_mean_goals/venue_goal_diff/history_flags land at Model D as part of "every V1-eligible feature") — but this is explicitly labeled a **proposal**, not a frozen decision, and nothing trains against it yet. Please confirm or correct this mapping.

## 6. What was and wasn't implemented

**Implemented and tested** (`src/models/ablation.py`, `tests/test_model_ablation.py`, 19 tests, all passing):
- All 14 schema-verified column groups, each confirmed to exist in the real schema.
- `select_feature_columns()` — a generic, leakage-safe restriction helper (same fail-loudly-never-substitute pattern as `run_experiments._select_approved_features()`), usable regardless of how the two open questions above are resolved.
- `train_model_a()` / `train_model_b()` / `train_model_c()` / `train_model_d()` — present as named entry points but each explicitly `raise NotImplementedError` with a message pointing at this report, so nothing can be accidentally invoked and silently produce a result built on an unconfirmed assumption.
- The full real per-fold coverage diagnostic, saved to `data/audit/phase4a_feature_group_diagnostic.json`.

**Not implemented**: any actual model training, any ablation experiment run, any comparison against V1's locked baseline. Nothing in this deliverable touches 2025/26 at all — it was never loaded or referenced anywhere in this stage's code.

## 7. V1 untouched — confirmed

`src/models/config.py` (`APPROVED_FEATURE_COLUMNS_V1`, `WALK_FORWARD_FOLDS`, `MODEL_VERSION`), `src/models/train.py`, `src/models/run_experiments.py`, `src/models/calibration.py`, every `docs/PHASE3*.md` file, and every `data/audit/phase3*` artifact were not opened for writing at any point in this session. `data/processed/features.db` and `data/processed/matches.db` MD5 checksums are unchanged (`e7ebe7fc...` / `fdeed042...`) from every previous checkpoint.

## Test suite

`python -m unittest discover -s tests -v`: **278 tests, 278 passed, 0 failed, 32 skipped** (unchanged skip set — all require scikit-learn, none newly introduced by this stage since `ablation.py` needs no ML library to define column groups or raise `NotImplementedError`). Up from 259 before this stage — 19 new tests in `tests/test_model_ablation.py`, 0 existing tests modified.

## Exact next step

Waiting on your decision on:
1. How to handle Models C/D's xG columns being 100%-null in fold_1/fold_2 training data (options (a)/(b)/(c) above, or your own).
2. Whether `PROPOSED_MODEL_A/B/C/D_COLUMNS` in `ablation.py` is the correct reading of the design report's intent, or needs correction.

Once both are confirmed, Models A and B can proceed to real training immediately (they have no xG dependency and no other open question) — that could start before the xG question for C/D is settled, if useful. Training itself will hit the same scikit-learn-unavailable-in-this-sandbox limitation documented in every prior Phase 3 step; running on your machine (`python -m unittest discover -s tests -v` then the eventual ablation runner) will be the same pattern as before.
