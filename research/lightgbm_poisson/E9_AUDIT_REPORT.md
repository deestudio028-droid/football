# E9 — Phase 1 Audit Report

**Date:** 2026-08-20
**Status:** Audit only. **No implementation code written.**
**Per §4 and §33:** the audit is reported before any module is built.

---

## BLOCKING FINDING — §4 STOP CONDITION TRIGGERED

> **LightGBM is not installed and cannot be installed in this environment.**

```
lightgbm : MISSING
sklearn  : MISSING
scipy    : MISSING
numpy    : 2.2.6
pandas   : 2.3.3
```

```
$ pip install lightgbm --break-system-packages
ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))
ERROR: Could not find a version that satisfies the requirement lightgbm (from versions: none)
ERROR: No matching distribution found for lightgbm
```

§4 states: *"If LightGBM is unavailable: STOP and report. Do NOT silently replace LightGBM with another library."*

**I am stopping and reporting.** No substitute (XGBoost, sklearn's `HistGradientBoostingRegressor`, or a hand-rolled booster) has been used or will be used. Note that `sklearn` is also missing, so **Arm A cannot run either** — E9 requires both arms locally.

`requirements.txt` currently contains only `requests>=2.31` and `pytest>=7.4`. **LightGBM is not a declared project dependency**, so this is not a broken environment — the library has never been part of this project.

The only pre-existing mention of "lightgbm" anywhere in the repository is in `research/temperature_scaling/test_temperature_scaling.py`, as a *banned import* in E8's scope-discipline test. There is **no existing LightGBM implementation to reuse or isolate** (audit point 15).

---

## What I did complete

Audit points that do not require the library are answered below, so the local run is turnkey. Points 3, 12, 13, 14 depend on the library and are marked accordingly.

---

## 1. Exact V3 87-feature ordering

Confirmed unchanged from E5/E6/E7:

| Positions | Group |
|---|---|
| 1–18 | Goals for/against/diff — last5, last10, season, home & away |
| 19–38 | Form — points, W/D/L rates, goal diff, last5/last10 |
| 39–43 | Attack/defence strength scores + `strength_diff` |
| 44 | `competition_id` |
| 45–62 | Shots for/against/diff |
| 63–80 | Shots-on-target for/against/diff |
| 81–84 | Venue splits |
| 85–87 | Causal Elo (`home_elo`, `away_elo`, `elo_diff`) |

Source of truth: `src/models/v3_contract.py::V3_FEATURE_COLUMNS`.

## 2. Exact V3 preprocessing

`src/models/train.py::LogisticRegressionPreprocessor`:

1. `SimpleImputer(strategy="median")` on the 86 numeric columns
2. `StandardScaler()` on the imputed result
3. One-hot of `competition_id` against categories seen at fit time
4. `np.hstack([scaled, onehot])`

**87 contract columns → 91 encoded columns** (86 numeric + 5 one-hot). Statistics fitted on training rows only.

## 3. Does LightGBM need scaling?

**No.** Gradient-boosted trees split on thresholds and are invariant to monotone rescaling. Standardisation is neither required nor harmful.

**But this creates a real design question**, and it is the sharpest one in this audit — see "Design conflict" below.

## 4. Categorical `competition_id`

V3 one-hot encodes it. Distinct values: `[200, 419, 423, 477, 499]` — five leagues, no ordinal meaning.

**Recommended for Arm B:** pass `competition_id` as a LightGBM **native categorical** feature (`categorical_feature=['competition_id']`) after mapping to a dense `pandas.Categorical`.

Justification per §6: LightGBM's categorical handling partitions category *sets* at each split, so it never imposes the false ordering that raw integers would in a linear model. This preserves exactly the information one-hot carries (league identity, no ordering) without changing the contract. **Column count stays 87.**

Rejected alternatives: raw integer without declaration (imposes false ordinal meaning), target encoding (leakage risk, forbidden), embeddings (out of scope).

## 5–7. Targets and objective

| Item | Value |
|---|---|
| Home target | `label_home_goals` (count) |
| Away target | `label_away_goals` (count) |
| V3 objective | `PoissonRegressor(alpha=1.0, max_iter=2000)` — Poisson deviance with L2 |
| Arm B objective | `LGBMRegressor(objective="poisson")` |

Two independent models per arm. No 1X2 classifier anywhere.

## 8–9. Score grid and probability conversion

`src/models/poisson.py`:

- `_grid_size(lam_max, tol=1e-15)` — **adaptive**, derived by exact forward CDF recursion until the Poisson upper tail falls below `TAIL_TOL = 1e-15`, capped at 20,000. **Not a hardcoded `max_goals`.**
- `hda_tail_safe` — factorised cumulative form for P(H)/P(D), P(A) by complement
- `predict_poisson` — safety gates: `MASS_TOL=1e-12`, `COMPLEMENT_TOL=1e-10`, `ROWSUM_TOL=1e-9`, plus finite and [0,1] checks

**K is computed from `max(lam_h.max(), lam_a.max())` across the batch.** Consequence for §20: if the two arms produce different maximum lambdas they will derive different K. The harness must compute **one shared K** from the pooled maximum across both arms and pass it to both, or the "same score grid" requirement is violated. This is a concrete implementation constraint, not a formality.

## 10. Chronological folds

`src/models/config.py::WALK_FORWARD_FOLDS`, confirmed unchanged:

| Fold | Train | Test |
|---|---|---|
| fold_1 | 2020/21, 2021/22 | 2022/23 |
| fold_2 | + 2022/23 | 2023/24 |
| fold_3 | + 2023/24 | 2024/25 |

Eligible universe:

| Season | n |
|---|---|
| 2020/21 | 1,826 |
| 2021/22 | 1,826 |
| 2022/23 | 1,827 |
| 2023/24 | 1,752 |
| 2024/25 | 1,752 |
| **Total** | **8,983** |
| **Pooled OOS** | **5,331** |

`FINAL_TEST_SEASONS = ('2025/2026',)` — quarantined, never touched.

## 11. Causal feature construction

Features come from `features.db` (built causally in Phase 2) plus causal Elo from `src/features/elo.py` — the two-pass-per-timestamp engine validated in E1 and re-verified in E5/E6.

## 12–14. LightGBM version, determinism

**Cannot be determined — library unavailable.**

For the local run, determinism requires: `random_state=42`, `n_jobs=1`, `deterministic=True`, `force_row_wise=True`. LightGBM is only bit-reproducible when thread count is fixed; multi-threaded histogram construction can reorder floating-point accumulation. **The harness must set `n_jobs=1` and the report must state the version**, since results are not comparable across LightGBM major versions.

## 15. Existing LightGBM code

**None.** Only the banned-import reference in E8's test. Nothing to reuse; nothing to isolate.

---

## Design conflict requiring a decision

Audit point 3 surfaces a genuine tension with §13's "identical preprocessing information".

**Measured missingness across the 87 features on all 10,734 rows:**

| Quantity | Value |
|---|---|
| Columns with at least one NaN | **83 of 87** |
| Total NaN cells | **30,788 (3.297%)** |
| Worst columns | 4 venue-season features, 1,164 NaN each |

V3 **median-imputes** these. LightGBM handles NaN natively by learning a default split direction, which is generally better — missingness here is informative (early-season fixtures have no season-to-date history yet).

The conflict: giving LightGBM native NaN handling means the two arms no longer receive identical inputs, so a win could come from better missing-data handling rather than from nonlinearity. Median-imputing for LightGBM keeps the comparison clean but handicaps it on 3.3% of cells.

**Recommendation:** use **V3's exact median-imputed matrix for both arms** in the primary experiment. This isolates the one variable E9 exists to test — linear versus gradient-boosted — per §2 and §13. Scaling is applied to both as well; it is a no-op for trees and keeps the inputs byte-identical.

A **secondary diagnostic** may report LightGBM with native NaN handling, clearly labelled as testing a *different* hypothesis (missing-data handling), and **not** feeding the E9 decision.

I want this confirmed rather than assumed, because it is the kind of choice that quietly decides an experiment.

---

## Proposed pre-registered hyperparameter set

Per §8, a compact list rather than the full Cartesian product. **16 configurations**, all within the §8 grid, centred on §9's default (config 1):

| # | leaves | lr | trees | min_child | subsample | colsample | reg_λ |
|---|---|---|---|---|---|---|---|
| 1 | 15 | 0.05 | 250 | 50 | 0.8 | 0.8 | 1.0 |
| 2 | 7 | 0.05 | 250 | 50 | 0.8 | 0.8 | 1.0 |
| 3 | 31 | 0.05 | 250 | 50 | 0.8 | 0.8 | 1.0 |
| 4 | 15 | 0.02 | 250 | 50 | 0.8 | 0.8 | 1.0 |
| 5 | 15 | 0.05 | 100 | 50 | 0.8 | 0.8 | 1.0 |
| 6 | 15 | 0.05 | 500 | 50 | 0.8 | 0.8 | 1.0 |
| 7 | 15 | 0.05 | 250 | 20 | 0.8 | 0.8 | 1.0 |
| 8 | 15 | 0.05 | 250 | 50 | 1.0 | 0.8 | 1.0 |
| 9 | 15 | 0.05 | 250 | 50 | 0.8 | 1.0 | 1.0 |
| 10 | 15 | 0.05 | 250 | 50 | 0.8 | 0.8 | 0.0 |
| 11 | 7 | 0.02 | 500 | 50 | 0.8 | 0.8 | 1.0 |
| 12 | 7 | 0.02 | 250 | 20 | 0.8 | 0.8 | 1.0 |
| 13 | 31 | 0.02 | 500 | 50 | 0.8 | 0.8 | 1.0 |
| 14 | 7 | 0.05 | 100 | 50 | 1.0 | 1.0 | 1.0 |
| 15 | 31 | 0.05 | 100 | 20 | 0.8 | 0.8 | 0.0 |
| 16 | 7 | 0.02 | 250 | 50 | 0.8 | 0.8 | 1.0 |

Rationale: configs 2–10 vary one axis at a time from the default (clean sensitivity); 11–16 sample the low-capacity corner, which is where a 3.6k-row training set most likely wants to live. Fold 1 trains on 3,652 rows — 31 leaves with 500 trees is already high-capacity for that.

**No early stopping** (§11 prefers fixed `n_estimators`), so no validation set is consumed and nothing can leak.

---

## Benchmarks — recorded before any result exists

| Benchmark | Log Loss | Universe |
|---|---|---|
| **Honest V3** (per-fold refit) | **0.990435** | E7 universe, n=3,479 |
| **Market alone** | **0.955905** | market subset, n=3,479 |
| ~~E2's V3~~ | ~~0.981357~~ | **DO NOT USE — in-sample contaminated** |

E2 scored the shipped artifact on seasons it was trained on; discovered in E7, root-caused in E8.

**A caution on scope:** both figures above are on the **3,479-fixture market subset**. E9's primary evaluation is the **5,331-fixture** full OOS universe. Arm A's log loss on 5,331 will not equal 0.990435, and that is not a discrepancy — it is a different universe. Per §15 the harness must report both separately with exact n, and must never compare across them. The market comparison is only valid on the overlap.

For scale, from E6's real run on the 5,331 universe: V3 = 0.987025. That is the number Arm A should land near.

---

## Protected file integrity (PRE)

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## Audit summary

| # | Point | Status |
|---|---|---|
| 1 | V3 87-feature ordering | **Resolved** |
| 2 | V3 preprocessing | **Resolved** — median-impute + scale + one-hot |
| 3 | LightGBM scaling need | **Resolved** — not needed; raises a design conflict |
| 4 | `competition_id` handling | **Resolved** — native categorical recommended |
| 5–7 | Targets and objective | **Resolved** |
| 8–9 | Score grid and conversion | **Resolved** — adaptive K; shared-K constraint identified |
| 10 | Folds | **Resolved** — 3 folds, 8,983 / 5,331 |
| 11 | Causal construction | **Resolved** |
| 12–14 | Version, determinism | **BLOCKED** — library unavailable |
| 15 | Existing LightGBM code | **Resolved** — none exists |

---

## Two decisions I need from you

1. **Missing-data handling** — median-impute both arms (recommended: isolates nonlinearity, the variable E9 tests), or native NaN for LightGBM (better model, confounded comparison)?

2. **Given the §4 STOP, how should I proceed?**
   - **(a)** Build the module, tests and harness anyway so E9 is one local command from an answer — matching E3/E4/E5/E7. Nothing runs here; the decision stays open.
   - **(b)** Halt entirely until LightGBM is installed.
   - **(c)** Add `lightgbm` and `scikit-learn` to `requirements.txt` first — arguably correct regardless, since E9 makes them real dependencies. That is a production-file edit, and §31 requires me to stop and ask before touching one.

My recommendation is **(a) plus (c)**, with (c) done only on your say-so.

**No implementation code has been written. No production file has been modified.**
