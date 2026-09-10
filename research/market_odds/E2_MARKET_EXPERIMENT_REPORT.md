# E2 — V3 vs V3 + Market: Controlled Research Experiment Report

**Date:** 2026-08-20
**Type:** Controlled research experiment (no promotion, no production change)
**Scope:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1

---

## EXECUTION STATUS — READ FIRST

| Experiment | Status |
|---|---|
| **A: V3 baseline** | **PENDING** — requires local execution |
| **B: Market-only** | **COMPLETE** — computed and reported below |
| **C: V3 + Market blend** | **PENDING** — depends on A |

**Reason:** V3 inference requires unpickling `v3_poisson_venue_elo_candidate.pkl`, which depends on scikit-learn. The sandbox environment has no scikit-learn and cannot install it (outbound PyPI blocked by proxy, HTTP 403). This is the same recurring limitation documented in the V3 Inference Integration Report (Suite 4).

Experiment B requires only numpy and the labels, so it ran here and its numbers below are **real measured results**, not projections.

The complete harness (`run_e2_experiment.py`) is written, self-checking, and ready. One local command produces A and C.

---

## 1. Dataset Definition

**Source:** `research/market_odds/research_dataset.sqlite`

Independently verified (not assumed from the prior phase):

| Property | Verified Value |
|---|---|
| Total fixtures | **4,001** |
| Leagues | 5 (all target leagues, no others) |
| Bookmaker | **Pinnacle only** (bookmaker_id=1), 4,001/4,001 |
| Seasons present | 2022/23, 2023/24, 2024/25 |
| 2025/26 fixtures | **0** — quarantine confirmed |
| De-vigged closing probs | 4,001 / 4,001 |
| De-vigged opening probs | 4,001 / 4,001 |
| Closing prob sum | min = 1.000000000000, max = 1.000000000000 |
| Overround range | 0.9714 – 1.1422 (mean 1.0297) |
| De-vig k range | 0.9638 – 1.1816 (mean 1.0326) |

**By league:**

| League | Fixtures |
|---|---|
| La Liga | 873 |
| Serie A | 866 |
| Premier League | 865 |
| Ligue 1 | 710 |
| Bundesliga | 687 |

**By league × season:**

| League | 2022/23 | 2023/24 | 2024/25 |
|---|---|---|---|
| Premier League | 110 | 375 | 380 |
| La Liga | 120 | 373 | 380 |
| Serie A | 111 | 375 | 380 |
| Bundesliga | 81 | 300 | 306 |
| Ligue 1 | 100 | 304 | 306 |

**Outcome distribution (n=4,001):** H = 1,736 (43.4%), D = 1,015 (25.4%), A = 1,250 (31.2%)

---

## 2. Fixture Matching

| Step | Count |
|---|---|
| Market-eligible fixtures | 4,001 |
| Matched to a V3 feature row in `features.db` | **4,001** |
| Matched to outcome labels | **4,001** |
| **Dropped** | **0** |

**Verified by direct join.** Every market fixture has both a V3 feature row and a label. Experiments A, B, and C therefore run on an **identical 4,001-fixture set** — no dataset mismatch is possible.

V3 *probabilities* for these fixtures are pending the local run, but the *matchability* is confirmed, so no fixtures will be lost at that step.

---

## 3. Market Coverage and Subset Bias

**This experiment is evaluated on the market-available subset, not the complete V3 historical dataset.** That subset is 44.5% of the eligible V3 data and is heavily skewed toward recent seasons.

| Season | Full V3 fixtures | Market fixtures | Coverage |
|---|---|---|---|
| 2020/2021 | 1,826 | 0 | **0.0%** |
| 2021/2022 | 1,826 | 0 | **0.0%** |
| 2022/2023 | 1,827 | 522 | **28.6%** |
| 2023/2024 | 1,752 | 1,727 | 98.6% |
| 2024/2025 | 1,752 | 1,752 | 100.0% |
| **Total** | **8,983** | **4,001** | **44.5%** |

### Consequences (stated plainly, not hidden):

1. **Two seasons are entirely absent.** No walk-forward fold can be built that tests on 2020/21 or 2021/22.
2. **2022/23 is a 28.6% non-random subset.** Which 522 of 1,827 fixtures OddAlerts retained is not under our control and is not verifiably random. Any fold trained on 2022/23 trains on a potentially biased sample.
3. **Only 2 walk-forward folds are constructible.** See Section 8.
4. **Results are not transferable to the full V3 dataset.** Any conclusion applies to the market-covered subset only.

---

## 4. De-Vig Methodology

**Method:** Power de-vig (primary, as specified). Simple proportional normalization was **not** used.

Solve for k:

```
(1/O_H)^k + (1/O_D)^k + (1/O_A)^k = 1
```

Then `P_i = (1/O_i)^k`.

**Implementation:** `research/market_odds/devig.py` — pure-Python bisection, tolerance 1e-12, max 200 iterations, automatic bracket expansion, final renormalization to eliminate residual floating-point error.

**Verification on the actual 4,001 fixtures:** closing probability sums have min = max = 1.000000000000 to 12 decimal places.

**Unit tests:** 14/14 pass (fair odds, vigged odds, invalid/None/NaN/Inf rejection, extreme odds, determinism, overround, symmetry, heavy favorite).

**Inputs used:** Pinnacle **closing** 1X2 (primary), Pinnacle **opening** 1X2 (robustness check only).
**Peak odds: NOT USED anywhere.**

---

## 5. V3 Baseline Methodology

**No retraining.** The frozen `v3_poisson_venue_elo_candidate.pkl` (MD5 `a2850a7687822a5916663301f5ccc96c`) is loaded and used as-is.

Inference path in `run_e2_experiment.py::load_v3_predictions`:

1. Load the 84 base features from `features.db` (read-only)
2. Compute causal Elo via `src/features/elo.py` — the two-pass-per-timestamp engine validated in Phase 1 (bit-identical to the research engine on all 10,735 fixtures)
3. Join to the exact 87-column `V3_FEATURE_COLUMNS` contract, with column-order assertion
4. `preprocessor.transform` → `model_home_goals.predict` / `model_away_goals.predict`
5. `predict_poisson(lam_h, lam_a, class_order)` — the same conversion V2 and V3 use in production

Elo features for a fixture depend only on matches completed strictly before that fixture's timestamp. No fixture's own result, and no future fixture, enters its features.

---

## 6. Market-Only Results (Experiment B) — MEASURED

Pooled over all 4,001 fixtures.

| Signal | Log Loss | Brier | RPS | Accuracy | Draw Recall | Draw Pred Rate | ECE |
|---|---|---|---|---|---|---|---|
| **Pinnacle closing** | **0.96045** | **0.57084** | **0.19218** | **0.5506** | 0.0049 | 0.0030 | **0.0052** |
| Pinnacle opening* | 0.97054 | 0.57746 | 0.19523 | 0.5334 | 0.0000 | 0.0005 | 0.0040 |

\* robustness check only, not the primary experiment

**Reference baselines on the same fixtures:**

| Baseline | Log Loss |
|---|---|
| Uniform (1/3, 1/3, 1/3) | 1.09861 |
| Class prior (H .434 / D .254 / A .312) | 1.07372 |
| **Market closing** | **0.96045** |

Closing beats opening on every metric except ECE — consistent with closing prices absorbing more information, as expected.

### By season (closing):

| Season | n | Log Loss | Brier | RPS | Accuracy |
|---|---|---|---|---|---|
| 2022/2023 | 522 | 0.99077 | 0.59111 | 0.20690 | 0.5421 |
| 2023/2024 | 1,727 | 0.95250 | 0.56577 | 0.18681 | 0.5576 |
| 2024/2025 | 1,752 | 0.95926 | 0.56980 | 0.19308 | 0.5462 |

### By league (closing):

| League | n | Log Loss | Brier | RPS | Accuracy |
|---|---|---|---|---|---|
| Premier League | 865 | 0.93312 | 0.55070 | 0.18886 | 0.5780 |
| La Liga | 873 | 0.95244 | 0.56501 | 0.18845 | 0.5498 |
| Serie A | 866 | 0.96420 | 0.57595 | 0.18611 | 0.5393 |
| Bundesliga | 687 | 0.97372 | 0.57928 | 0.19456 | 0.5444 |
| Ligue 1 | 710 | 0.98620 | 0.58815 | 0.20590 | 0.5380 |

### Notable finding — the market almost never predicts a draw

Draw recall is **0.0049** (5 of 1,015 actual draws), and the market makes D the argmax on only **0.30%** of fixtures. This is structural: a de-vigged draw probability rarely exceeds both H and A. Any blend that improves draw handling would have to come from V3, not from the market. This is worth tracking in Experiment C rather than treating accuracy as the headline metric.

---

## 7. V3 + Market Results (Experiment C)

**PENDING LOCAL EXECUTION.**

The blend is `P_final = w · P_market + (1−w) · P_V3`, renormalized.

Weight grid: `w ∈ {0.00, 0.05, 0.10, …, 1.00}` (21 values).

**Weight selection protocol (enforced in code):**

1. For each fold, `learn_weight()` receives **only training-period** `y`, `p_v3`, `p_market`
2. It scans the grid and selects the `w` minimizing **training** log loss
3. That `w` is frozen
4. It is applied to the test fold
5. **Test-fold outcomes are never passed to `learn_weight()`** — they are not in its signature

There is no global tuning over the complete dataset. The full weight curve per fold is recorded in `e2_results.json` for inspection.

---

## 8. Learned Blend Weights by Fold

**PENDING** — values populate after the local run.

**Fold structure (verified against actual data):**

| Fold | Train | n_train | Test | n_test | Train max unix | Test min unix | Strictly chronological |
|---|---|---|---|---|---|---|---|
| 1 | 2022/23 | 522 | 2023/24 | 1,727 | 1686509100 | 1691775000 | **True** |
| 2 | 2022/23 + 2023/24 | 2,249 | 2024/25 | 1,752 | 1717344000 | 1723741200 | **True** |

Both folds are strictly chronological — every training fixture kicks off before every test fixture. No shuffling, no random K-fold.

### Structural limitation — documented, not worked around

**Only 2 folds exist**, and **fold 1 trains on 522 fixtures that are a 28.6% non-random subset of 2022/23.**

Per the instruction to document rather than invent a workaround: no synthetic fold was created, no season was subdivided to manufacture extra folds, and no relaxation of the chronological rule was applied. Two folds with one thin, biased training period is weak evidence for a weight-selection procedure, regardless of which direction the metrics move.

---

## 9. Season-Wise Results

Market-only measured (Section 6). V3 and blend columns pending local run.

---

## 10. League-Wise Results

Market-only measured (Section 6). V3 and blend columns pending local run.

League-wise consistency is a decision input: a blend that improves pooled log loss but helps in only 2 of 5 leagues should not be read as a win.

---

## 11. Pooled Results

Market-only measured (Section 6). Full A/B/C comparison pending local run.

### A comparison that must NOT be made

It is tempting to compare market closing (LL 0.96045) against the V3 candidate's pooled walk-forward log loss of **0.986913** from Phase 1 and conclude the market is stronger. **That comparison is invalid:**

- Different fixture sets — Phase 1 used the full historical dataset; this uses a 44.5% recent-skewed subset
- Different seasons — Phase 1 included 2020/21 and 2021/22, which have zero market coverage
- Different protocol — Phase 1 used 3 folds; this has 2

V3's log loss **on these 4,001 fixtures** is what Experiment A produces, and only that number is comparable to 0.96045.

---

## 12. Calibration Results

**Market closing reliability (pooled over all three classes, 10 bins, n=10,003 class-fixture pairs):**

| Bin | n | Mean predicted | Observed freq | Gap |
|---|---|---|---|---|
| [0.0, 0.1) | 494 | 0.0712 | 0.0547 | −0.0166 |
| [0.1, 0.2) | 1,950 | 0.1569 | 0.1559 | −0.0010 |
| [0.2, 0.3) | 4,225 | 0.2547 | 0.2540 | −0.0008 |
| [0.3, 0.4) | 1,961 | 0.3399 | 0.3325 | −0.0074 |
| [0.4, 0.5) | 1,222 | 0.4461 | 0.4566 | +0.0105 |
| [0.5, 0.6) | 968 | 0.5488 | 0.5641 | +0.0153 |
| [0.6, 0.7) | 608 | 0.6464 | 0.6414 | −0.0050 |
| [0.7, 0.8) | 385 | 0.7447 | 0.7532 | +0.0085 |
| [0.8, 0.9) | 171 | 0.8394 | 0.8421 | +0.0027 |
| [0.9, 1.0) | 19 | 0.9156 | 0.8947 | −0.0209 |

**ECE = 0.0052.** The de-vigged market is very well calibrated — largest gap is 2.1 percentage points, in the sparsest bin (n=19). The mid-range bins, which carry most of the mass, are accurate to within ~1.5 points.

V3 and blend calibration pending local run.

---

## 13. Leakage / Causality Checks

| Check | Status | How verified |
|---|---|---|
| No 2025/26 fixtures in dataset | **PASS** | Direct query: 0 rows with season = 2025/2026 |
| Market probs derived from odds only | **PASS** | `power_devig()` takes three floats; no label parameter exists |
| Fixture result never used to build market probability | **PASS** | De-vig is a pure function of `(O_H, O_D, O_A)` |
| Fixture result never used to select blend weight | **PASS** | `learn_weight()` receives only training-fold arrays |
| Future fixtures never influence V3 features | **PASS** | Causal Elo two-pass engine, validated bit-identical in Phase 1 |
| Future outcomes never influence blend weight | **PASS** | Weight frozen from training period before test application |
| All folds strictly chronological | **PASS** | Verified on actual data: train_max_unix < test_min_unix for both folds |
| Identical fixture set across A/B/C | **PASS** | Single joined DataFrame; all three read the same row index |
| Peak odds used | **NO** | Not selected in any query; excluded from the research dataset |
| Production DB modified | **NO** | All reads use SQLite `mode=ro` URI |
| Model artifact modified | **NO** | MD5 verified (Section 14) |

### Causality caveat — stated explicitly

The market signal is **Pinnacle closing odds**. OddAlerts provides **no independent per-record timestamps** on the `odds/history` endpoint (the timestamped `odds/movement` endpoint has 21-day retention and is unusable for these seasons).

Therefore: closing odds are treated as pre-match on the basis of **OddAlerts' definition**, not independently verified timing.

**We do not claim independently timestamped causality.** Opening odds — structurally the earliest recorded price — are carried as a robustness check precisely because their pre-match status is less dependent on that definition.

---

## 14. Integrity — Protected Files

Verified before and after all work in this phase:

| File | Expected MD5 | Actual MD5 | Status |
|---|---|---|---|
| `data/models/v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `data/models/v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `data/models/v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

`run_e2_experiment.py` re-runs this check automatically before and after the experiment and writes both to `e2_results.json`.

### `git status --short`

Not available — the workspace is not a git repository (`fatal: not a git repository`). Filesystem-level change list instead:

**Created (all under `research/market_odds/`):**

| File | Purpose |
|---|---|
| `e2_metrics.py` | Metric functions (LogLoss, Brier, RPS, Accuracy, Draw Recall, ECE, calibration) — pure numpy |
| `run_e2_experiment.py` | Full E2 harness: A, B, C, folds, weight learning, leakage checks, integrity |
| `e2_results.json` | Machine-readable results (Experiment B complete; A/C pending) |
| `E2_MARKET_EXPERIMENT_REPORT.md` | This report |

**Modified:** none. No file outside `research/market_odds/` was created or changed.

---

## 15. Limitations

1. **Experiments A and C not executed here.** scikit-learn is unavailable in the sandbox and cannot be installed (PyPI blocked, HTTP 403). V3 cannot be unpickled. Local execution required.

2. **Only 2 walk-forward folds.** Market coverage is 0% for 2020/21 and 2021/22, so no fold can test on those seasons. Two test periods is thin for a weight-selection procedure.

3. **Fold 1 trains on a biased 28.6% subset.** The 522 covered 2022/23 fixtures are not a verifiably random sample of that season's 1,827.

4. **Subset is 44.5% of the V3 dataset and recency-skewed.** Conclusions do not transfer to the full V3 historical set.

5. **No independent timestamp verification on closing odds.** See Section 13.

6. **Single bookmaker.** Pinnacle only. No cross-bookmaker consensus or disagreement signal is available in this dataset.

7. **Market draw recall ≈ 0.** Accuracy comparisons are dominated by H/A discrimination; draw performance must be read from log loss, Brier, and RPS instead.

8. **Weight grid is discrete** (0.05 steps). The true optimum may fall between grid points. This is intentional — a coarse grid resists overfitting on a 522-fixture training fold.

---

## 16. Final Decision

# INCONCLUSIVE

**Two independent reasons, both substantive:**

**1. The experiment is incomplete.** Experiments A and C did not run. Without V3's probabilities on these exact 4,001 fixtures, there is no V3 baseline to compare against and no blend to evaluate. Declaring PASS or FAIL now would be asserting a result that was never measured.

**2. Even after local execution, the evidence base is structurally thin.** Two folds, one of which trains on a 522-fixture biased subset, cannot firmly establish that a learned blend weight generalizes. This limitation is a property of the market data's coverage, not of the harness — running the script will not remove it.

**What was established, with measured evidence:**

- The de-vigged Pinnacle closing market is well calibrated (ECE 0.0052) and clearly beats both naive baselines (LL 0.96045 vs 1.07372 class prior, 1.09861 uniform)
- All 4,001 market fixtures match V3 feature rows — zero attrition, identical fixture set guaranteed across A/B/C
- Both folds are strictly chronological
- No leakage vector was found; no protected file changed

**What was not established:** whether market odds add incremental information **beyond V3**. That is precisely the question E2 exists to answer, and it remains open.

### To complete E2

```bash
cd "E:\Football Prediction Project"
python research/market_odds/run_e2_experiment.py
```

The script prints the full A/B/C comparison, per-fold learned weights, season and league breakdowns, calibration, leakage checks, and pre/post MD5, then writes the complete `e2_results.json` and applies the decision rule:

- **PASS** — all 3 pooled metrics (LogLoss, Brier, RPS) improve out-of-sample **and** improvement holds in ≥ 4 of 5 leagues
- **FAIL** — no pooled metric improves
- **INCONCLUSIVE** — anything mixed, or fewer than 2 folds

Even a PASS from that rule should be read against Limitations 2–4 above.

---

## ROADMAP PROGRESS

| Phase | Status |
|---|---|
| Phase 1 — V3 Controlled Experiment | **COMPLETE** |
| Phase 2 — Market Data Audit | **COMPLETE** |
| Phase 3 — Market Odds Ingestion | **COMPLETE** |
| Phase 3C — Market Audit | **COMPLETE** |
| **E2 — V3 vs Market** | **INCONCLUSIVE** |

**Market odds are inconclusive as an incremental signal beyond V3 for the five target leagues.**

The market is demonstrably a strong standalone predictor on this subset. Whether it adds anything V3 does not already capture has not been measured.

---

**No V4 promoted. Production default unchanged. V2 unmodified. V3 artifact unmodified. Research only.**
