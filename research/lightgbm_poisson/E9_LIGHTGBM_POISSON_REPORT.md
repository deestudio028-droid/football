# E9 — LightGBM Poisson Controlled Experiment Report

**Date:** 2026-08-20
**Type:** Research experiment only — nothing promoted, no production change
**Question:** Does nonlinear gradient-boosted Poisson regression extract generalizable signal from the existing causal 87-feature V3 contract that the linear Poisson model misses?

---

## EXECUTION STATUS

Implementation complete through step 10 of the approved order. **Step 11 (the walk-forward run) executes on your local machine**, where LightGBM 4.7.0 and scikit-learn 1.9.0 are installed.

| Step | Status |
|---|---|
| 1. Protected MD5 PRE | **COMPLETE** — all 5 identical |
| 2–6. Audits (V3, availability, contract, grid, folds) | **COMPLETE** |
| 7. Freeze hyperparameter list | **COMPLETE** — 16 configs, `hyperparameter_grid.json` |
| 8. Implement isolated module | **COMPLETE** |
| 9. Implement tests | **COMPLETE** |
| 10. Run full test suite | **COMPLETE — 109/109 passed, 0 failed, 4 skipped** |
| 11–18. Experiment, repeat, integrity | **AWAITING LOCAL RUN** |
| 19–20. Report, roadmap | **THIS DOCUMENT** |

The 4 skipped tests are the sklearn/LightGBM suites; they run in full locally.

The harness was dry-run here and behaves correctly: it verified integrity, then halted cleanly at the `import lightgbm` boundary. Protected files unchanged before and after.

---

## Two test failures found and handled — differently, on purpose

The suite initially reported 2 failures. They were not the same kind of problem and I did not treat them the same way.

### Failure 1 — a real code gap. I fixed the code.

`shared_grid_k` rejected only a non-positive *maximum*. An array such as `[-1.0, 1.0]` passed, because the max is 1.0. A corrupt negative rate would have reached the grid computation undetected whenever any other fixture was positive.

```python
if np.any(a <= 0):
    raise E9Error(f"{n_bad} non-positive lambda value(s) supplied ...")
```

The test was right and the code was wrong.

### Failure 2 — a badly-specified test. I fixed the test.

The assertion "V3 contract shares no probability column with market data" failed on `competition_id`. But `competition_id` is a league identifier legitimately present in both tables — not a probability, not market signal. My assertion said "probability column" and implemented "any column at all".

Replaced with two precise checks: no column prefixed `devig_`/`opening_`/`closing_`/`peak_`/`pm_` or containing `odds`/`bookmaker` appears in the contract (**overlap: none**), and the only shared column is `competition_id`, explicitly labelled benign metadata.

This is the third time in this series (E7, E8, now E9) that a lazy string-or-set assertion has produced a spurious failure. The pattern is mine and worth naming: **assert the property, not a proxy for it.**

---

## 1. Roadmap Status

| Experiment | Status |
|---|---|
| E0 — V2 Baseline | COMPLETE |
| E1 — Causal Elo → V3 | PASS |
| E2 — Market Odds | PASS |
| E3 — Dixon-Coles | INCONCLUSIVE |
| E4 — Exponential Time Decay | INCONCLUSIVE |
| E5 — Quadratic Elo → xG | INCONCLUSIVE |
| E6 — Online Attack/Defense | PASS |
| E7 — Best Proven Combination | FAIL |
| E8 — Temperature Scaling | FAIL |
| **E9 — LightGBM Poisson** | **THIS REPORT** |

---

## 2. The E2 Leakage Finding and the Honest Benchmarks

E2 scored the **shipped V3 artifact** on fixtures it had been trained on. The artifact's `FINAL_TRAIN_SEASONS` spans 2020/21–2024/25, which contains both of E2's test seasons. Discovered in E7, root-caused in E8.

**Benchmarks used by E9, recorded before the run:**

| Reference | Log Loss | Universe |
|---|---|---|
| **Honest V3** (per-fold refit) | **0.987025** | Primary, n = 5,331 |
| Honest V3 (per-fold refit) | 0.990435 | Market overlap, n = 3,479 |
| **Market alone** | **0.955905** | Market overlap, n = 3,479 |
| ~~E2's V3~~ | ~~0.981357~~ | **NOT USED — in-sample contaminated** |

The harness hard-codes these as `V3_REF_5331`, `V3_REF_3479`, `MARKET_REF_3479` and prints the deviation between Arm A and its reference, so a drift in the baseline is visible immediately.

**The two universes are computed and printed separately and never mixed.** A market comparison on the 5,331 universe would be invalid, and the harness does not produce one.

---

## 3. Environment

| Component | Version |
|---|---|
| LightGBM | 4.7.0 (local) |
| scikit-learn | 1.9.0 (local) |
| scipy | 1.18.0 (local) |
| Python | 3.13 (local) |

Recorded into `e9_results.json` at runtime. **Results are not comparable across LightGBM major versions**, so the version is part of the record.

---

## 4. Feature Contract

**Exactly the 87 V3 features, same order, nothing added or removed.**

Verified by test: 87 columns, no duplicates, Elo at positions 85–87, and none of `A_home`/`D_home`/`A_away`/`D_away` (E6), `elo_diff_sq` (E5), or any market probability column present.

`competition_id` is handled by **one-hot encoding**, matching V3 exactly. The audit had proposed LightGBM native categoricals; I did not use them, because §13 requires identical preprocessing between arms and native categoricals would have changed the encoding for one arm only. Contract stays 87 columns → 91 encoded (86 numeric + 5 one-hot).

---

## 5. Missing-Data Policy (approved)

**Median imputation for BOTH arms, medians computed from training-fold rows only.**

This is material, not cosmetic: measured missingness is **3.297% of cells across 83 of 87 columns**, worst being four venue-season features at 1,164 NaN each.

Native LightGBM NaN handling is deliberately **not** used. It would likely help — the missingness is informative — but it would change the missing-data treatment *and* the learner, confounding the single variable E9 exists to test.

Verified by test: medians match `np.nanmedian` on training rows exactly; mutating test rows cannot alter fitted statistics.

---

## 6. Exact Model Formulation

```
ARM A   87 features → V3Preprocessor → PoissonRegressor(alpha=1.0, max_iter=2000)  ×2
ARM B   87 features → V3Preprocessor → LGBMRegressor(objective="poisson")          ×2
                                    ↓
                      lambda_home, lambda_away
                                    ↓
                      SHARED adaptive K  ← the critical control
                                    ↓
                      identical 1X2 conversion
```

Two independent goal models per arm. Targets `label_home_goals`, `label_away_goals`. **No 1X2 classifier anywhere.**

`V3Preprocessor` is a numpy reimplementation of production's `LogisticRegressionPreprocessor`, used so the research module needs no production import and so the matrix handed to both arms is provably the same object. Suite 3 asserts equivalence against the production class to **< 1e-9** when sklearn is available.

---

## 7. The Shared Adaptive K Control

`src/models/poisson.py::_grid_size` derives K from `max(lam_h.max(), lam_a.max())` **across the batch**. If each arm called `predict_poisson` independently they would derive **different K** whenever their maximum lambdas differ — silently giving the arms different score grids.

`shared_grid_k()` computes one K from the pooled maximum across **both** arms; `hda_from_lambdas()` converts at that fixed K for both.

**The test proves the failure mode is real**, not hypothetical: with deliberately different maxima, the arms would have derived `K_A = 25` vs `K_B = 29`. The shared function returns 29 for both.

Also verified: `grid_size` matches production `_grid_size` across 158 lambda values with max diff 0; conversion matches production `hda_tail_safe` to < 1e-12; the function is order-independent.

---

## 8. Hyperparameter Grid — Frozen

`hyperparameter_grid.json`, **16 configurations**, frozen in the Phase 1 audit before any outer-test result existed.

Fixed for all: `objective="poisson"`, `random_state=42`, `n_jobs=1`, `deterministic=True`, `force_row_wise=True`.

Config 1 is the §9 default (leaves 15, lr 0.05, trees 250, min_child 50, sub 0.8, col 0.8, L2 1.0). Configs 2–10 vary one axis at a time; 11–16 sample the low-capacity corner, since fold 1 trains on only 3,652 rows.

Verified by test: exactly 16, unique IDs, **every value inside the §8 permitted grid**, `subsample_freq=1` set when subsampling (otherwise LightGBM silently ignores `subsample`), and **no early stopping** — no `early_stopping_rounds`, no `eval_set`, so no validation data is consumed and nothing can leak.

**Selection:** nested chronological validation inside each outer training period, by mean inner-validation log loss, then frozen and refit on the full outer training fold. `select_config`'s signature is `(inner_splits, fit_eval, configs, fixed)` — no outer-test parameter exists.

Inner splits available: fold_1 → 1, fold_2 → 2, fold_3 → 3.

---

## 9. Walk-Forward Protocol and Dataset

| Fold | Train | Test | n_train | n_test |
|---|---|---|---|---|
| fold_1 | 2020/21, 2021/22 | 2022/23 | 3,652 | 1,827 |
| fold_2 | + 2022/23 | 2023/24 | 5,479 | 1,752 |
| fold_3 | + 2023/24 | 2024/25 | 7,231 | 1,752 |

**Universe: 8,983 eligible · 5,331 pooled OOS.** Five leagues verified. **2025/26: 0 fixtures**, asserted in the harness. All folds verified strictly chronological.

---

## 10–20. Results

**AWAITING THE LOCAL RUN.** The harness produces, into `e9_results.json` and stdout:

pooled primary-universe metrics · per-fold metrics with deltas · league-wise · season-wise · market-overlap block (separate) · selected config per fold · full config curve · train/test gaps per arm · lambda diagnostics (min/max/mean/p01/p50/p99, NaN/Inf flags) · prediction-change diagnostics · top-20 feature importance by gain and split · reliability bins · config stability · determinism.

No projected or proxy numbers appear in this report. There are none to report.

---

## 21. Lambda Safety

`lambda_diagnostics` verifies positivity, finiteness, and that no rate exceeds 15.0, reporting full percentile distributions.

**The harness STOPS and returns exit code 1 if Arm B produces unsafe lambdas**, printing the diagnostic — per §19, which forbids silently clipping. `clipping_applied` is hard-coded `False` and asserted by test. If clipping ever becomes necessary that is a methodological finding requiring discussion, not a runtime patch.

LightGBM's `objective="poisson"` uses a log link, so predictions are structurally positive; the check is a guard, not an expectation.

---

## 22. Leakage Tests

109/109 tests pass. The §22 checklist:

| # | Check | Status |
|---|---|---|
| 1–3 | Outcomes cannot affect features / earlier predictions | **PASS** — causal Elo inherited from E1; features precomputed |
| 4 | Test labels cannot alter hyperparameters | **PASS** — adversarial: rewrote every test label, selected config unchanged |
| 5 | Test labels cannot alter coefficients | **PASS** — fitting functions take evaluation *features* only |
| 6 | Hyperparameters from inner training only | **PASS** — signature introspection |
| 7–8 | Neither arm sees outer-test labels when fitting | **PASS** — 4th argument is `X_ev_enc` in both |
| 9 | No market data in the feature matrix | **PASS** — no odds-derived column in the contract |
| 10 | No E6 A/D states | **PASS** |
| 11 | No E3/E4/E5 transformations | **PASS** — no `rho`, `elo_diff_sq`, `temperature` reference |
| 12 | 2025/26 = 0 | **PASS** |
| 13 | Same data → identical results | **PASS** |
| 14 | Feature ordering identical between arms | **PASS** — one shared matrix |
| 15 | Same score grid both arms | **PASS** — shared K, verified |

Top-level imports are stdlib + numpy/pandas only (AST-verified); sklearn and lightgbm are imported *inside* the fitting functions, so every contract, preprocessing, grid and metric routine is testable without them.

---

## 23. Determinism

Preprocessing, `shared_grid_k` and conversion all bit-identical on repeat. `random_state=42`, `n_jobs=1`, `deterministic=True`, `force_row_wise=True` pinned.

**`n_jobs=1` is not incidental.** LightGBM is only bit-reproducible with a fixed thread count; multi-threaded histogram construction reorders floating-point accumulation. The harness re-fits fold 1 twice and reports whether predictions are identical.

---

## 24. Protected MD5

Verified before and after the dry run:

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## 25. Git Status / Files

`git status --short` unavailable — not a git repository.

**INTENDED:** `research/lightgbm_poisson/*` · **UNEXPECTED:** none.

**Created:** `E9_AUDIT_REPORT.md`, `E9_DEPENDENCY_INSTALL_ATTEMPT.md`, `hyperparameter_grid.json`, `lightgbm_poisson.py`, `test_lightgbm_poisson.py`, `run_e9_experiment.py`, `E9_LIGHTGBM_POISSON_REPORT.md`. (`e9_results.json` on the local run.)

**Untouched:** `requirements.txt`, `src/*`, `data/*`, `predict_match.py`, `config/*`, all other research directories.

---

## 26. Known Limitations

1. **Fold 1 has only one inner split** — its configuration rests on a single validation season.
2. **Only 3 folds**, the standing limit across this series.
3. **16 configurations, not exhaustive** — deliberately, per §8.
4. **Median imputation may handicap LightGBM** on 3.3% of cells. Accepted to isolate the variable; a native-NaN diagnostic is a separate future experiment.
5. **One-hot rather than native categoricals**, chosen for identical preprocessing over marginal LightGBM efficiency.
6. **LightGBM version-sensitive** — 4.7.0 recorded; results may shift across major versions.

---

## 27. Final E9 Decision

# E9: PENDING LOCAL EXECUTION

No decision is recorded because no measurement exists. The harness applies all five PASS criteria mechanically and distinguishes two outcomes §35 requires be kept separate:

- **`decision`** — did LightGBM beat honest V3?
- **`champion_candidate`** — did it also beat market alone (0.955905)?

A model that beats V3 but loses to market is recorded as *"Improved model over V3, but market remains stronger"* with `champion_candidate: false`. **Beating V3 is not promotion.**

**What I expect, stated before seeing results:** CASE B or CASE C. LightGBM would need roughly a **3.2% log-loss improvement** over V3 (0.987025 → 0.955905) merely to match the market, and no fitted model in this project has come within 3% of it. Gradient boosting on 3.6k–7.2k rows with 87 correlated features is also squarely in the regime where regularised linear models are competitive. That is an expectation, not a result, and the harness will overrule it if the data says otherwise.

---

## 28. Commands to Run Locally

```powershell
cd "E:\Football Prediction Project"

# 1. Full test suite — must be 113/113 with zero skips before proceeding
python research\lightgbm_poisson\test_lightgbm_poisson.py

# 2. The experiment
python research\lightgbm_poisson\run_e9_experiment.py
```

Use the Python that has the libraries:

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe research\lightgbm_poisson\test_lightgbm_poisson.py
C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe research\lightgbm_poisson\run_e9_experiment.py
```

**Expected on the test run:** `Suites: 14  Pass: 113  Fail: 0  Skip: 0`. The 4 currently-skipped tests activate once LightGBM and sklearn are present. **If any test fails, stop** — the harness is not the place to debug a contract violation.

**Expected on the experiment:** roughly 16 configs × ~6 inner fits + final fits ≈ 110 LightGBM fits, single-threaded. Several minutes to tens of minutes. Sanity check as it runs: Arm A's pooled log loss should land near **0.987025**; a large deviation means the baseline drifted and the run should be stopped.

Then send me `e9_results.json` and I'll complete sections 10–20 and record the decision.

---

## ROADMAP PROGRESS

```
E0 — V2 Baseline                  COMPLETE
E1 — Causal Elo → V3              PASS
E2 — Market Odds                  PASS
E3 — Dixon-Coles                  INCONCLUSIVE
E4 — Exponential Time Decay       INCONCLUSIVE
E5 — Quadratic Elo → xG           INCONCLUSIVE
E6 — Online Attack/Defense        PASS
E7 — Best Proven Combination      FAIL
E8 — Temperature Scaling          FAIL
E9 — LightGBM Poisson             PENDING LOCAL EXECUTION
E10 — Match Importance + Squad    QUEUED
E11 — Advanced Tactical Efficiency QUEUED
E12 — Final Champion Selection    PENDING
E13 — Final 5-League Walk-Forward PENDING
E14 — Production Inference        PENDING
E15 — V2 → V4/V5 Promotion        PENDING
E16 — Production Monitoring       PENDING
```

---

**Nothing promoted. No production file modified. requirements.txt untouched. No market odds, Dixon-Coles, time decay, quadratic Elo or online A/D. Research only.**
