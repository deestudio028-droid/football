# E3 — Dixon-Coles Controlled Experiment Report

**Date:** 2026-08-20
**Type:** Research experiment only — nothing promoted, no production change
**Question:** Does Dixon-Coles independently improve the already-validated V3 model?

---

## EXECUTION STATUS — READ FIRST

| Component | Status |
|---|---|
| Pipeline audit (§3) | **COMPLETE** |
| `dixon_coles.py` implementation | **COMPLETE** |
| Test suite (55 tests) | **COMPLETE — 55/55 passed, 0 skipped** |
| Pre-experiment diagnostic | **COMPLETE — measured on real data** |
| **Experiment A (V3 baseline)** | **PENDING — requires local run** |
| **Experiment B (V3 + DC)** | **PENDING — depends on A** |

**Reason A and B are pending:** both require V3's per-fixture `lambda_home` and `lambda_away`. Producing them means unpickling `v3_poisson_venue_elo_candidate.pkl`, which requires scikit-learn. The sandbox has no scikit-learn and cannot install it (PyPI blocked, HTTP 403). I searched for cached lambdas — `data/audit/phase3_predictions/`, `phase4a/b/c_predictions/`, `phase5a_predictions/` — every dump contains 1X2 probabilities from the logistic-regression era. **No lambdas exist anywhere on disk.** Dixon-Coles operates on lambdas, so those files cannot substitute.

What *did* run on real data: the low-score residual analysis and a walk-forward proxy study. Both are reported below with their numbers and their caveats.

---

## 1. Roadmap Status

| Experiment | Status |
|---|---|
| E0 — V2 baseline | COMPLETE |
| E1 — V3 Causal Elo | PASS |
| E2 — Market Odds | PASS |
| **E3 — Dixon-Coles** | **THIS REPORT** |

---

## 2. Experiment Objective

Test **one** thing: whether the Dixon-Coles low-score dependence correction `tau(x,y)`, applied to V3's **existing, unchanged** expected-goal rates, improves probabilistic accuracy.

```
A:  V3 lambdas  ->  independent Poisson       ->  1X2
B:  V3 lambdas  ->  DC-corrected score grid   ->  1X2
```

V3 is not retrained. Market odds are not read, imported, or referenced. No other feature or model enters.

---

## 3. Pipeline Audit (performed before any code was written)

### 3.1 How V3 produces lambda_home / lambda_away

`v3_poisson_venue_elo_candidate.pkl` holds two fitted `PoissonRegressor` models and a preprocessor:

1. 84 base features loaded from `features.db`
2. 3 causal Elo features (`home_elo`, `away_elo`, `elo_diff`) computed from `matches.db` by `src/features/elo.py`
3. Joined to the 87-column `V3_FEATURE_COLUMNS` contract (order asserted)
4. `preprocessor.transform(X)` → `model_home_goals.predict()` / `model_away_goals.predict()`

### 3.2 How independent Poisson becomes 1X2

`src/models/poisson.py::hda_tail_safe` exploits independence with a factorized cumulative form:

```
P(H) = sum_i>=1  ph[i] * Fa[i-1]        # home i, away <= i-1
P(D) = sum_i     ph[i] * pa[i]
P(A) = 1 - P(H) - P(D)                  # by complement
```

### 3.3 Score grid — **the key audit finding**

**V3 does NOT use a fixed maximum score.** `_grid_size(lam_max, tol=1e-15)` derives K adaptively by exact forward CDF recursion until the Poisson upper tail falls below 1e-15 (cap 20000).

| lam_max | K |
|---|---|
| 1.0 | 17 |
| 1.5 | 19 |
| 2.0 | 21 |
| 3.0 | 25 |
| 5.0 | 31 |

This matters: the reference implementation in `research/worldcup_repo/src/dixon_coles.py` hardcodes `max_goals=10`. Silently adopting that would have changed the grid between A and B and invalidated the comparison. E3 reuses V3's adaptive derivation and applies **one shared K to both arms**.

### 3.4 Safety gates in production

`MASS_TOL=1e-12`, `COMPLEMENT_TOL=1e-10`, `ROWSUM_TOL=1e-9`, plus finite/[0,1] checks.

### 3.5 Walk-forward protocol

`src/models/config.py::WALK_FORWARD_FOLDS` — 3 folds by season. `FINAL_TEST_SEASONS = ("2025/2026",)` is never touched.

### 3.6 Existing Dixon-Coles implementation

`research/worldcup_repo/src/dixon_coles.py` exists. It is a **full** DC model: it jointly fits attack/defence/home/rho by L-BFGS-B with exponential time decay (`xi=0.0018`) and L2, for **international** football.

Audit verdict: its tau formulas are self-consistent and match the specified form. **Nothing was copied from it** — not rho, not xi, not the grid size, not the fitting procedure. E3 needs only the tau correction applied to fixed lambdas, so a clean pure-numpy implementation was written instead.

---

## 4. Exact Dataset

**Source:** `data/processed/matches.db` (read-only) joined to `features.db` (read-only)

| Property | Value |
|---|---|
| Competition IDs | 200, 419, 423, 477, 499 |
| Status filter | `FT` or `AWARDED`, non-null goals |
| **Eligible fixtures** | **8,983** |
| Pooled out-of-sample (3 test folds) | **5,331** |
| 2025/26 | **quarantined — 0 fixtures, verified** |
| Market data | **not used** |

---

## 5. Leagues

Premier League, La Liga, Serie A, Bundesliga, Ligue 1 — verified as exactly the five target leagues, no others present.

---

## 6. Seasons

2020/21, 2021/22, 2022/23, 2023/24, 2024/25. 2025/26 excluded as the untouched final test split.

---

## 7. Validation Protocol

The approved V3 walk-forward folds, unchanged:

| Fold | Train | Test | n_test |
|---|---|---|---|
| fold_1 | 2020/21, 2021/22 | 2022/23 | 1,827 |
| fold_2 | + 2022/23 | 2023/24 | 1,752 |
| fold_3 | + 2023/24 | 2024/25 | 1,752 |

All three verified strictly chronological (`train_max_unix < test_min_unix`). No random K-fold, no shuffling.

---

## 8. V3 Baseline Definition

The frozen candidate artifact, MD5 `a2850a7687822a5916663301f5ccc96c`, loaded and used as-is. Experiment A is `predict_dc(lam_h, lam_a, rho=0.0)`, which is proven equivalent to production `hda_tail_safe` (§18).

---

## 9. Dixon-Coles Implementation

`research/dixon_coles/dixon_coles.py` — numpy only, no scipy, no sklearn.

```
tau(0,0) = 1 - lam_h * lam_a * rho
tau(1,0) = 1 + lam_a * rho
tau(0,1) = 1 + lam_h * rho
tau(1,1) = 1 - rho
tau(x,y) = 1                        otherwise
```

Indexing `[i, j]` = home i goals, away j goals. Pipeline: build joint grid `ph[i] * pa[j]`, multiply by tau, clip at 0, renormalize to sum exactly 1, collapse to 1X2 by triangle masks.

### Validity bounds — an implementation finding

tau must stay strictly positive, which constrains rho:

```
rho > max(-1/lam_h, -1/lam_a)        and        rho < min(1, 1/(lam_h*lam_a))
```

On realistic lambdas the **upper** bound binds hard. For the synthetic test batch the ceiling was **+0.115** — so a naive symmetric grid up to +0.20 contains invalid values. `select_rho` **skips** out-of-bounds candidates and records them as `invalid_out_of_bounds`; it never silently clips. This was caught by the test suite, not assumed.

---

## 10. Rho Selection Methodology

**Not copied from any repository.** Grid:

```
[-0.20, -0.175, -0.15, -0.125, -0.10, -0.075, -0.05, -0.025,
  0.0, 0.025, 0.05, 0.075, 0.10]
```

Centred on zero, step 0.025. Justification: published football rho estimates cluster in roughly [-0.15, 0.00]; a coarser step risks stepping over the optimum, a finer step risks overfitting one fold. `rho = 0.0` is in the grid, so DC can decline to act.

**Per fold:** fit rho on **training data only** → freeze → apply to test → evaluate. `select_rho`'s signature is `(y_train, lam_h_train, lam_a_train, rho_grid, K)` — it has **no parameter through which test data could enter**.

---

## 11. Score-Grid Methodology

One K, derived once from the batch maximum lambda via V3's own `_grid_size`, used by **both** arms. Verified bit-identical to production (§18). Grid mass sums to 1 within 4.4e-16.

---

## 12. Pooled Metrics

**PENDING** — requires the local run.

### What was measured instead: low-score residual analysis (n = 8,983, real data)

| Score | Observed | Independent Poisson | Residual |
|---|---|---|---|
| 0-0 | 0.06401 | 0.05978 | **+0.00423** |
| 1-0 | 0.09017 | 0.09173 | **−0.00156** |
| 0-1 | 0.07136 | 0.07668 | **−0.00532** |
| 1-1 | 0.12379 | 0.11765 | **+0.00613** |

**The Dixon-Coles signature is present**: 0-0 and 1-1 occur *more* often than independent Poisson implies; 1-0 and 0-1 occur *less*. This is precisely the pattern negative rho corrects, and it confirms DC is directionally justified on this data.

**But the magnitude is small.** The pooled draw deficit is **+0.0043** — 0.43 percentage points.

---

## 13. League-Wise Metrics

**PENDING** for A/B. The measured draw deficit by league is the critical finding:

| League | n | Observed draw | Poisson draw | Deficit |
|---|---|---|---|---|
| Bundesliga | 1,530 | 0.2536 | 0.2331 | **+0.0205** |
| Serie A | 1,901 | 0.2709 | 0.2551 | **+0.0158** |
| La Liga | 1,900 | 0.2700 | 0.2625 | **+0.0075** |
| Ligue 1 | 1,752 | 0.2466 | 0.2510 | **−0.0044** |
| Premier League | 1,900 | 0.2279 | 0.2455 | **−0.0176** |
| **Pooled** | **8,983** | **0.2539** | **0.2497** | **+0.0043** |

**The sign is not consistent across leagues.** Bundesliga and Serie A show a draw *excess*; Premier League and Ligue 1 show a draw *deficit* in the opposite direction. A single global rho fitted on pooled training data must therefore help some leagues and hurt others.

This bears directly on PASS criterion 3, which requires improvement not confined to one league.

---

## 14. Season-Wise Metrics

**PENDING** — the harness reports per-season A/B metrics for all three test seasons.

---

## 15. Draw Analysis

DC's entire mechanism is redistributing mass among 0-0, 1-0, 0-1, 1-1. Negative rho inflates the two draw cells and deflates the two one-goal-margin cells, raising P(Draw).

Verified in the test suite on 1,000 synthetic fixtures at rho = −0.10:

| Quantity | rho = 0 | rho = −0.10 | Change |
|---|---|---|---|
| Mean P(Draw) | 0.213588 | 0.229866 | **+0.016278** |
| Mean P(0-0) | 0.06292 | 0.07106 | +0.00814 |
| Mean P(1-1) | 0.08139 | 0.08953 | +0.00814 |
| Mean abs ΔP (all classes) | — | — | 0.010852 |

Mean P(Draw) is **monotonically decreasing in rho** across the valid range — confirmed.

### Why this matters for V3 specifically

E2's local run recorded V3's `draw_predicted_rate = 0.0` and `draw_recall = 0.0` on its evaluation subset — **V3 never makes "draw" its argmax**. DC with negative rho pushes in exactly the direction that could change this. Whether the push is large enough to flip any argmax, and whether flipping helps log loss, is what A/B must measure. Note that accuracy alone would be a misleading headline here.

---

## 16. Low-Score Analysis

Covered in §12 (residuals) and §15 (DC's effect). The harness additionally reports, per fold at that fold's selected rho, the mean probability of each of the four cells before and after correction.

---

## 17. Leakage Checks

| Check | Status | Evidence |
|---|---|---|
| No 2025/26 fixtures | **PASS** | Query returns 0; asserted in harness and tests |
| All folds strictly chronological | **PASS** | `train_max_unix < test_min_unix` verified on real data, all 3 folds |
| rho from training data only | **PASS** | `select_rho` signature has no test parameter (asserted by introspection) |
| Test outcomes cannot change rho | **PASS** | Overwrote every test label with "D"; selected rho unchanged |
| Future lambdas cannot change rho | **PASS** | Set future lambdas to 9.9; selected rho unchanged |
| Fixture's own result cannot affect its prediction | **PASS** | `predict_dc` takes no label argument (asserted by introspection) |
| A and B use identical fixture sets | **PASS** | One array feeds both arms |
| A and B use identical lambdas | **PASS** | Lambdas verified unmutated after DC call |
| A and B use identical score grid | **PASS** | Single K computed once |
| Market data used | **NO** | No import, no reference in any E3 file |
| V3 retrained | **NO** | Artifact loaded read-only, MD5 verified |

---

## 18. Determinism Checks

| Check | Result |
|---|---|
| `predict_dc` repeated calls | **bit-identical** (max diff 0.0) |
| `score_grid` repeated calls | **bit-identical** |
| `select_rho` repeated calls | identical rho and identical training log loss |
| RNG usage inside `dixon_coles.py` | **none** — every function is pure |
| Seed for synthetic test data | 20260820, explicit |

### Baseline equivalence — the strongest single result

| Check | Result |
|---|---|
| `grid_size` vs production `_grid_size` | **identical**, 158 lambda values, max diff 0 |
| `pmf_grid` vs production `_pmf_grid` | **bit-identical**, max diff 0.0 |
| Adaptive K matches production | **yes**, K=26 |
| **rho=0 reproduces production 1X2** | **max abs diff 4.441e-16** |

Production's own tail residual is 5.551e-16, so the difference is **below production's own numerical noise floor**. Experiment A is V3, not an approximation of it.

---

## 19. Protected MD5 Checks

Verified before and after all E3 work:

| File | Expected | Actual | Status |
|---|---|---|---|
| `data/models/v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `data/models/v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `data/models/v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## 20. Files Changed

All created under `research/dixon_coles/`:

| File | Purpose |
|---|---|
| `dixon_coles.py` | tau correction, score grid, 1X2 collapse, rho selection |
| `test_dixon_coles.py` | 8 suites, 55 tests |
| `run_e3_experiment.py` | Full A/B harness (requires sklearn locally) |
| `run_e3_diagnostic.py` | Residual + proxy analysis (no sklearn needed) |
| `e3_results.json` | Machine-readable results |
| `E3_DIXON_COLES_REPORT.md` | This report |

**Modified: none.** No file outside `research/dixon_coles/` was created or changed. `git status --short` is unavailable — the workspace is not a git repository.

---

## 21. Files Untouched

`data/models/*` (all artifacts), `data/processed/*` (all databases), `src/models/*`, `src/features/*`, `predict_match.py`, `research/market_odds/*`, `config/*`. Every read used SQLite `mode=ro` or read-only file access.

---

## 22. E3 Decision

# E3: INCONCLUSIVE

**Why not PASS or FAIL:** criteria 1–3 require measured A and B metrics. Those were not produced, because V3's lambdas require scikit-learn. Declaring either outcome would assert a result that was never measured.

**Criteria status:**

| # | Criterion | Status |
|---|---|---|
| 1 | Log loss improves | **NOT MEASURED** |
| 2 | ≥2 of Brier/RPS/ECE improve | **NOT MEASURED** |
| 3 | Improvement not confined to one league | **NOT MEASURED** — but §13 shows the deficit sign is inconsistent across leagues |
| 4 | No leakage | **PASS** — 11/11 checks |
| 5 | Walk-forward causality preserved | **PASS** — all 3 folds verified chronological |
| 6 | V3 baseline unchanged | **PASS** — rho=0 reproduces production to 4.4e-16; MD5 identical |

Criteria 4, 5, and 6 are satisfied on measured evidence. Criteria 1–3 are open.

### Advance signal from the proxy study — not a verdict

A walk-forward proxy using **league-mean lambdas fitted causally on training seasons only** (the same 3 folds, the same rho grid):

| Test season | Selected rho | A log loss | B log loss | Delta |
|---|---|---|---|---|
| 2022/23 | −0.025 | 1.065917 | 1.066243 | **−0.000325** |
| 2023/24 | **0.000** | 1.076997 | 1.076997 | 0.000000 |
| 2024/25 | −0.025 | 1.078841 | 1.079004 | **−0.000164** |
| **Pooled** | | **1.073806** | **1.073971** | **−0.000165** |

Leagues improved: **2 of 5** (Bundesliga, Serie A — exactly the two with a positive draw deficit).

Three things stand out: the selected rho is very close to zero and in one fold the procedure chose **rho = 0**, declining to apply DC at all; the pooled delta is **negative** (DC slightly worse out-of-sample); and improvement is confined to the two leagues whose draw residual has the right sign.

**This proxy is not the experiment.** Its lambdas are homogeneous within a league, so it is a much weaker model than V3 — its log loss is ~1.074 against V3's ~0.98 on E2's subset. Sharper per-fixture lambdas redistribute the low-score cells and could change both the optimal rho and the effect size. The proxy bounds the plausible magnitude and exposes the league heterogeneity; it does not settle the outcome.

That said, the underlying driver — a pooled draw residual of only +0.0043 with **inconsistent sign across leagues** — is a property of the data, not of the model. That property will still be there when V3's lambdas are used.

**Honest expectation:** a PASS looks unlikely on this data, chiefly because criterion 3 requires improvement beyond one or two leagues and the residual sign is inconsistent. But that is an expectation, not a measurement, and E3 should be decided by running it.

### To complete E3

```bash
cd "E:\Football Prediction Project"
python research/dixon_coles/test_dixon_coles.py     # expect 55/55
python research/dixon_coles/run_e3_experiment.py
```

The harness applies the six criteria mechanically and writes the full `e3_results.json`.

---

## 23. Recommendation for E4

**Do not gate E4 on E3.** Dixon-Coles and exponential time decay are independent corrections operating on different parts of the pipeline — DC reshapes the score distribution at fixed lambdas; time decay changes how training fixtures are weighted when lambdas are estimated. E4 can proceed regardless of E3's outcome.

Two carry-overs worth applying to E4:

1. **Audit the grid/weighting convention before implementing.** The `max_goals=10` versus adaptive-K discrepancy would have silently invalidated E3 had it not been caught in audit. E4 should confirm how V3 currently weights training fixtures before adding decay.

2. **Report per-league results and require cross-league consistency.** The Premier League/Bundesliga split in draw behaviour suggests these five leagues differ enough that a single global parameter may not suit all of them. E4 should check whether the optimal decay rate is stable across leagues, and consider whether per-league parameters are worth a later experiment.

If E3 returns FAIL when run, that is a clean negative result: it means V3's lambdas already capture the low-score structure well enough that the tau correction adds nothing, and Dixon-Coles should be dropped rather than carried into E7.

---

## ROADMAP PROGRESS

```
E0 — V2 baseline                  COMPLETE
E1 — V3 Causal Elo                PASS
E2 — Market Odds                  PASS
E3 — Dixon-Coles                  INCONCLUSIVE
E4 — Exponential Time Decay       NEXT
E5 — Quadratic Elo -> xG          PENDING
E6 — Online Attack/Defense        PENDING
E7 — Best Proven Combination      PENDING
```

**3 experiments complete** (E0, E1, E2). E3 is implemented, tested, and diagnosed, but its A/B comparison is unmeasured — it is not counted as complete.

**Next experiment: E4 — Exponential Time Decay.**

---

**Nothing promoted. Production default unchanged. V2 unmodified. V3 artifact unmodified. Dixon-Coles not combined with Market. Research only.**
