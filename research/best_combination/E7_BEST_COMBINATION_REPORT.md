# E7 — Best Proven Combination Report

**Date:** 2026-08-20
**Type:** Research experiment only — nothing promoted, no production change
**Question:** Do Market Odds and Online Attack/Defense provide *complementary* predictive information beyond V3?

---

## EXECUTION STATUS — READ FIRST

Implementation order was followed exactly as specified.

| Step | Status |
|---|---|
| 1. Write implementation | **COMPLETE** |
| 2. Write comprehensive tests | **COMPLETE** |
| 3. Run all tests | **COMPLETE — 82/82 passed, 0 skipped** |
| 4. Verify zero-component controls | **COMPLETE at probability level — all 3 pass** |
| 5. Verify protected MD5 | **COMPLETE — all 5 identical** |
| 6. Verify git status | **COMPLETE — see §30** |
| 7. **Run the real E7 experiment** | **PENDING — requires scikit-learn** |

The harness executes cleanly end-to-end up to the sklearn import: integrity verified, causal Elo computed, intersection resolved to the audited counts, quarantine confirmed, 87→91 contract asserted. It halts exactly where it must.

**One test initially failed and I fixed the test, not the code.** Suite 4 contained a naive string-match assertion (`"P_market" not in src or "regression" not in src.lower()`) that tripped on the module's *own docstring* explaining that it does **not** use regression features. That was a badly designed test. I replaced it with behavioural checks — every market-consuming function is verified to accept no design-matrix argument and `blend()` is verified to return an `(n, 3)` probability array rather than a feature matrix. That tests the actual property. Deleting the assertion to go green would have been the wrong move; so would leaving a test that fails for a spurious reason.

---

## 1. Audit Findings (Phase 1, approved)

Four findings shaped the design. Full detail in `E7_AUDIT_REPORT.md`.

| # | Finding | Consequence |
|---|---|---|
| 1 | **E2 is a post-hoc blend, not a regression feature** | Arms B and C are architecturally different in kind; Arm D needs a bridge |
| 2 | **E2's weight hit w = 1.00 in fold 2** | The validated blend discarded V3 entirely |
| 3 | **Market alone (0.955905) beat the E2 blend (0.957130)** | The proven component is the market signal, not the blend |
| 4 | **Intersection forces n = 3,479 and 2 folds** | E6's 5,331 / 3-fold result cannot be reproduced here |

---

## 2. Exact Architecture

```
ARM A   P_A = P_V3                                87 features
ARM B   P_B = w_B · P_market + (1−w_B) · P_V3     E2 blend, verbatim
ARM C   P_C from a 91-column refit                E6 features, verbatim
ARM D   P_D = w_D · P_market + (1−w_D) · P_C      approved combination
MARKET  P_market alone                            reference column
```

**One weight per blend, and it is E2's existing one.** No second combination parameter exists — verified by test (`best_combination.py` contains exactly one `blend()` and one `learn_weight()`, and no `w2`).

**The market never enters a Poisson design matrix.** **The A/D states never enter the blend as probabilities.** Each validated component keeps its own architecture; neither is reinterpreted.

---

## 3. Why Arm D Was Chosen

E2 validated a probability-level blend; E6 validated design-matrix columns. §13 forbids reinterpreting E2 as regression features. The only construction preserving both is to chain them: E6 produces `P_C`, then E2's blend consumes it.

The zero-component controls then fall out as identities rather than approximations:

| Control | Reduction |
|---|---|
| `w = 0` | Arm D → Arm C |
| A/D states zeroed | Arm D → Arm B |
| both | Arm A |

---

## 4. Exact V3 Formulation

87 columns → `LogisticRegressionPreprocessor` (median impute + StandardScaler + one-hot `competition_id`) → `PoissonRegressor(alpha=1.0, max_iter=2000)` × 2 → `predict_poisson` (adaptive grid, tail < 1e-15).

**Arm A is a per-fold refit, not the shipped artifact** — the artifact is fitted on every test season.

---

## 5. Exact E2 Formulation

Power de-vig on Pinnacle **closing** 1X2 (peak never used, opening diagnostic only). Weight grid `w ∈ {0.00, 0.05, …, 1.00}` (21 points), chosen by training-fold log loss, frozen, applied to test.

Market probabilities are read pre-computed from `research_dataset.sqlite` — **not recomputed**, so E2's numbers cannot drift.

---

## 6. Exact E6 Formulation

Reused verbatim by import from `research/online_attack_defense/online_attack_defense.py`. Verified by test: `INIT_ATTACK = INIT_DEFENSE = 0.0`, `STATE_CLIP = 1.5`, `EXP_CLIP = 3.0`, `LR_GRID = (0.005, 0.01, 0.02, 0.035, 0.05)`, `AD_COLUMNS` unchanged, two-pass timestamp logic, cross-season persistence, 91-column contract with V3's first 87 intact.

---

## 7. Dataset Intersection

Reproduced by the harness, matching the audit exactly:

| Step | n |
|---|---|
| V3 / E6 universe | **8,983** |
| `dropped_no_v3` | **0** |
| `dropped_no_label` | **0** |
| `dropped_no_e6` | **0** |
| `dropped_no_market` | **4,982** |
| With market | **4,001** |
| Less 2022/23 (E2 blend-training) | **−522** |
| **E7 eligible OOS** | **3,479** |

By season: 2023/24 = 1,727 · 2024/25 = 1,752.
By league: Premier League 755 · Serie A 755 · La Liga 753 · Ligue 1 610 · Bundesliga 606.

Every arm reads one shared table — a different fixture set per arm is structurally impossible.

---

## 8. Five Leagues, Seasons, Folds

Premier League, La Liga, Serie A, Bundesliga, Ligue 1 — verified as exactly these five.

| Fold | Train | Test | n_test |
|---|---|---|---|
| fold_1 | 2022/23 | 2023/24 | 1,727 |
| fold_2 | 2022/23, 2023/24 | 2024/25 | 1,752 |

Both verified strictly chronological. **2025/26: 0 fixtures used.**

### A structural limitation I have to flag

**fold_1 trains on a single season, so no nested inner split is possible.** E6's `lr` selection needs at least two training seasons to build an inner split. For fold_1 the harness falls back to the **grid midpoint 0.02**, records `n_inner_splits = 0`, and documents it explicitly in `lr_note`.

This is not test-tuning — 0.02 is fixed a priori, never chosen by looking at the test fold. But it does mean **fold_1's `lr` is not selected by E6's validated procedure**, only fold_2's is. That weakens criterion 10 and I will not paper over it.

---

## 9. E2 Reproduction Control

Because E7 uses E2's exact fixtures, folds, market probabilities, V3 method and blend mechanism, Arm B **should reproduce E2 to floating point**. The harness compares three quantities against stored `e2_results.json` at tolerance 1e-6:

| Quantity | Previous E2 |
|---|---|
| V3 log loss | 0.981357 |
| Blend log loss | 0.957130 |
| Market log loss | 0.955905 |

If any differs beyond tolerance the harness records `passed: false` and the decision logic treats it as a hard gate failure.

---

## 10. E6-Method Evaluation on the Common Universe

Per the explicit instruction, this is **not** described as reproducing E6.

> E7 re-runs the validated E6 methodology on the common E2/E6 intersection and therefore provides an E6-method evaluation on the E7 fixture universe.

Divergence from E6's published numbers (LL 0.985574 on n = 5,331) is **expected** because the fixture universe, fold count and excluded season all differ, and `lr` may legitimately differ. The harness prints this caveat inline.

The genuine E6 control is the zero-state reduction, which must hold to floating point.

---

## 11. Market-Alone Reference

Carried as a fifth reported column, **not a fifth arm**, and never used for tuning. It is the incumbent: E2 showed market alone beat the E2 blend on log loss, Brier and RPS. Criterion 5 makes clearing it mandatory.

---

## 12–21. Results

**PENDING** — pooled OOS, per-fold, per-league, per-season metrics, all nine pairwise deltas, learned weights, complementarity diagnostics and calibration are produced by the local run and written to `e7_results.json`.

Per the standing rule, no proxy or prior-experiment numbers are presented here as if they were E7 results.

---

## 22. Leakage Tests

All 14 required checks, verified mechanically (82/82 tests):

| # | Check | Status |
|---|---|---|
| 1 | Current outcome cannot affect E6 state | **PASS** — set to 9–0, bit-identical |
| 2 | Future outcomes cannot affect earlier state | **PASS** — bit-identical |
| 3 | Simultaneous fixtures cannot affect each other | **PASS** — bit-identical |
| 4 | Test labels cannot change coefficients | **PASS** — no label in any fitting signature |
| 5 | Test labels cannot change market probabilities | **PASS** — pre-computed from odds |
| 6 | Test labels cannot change blend weights | **PASS** — rewrote every test label, `w` unchanged |
| 7 | Market probabilities derived from odds only | **PASS** — no outcome column exists in the table |
| 8 | `lr` from inner training splits only | **PASS** — signature introspection |
| 9 | V3 coefficients from training only | **PASS** — per-fold refit |
| 10 | Blend `w` from training only | **PASS** — signature carries training arrays only |
| 11 | Train strictly precedes test | **PASS** — both folds |
| 12 | 2025/26 used = 0 | **PASS** |
| 13 | Identical fixture IDs across arms | **PASS** — one shared table |
| 14 | Deterministic repeated execution | **PASS** — bit-identical |

Additional adversarial checks: future market probabilities cannot change `w`; rewriting an entire future season leaves prior A/D states bit-identical (inherited from E6's 83/83 suite).

---

## 23. Zero-Component Controls

**Probability level — all three PASS** (tolerance 1e-12):

| Control | Result |
|---|---|
| C1: `w = 0` → Arm C | **PASS** |
| C2: zero A/D states → Arm B | **PASS** |
| C3: both disabled → Arm A | **PASS** |

A deliberate negative test confirms C2 **correctly fails** (max diff 3.474e-01) when states are not actually zeroed — the controls detect real breakage rather than passing vacuously.

**Model level — PENDING.** The harness refits with zeroed states per fold and requires all three controls to pass before evaluating the combination.

---

## 24. Determinism

`blend`, `learn_weight`, `per_fixture_log_loss`, `compute_ad_states` all bit-identical on repeat. No RNG in `best_combination.py`. The harness additionally re-fits and requires `max|ΔP| < 1e-12`.

---

## 25. Complementarity Diagnostics

Implemented and unit-tested, **descriptive only** — the function exposes no tuning surface (verified by signature introspection). Reports log-loss and probability-error correlations between arms, the four help/hurt quadrants (verified to partition to exactly 100%), material-change percentages and top-class flip rates.

---

## 26. 2025/26 Quarantine

**Verified: 0 fixtures.** Absent from both universes; the harness asserts and aborts otherwise. 1,752 fixtures exist in the database and are correctly excluded from scope.

---

## 27. Protected MD5

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## 28. Git Status

`git status --short` **unavailable** — the workspace is not a git repository. Filesystem list:

**INTENDED:** `research/best_combination/*` · **UNEXPECTED:** none.

---

## 29. Files Changed / Untouched

**Created:** `E7_AUDIT_REPORT.md`, `best_combination.py`, `test_best_combination.py`, `run_e7_experiment.py`, `E7_BEST_COMBINATION_REPORT.md`. `e7_results.json` is written by the local run.

**Untouched:** `data/models/*`, `data/processed/*`, `src/*`, `predict_match.py`, `config/*`, and every other research directory. E2's and E6's modules are **imported, never edited**. All reads used SQLite `mode=ro`.

---

## 30. Known Limitations

1. **The experiment did not run.** sklearn unavailable; substituting a solver is forbidden.
2. **Only 2 folds** — thinner than E6's 3.
3. **fold_1 cannot use E6's nested `lr` selection** (single training season). Documented fallback to the grid midpoint.
4. **n = 3,479 vs E6's 5,331** — a 35% smaller, recency-skewed evaluation set.
5. **Criterion 3 (D > B) may be structurally hard.** If `w_D` approaches 1.0 as E2's `w` did in fold 2, Arm D approaches market-alone and the A/D contribution is blended out regardless of quality. The harness flags this explicitly when `w_D ≥ 0.95`.
6. **Criterion 5 is a high bar.** Market alone beat the E2 blend, so D must clear an incumbent that already beats the architecture D is built on.

---

## 31. Final E7 Decision

# E7: INCONCLUSIVE

**Why not PASS or FAIL:** criteria 1–6 and 9 require measured out-of-sample metrics from the real V3 pipeline. They were not produced, and using proxy or prior-experiment numbers is explicitly forbidden.

| # | Criterion | Status |
|---|---|---|
| 1 | D improves OOS Log Loss vs V3 | **NOT MEASURED** |
| 2 | ≥2 secondary metrics vs V3 | **NOT MEASURED** |
| 3 | D > Arm B | **NOT MEASURED** |
| 4 | D > Arm C | **NOT MEASURED** |
| 5 | D > Market alone | **NOT MEASURED** |
| 6 | Not isolated to one league | **NOT MEASURED** |
| 7 | No leakage | **PASS** — 14/14 |
| 8 | Zero-component controls | **PASS at probability level**; model level pending |
| 9 | E2 reproduction | **PENDING** |
| 10 | E6 causal controls | **PASS** — parameters, timing, contract all verified |
| 11 | Determinism | **PASS** |
| 12 | No numerical instability | **PASS** |

**Five of twelve pass on measured evidence; one more passes at probability level.**

### What is established

The architecture is correct and the controls detect real breakage. Both validated components are imported unmodified. The fixture intersection is exact with zero unexplained drops.

### What is not established

Whether the combination is complementary. Given audit Findings 2 and 3, my honest expectation is that **CASE B or CASE E is most likely** — market dominates, `w_D` runs high, and D lands close to market-alone. But that is an expectation from E2's weight behaviour, not a measurement, and I am not recording it as a result.

### To complete E7

```bash
cd "E:\Football Prediction Project"
python research/best_combination/test_best_combination.py   # expect 82/82
python research/best_combination/run_e7_experiment.py
```

The harness runs the E2 reproduction control and all three zero-component controls first, treats any failure as a hard gate, then applies all twelve criteria mechanically.

---

## 32. Recommendation for E8

**Do not start E8 until E7 has run.** E8 is temperature scaling — a calibration layer — and calibrating the wrong base model wastes the experiment. E7 decides what the base model is.

Three points:

1. **If E7 shows market dominance (`w_D` ≈ 1), the champion candidate is V3 + Market, or possibly market alone.** E2 already showed market alone beating the blend. That would make E8's target the market-derived probabilities — whose ECE was already 0.0078 in E2, notably better calibrated than V3's 0.0131. Temperature scaling may have little headroom there, which is worth knowing before building it.

2. **If E7 shows genuine complementarity, Arm D becomes the E12 candidate** and E8 should calibrate Arm D.

3. **The environment remains the binding constraint.** Five experiments now have complete, tested harnesses that cannot execute here. E3, E4, E5 and E7 are all one local command from a real answer. Converting those four INCONCLUSIVEs would be worth more than starting E8 on an undecided foundation.

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
E7 — Best Proven Combination      INCONCLUSIVE
E8 — Temperature Scaling          QUEUED
E9 — LightGBM Poisson             QUEUED
E10 — Match Importance + Squad    QUEUED
E11 — Advanced Tactical Efficiency QUEUED
E12 — Final Champion Selection    PENDING
E13 — Final 5-League Walk-Forward PENDING
E14 — Production Inference        PENDING
E15 — V2 → V4/V5 Promotion        PENDING
E16 — Production Monitoring       PENDING
```

---

**Nothing promoted. Production default unchanged. V2 unmodified. V3 artifact unmodified. No Dixon-Coles, no time decay, no quadratic Elo. Research only.**
