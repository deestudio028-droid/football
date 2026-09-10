# E8 — Temperature Scaling Controlled Experiment Report

**Date:** 2026-08-20
**Type:** Research experiment only — nothing promoted, no production change
**Question:** Does a single globally learned temperature, selected causally, improve the incumbent probability distribution out-of-sample?

---

## EXECUTION STATUS

**E8 RAN IN FULL.** Unlike E3–E7, the base is a stored probability dataset requiring no model fitting, so no scikit-learn was needed. Every number below is measured.

All 16 execution steps completed in the specified order. **No integrity or leakage gate failed.**

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
| **E8 — Temperature Scaling** | **THIS REPORT** |

---

## 2. E7 Context and the E2 Reproduction Audit

### The discrepancy — resolved, CASE B

| Quantity | E2 | E7 | Diff | Produced how |
|---|---|---|---|---|
| **Market** | 0.955905 | 0.955905 | **0.000000** | Stored file — no fitting |
| **V3** | 0.981357 | 0.990435 | **+0.009078** | Artifact vs per-fold refit |
| **Blend** | 0.957130 | 0.967628 | **+0.010498** | Inherits V3 |

**Root cause, located precisely.** `run_e2_experiment.py:128` loaded the shipped V3 artifact:

```python
def load_v3_predictions(fixture_ids):
    """Uses the frozen V3 candidate artifact..."""
    art = load_v3_artifact(path)
```

And from `src/models/config.py`:

```
artifact FINAL_TRAIN_SEASONS: ('2020/2021','2021/2022','2022/2023','2023/2024','2024/2025')
E2 OOS test seasons:          2023/2024, 2024/2025
OVERLAP:                      ['2023/2024', '2024/2025']
```

**The artifact was fitted on 100% of the fixtures E2 evaluated it on.** E2's V3 figure is an in-sample number.

**Ruled out as a bug** by three independent checks: both fitted quantities moved the *same* direction (a bug has no reason to be uniformly one-signed); +0.009 is the right magnitude for in-sample optimism; and the blend gap follows arithmetically, since both experiments learned `w = 1.00` in fold 2 (blend = pure market, V3 irrelevant there), leaving fold 1 to carry the gap — E2 `w = 0.85` vs E7 `w = 0.30`, i.e. E7's optimiser correctly leaned harder on the market once V3 stopped being flattered.

**Classification: CASE B — explained and proven harmless**, with a correction to the record:

- **E7's numbers are correct.** V3's honest OOS figure is 0.990435.
- **E2's V3 and Blend are in-sample contaminated** and must not be used as references.
- **E2's Market (0.955905) is unaffected** — no fitting involved.
- **E2's PASS direction is unaffected.** Correcting V3 downward makes "market beats V3" stronger.

`e2_results.json` was **not** modified. The historical record stands; this report documents the correction.

**E7's FAIL verdict stands** on its own merits independently of the gate: `w_D` reached 1.00 and Arm D (0.970214) lost to market alone (0.955905), failing mandatory criterion 5.

### Verified in this experiment

E8's base reproduced E2's market figure to **diff = 0.000e+00**.

---

## 3. Calibration Infrastructure Audit

`src/models/calibration.py` contains `compute_ece`, `compute_reliability_bins`, `IdentityCalibrator`, `MulticlassPlattCalibrator`, `MulticlassIsotonicCalibrator`. **Temperature scaling does not exist.** The files named in the spec (`run_calibrated.py`, `run_matrix_calibration.py`, `calibrate_temperature.py`) **do not exist** in this repository.

Platt and isotonic are excluded by §1, so nothing existing was reused. E8 implements its own ECE inside `research/` rather than importing from `src/`, preserving isolation. No shared utility was modified.

---

## 4. Base Model Selection

Full E7 pooled OOS (n = 3,479):

| Arm | Log Loss |
|---|---|
| A — V3 | 0.990435 |
| B — V3 + Market | 0.967628 |
| C — V3 + Online A/D | 0.991780 |
| D — V3 + A/D + Market | 0.970214 |
| **MARKET alone** | **0.955905** |

> **PRIMARY BASE = MARKET ALONE** (de-vigged Pinnacle closing 1X2)

Because it is the strongest incumbent, the only exactly-reproducible quantity, immune to the leakage that contaminated E2's V3/Blend, and requires no fitting — so temperature is tested in isolation.

**V3 secondary diagnostic was not run** (requires sklearn, unavailable). It could not have altered the decision.

---

## 5. Formula, Grid, Selection Protocol

```
P'_k = P_k^(1/T) / Σ_j P_j^(1/T)
```

Computed in log space with max-subtraction. Verified to match the direct power form to **< 1e-12** at every grid point.

**Grid, frozen before any result was seen:** `{0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20}` — never expanded.

**Epsilon, documented:** probabilities clipped to `1e-300` before `log`. It never binds (observed minimum ~1e-3, ~297 orders of magnitude above). `T = 1.0` short-circuits and returns an unchanged copy, so the identity is **exact**, not merely within tolerance.

**Selection:** training-only, per fold. Fold 2 used nested inner validation (1 split). **Fold 1 could not** — it trains on a single season, so no inner split is constructible; T was selected on the outer training period directly (2022/23 outcomes only, never test). Training-only and therefore not leakage, but *not* nested, and reported as such.

---

## 6–8. Dataset, Leagues, Seasons, Folds

| Step | n |
|---|---|
| Market fixtures | 4,001 |
| `dropped_no_label` | **0** |
| `dropped_2025_26` | **0** |
| Final | 4,001 |
| **Eligible OOS** | **3,479** |

| Fold | Train | Test | n_train | n_test |
|---|---|---|---|---|
| fold_1 | 2022/23 | 2023/24 | 522 | 1,727 |
| fold_2 | 2022/23, 2023/24 | 2024/25 | 2,249 | 1,752 |

Five leagues verified. **2025/26: 0 fixtures.** Both folds strictly chronological.

---

## 9. Selected T Per Fold

| Fold | Test | Selected T | Method | Direction |
|---|---|---|---|---|
| fold_1 | 2023/24 | **1.10** | outer-train direct | soften |
| fold_2 | 2024/25 | **0.90** | nested (1 split) | sharpen |

**The two folds selected opposite directions.** Fold 1 wanted softening, fold 2 wanted sharpening. Spread 0.20 — two full grid steps — classified **UNSTABLE**.

Neither selection is at a grid boundary, so the grid was wide enough; the instability is real, not an artifact of grid width.

---

## 10. Pooled OOS Results

n = 3,479, identical fixtures both arms.

| Arm | Log Loss | Brier | RPS | ECE | MaxCE | Accuracy | Draw Recall | Mean Conf |
|---|---|---|---|---|---|---|---|---|
| **A base** | **0.955905** | **0.567798** | **0.189968** | **0.007783** | **0.042140** | 0.5519 | 0.0056 | 0.538530 |
| B scaled | 0.957486 | 0.568793 | 0.190481 | 0.011166 | 0.055732 | 0.5519 | 0.0056 | 0.540228 |

**Deltas (positive = temperature better):**

| Metric | Delta |
|---|---|
| Log Loss | **−0.00158100** |
| Brier | **−0.00099500** |
| RPS | **−0.00051300** |
| ECE | **−0.00338300** |

**Temperature scaling made the incumbent worse on all four metrics.** Per §27, these are the exact figures — a log-loss degradation of 0.0016, roughly 0.17% of the base. Small, but consistently negative and in the wrong direction on every metric including the one calibration exists to improve.

For scale: the fold-to-fold spread in base log loss is 0.952502 → 0.959258, a range of 0.0068. The degradation (0.0016) is about a quarter of that natural variation.

---

## 11. Per-Fold Results

| Fold | T | Base LL | Scaled LL | Δ LL | Δ Brier | Δ RPS | Δ ECE |
|---|---|---|---|---|---|---|---|
| fold_1 | 1.10 | 0.952502 | 0.955050 | **−0.002548** | −0.001443 | −0.000854 | −0.003513 |
| fold_2 | 0.90 | 0.959258 | 0.959887 | **−0.000629** | −0.000553 | −0.000176 | −0.007243 |

**Both folds degraded on every metric.** The temperature learned on training data did not transfer to the test period in either fold — in opposite directions.

---

## 12. League-Wise Results

| League | n | Base LL | Scaled LL | Δ LL | Base ECE | Scaled ECE |
|---|---|---|---|---|---|---|
| Ligue 1 | 610 | 0.983426 | 0.983335 | **+0.000091** | 0.005260 | 0.021487 |
| Serie A | 755 | 0.957654 | 0.957857 | −0.000203 | 0.017016 | 0.012500 |
| La Liga | 753 | 0.945696 | 0.946852 | −0.001156 | 0.018318 | 0.018430 |
| Bundesliga | 606 | 0.967520 | 0.970616 | −0.003096 | 0.028472 | 0.029756 |
| Premier League | 755 | 0.932777 | 0.936297 | −0.003520 | 0.008580 | 0.012204 |

**1 of 5 leagues improved**, by +0.000091 — a margin indistinguishable from noise on 610 fixtures. Note that Ligue 1's *calibration* got four times worse (ECE 0.0053 → 0.0215) while its log loss nominally improved, which is exactly the kind of split §17 warns against reading as success.

---

## 13. Season-Wise Results

| Season | n | Base LL | Scaled LL | Δ |
|---|---|---|---|---|
| 2023/24 | 1,727 | 0.952502 | 0.955050 | −0.002548 |
| 2024/25 | 1,752 | 0.959258 | 0.959887 | −0.000629 |

Both negative.

---

## 14. Calibration Diagnostics and Reliability

Pooled reliability, base vs scaled:

| Bin | n | Base predicted | Base observed | Base gap | Scaled gap |
|---|---|---|---|---|---|
| [0.2, 0.3) | 3,687 | 0.2548 | 0.2571 | +0.0023 | +0.0050 |
| [0.3, 0.4) | 1,675 | 0.3396 | 0.3296 | −0.0100 | −0.0173 |
| [0.4, 0.5) | 1,070 | 0.4465 | 0.4570 | +0.0106 | +0.0072 |
| [0.5, 0.6) | 849 | 0.5484 | 0.5607 | +0.0123 | +0.0378 |
| [0.6, 0.7) | 521 | 0.6468 | 0.6545 | +0.0077 | −0.0174 |
| [0.7, 0.8) | 344 | 0.7446 | 0.7558 | +0.0112 | −0.0054 |
| [0.8, 0.9) | 146 | 0.8396 | 0.8562 | +0.0166 | +0.0032 |
| [0.9, 1.0) | 16 | 0.9171 | 0.8750 | −0.0421 | −0.0557 |

**Calibration got worse, not better.** ECE 0.007783 → 0.011166; max calibration error 0.042140 → 0.055732. The heavily-populated [0.5, 0.6) bin degraded from +0.0123 to +0.0378.

This is the decisive finding and it was foreseeable: the audit flagged that market-alone already had the best ECE measured anywhere in this project (0.0078 vs V3's 0.0131). **Temperature scaling exists to fix miscalibration; applied to a well-calibrated distribution it can only introduce some.**

---

## 15. Probability-Change and Ranking Diagnostics

| Fold | T | Direction | Confirmed | Mean abs ΔP | Top-class changed |
|---|---|---|---|---|---|
| fold_1 | 1.10 | soften | **Yes** | 0.011422 | **0.0%** |
| fold_2 | 0.90 | sharpen | **Yes** | 0.013496 | **0.0%** |

The transform did exactly what the mathematics implies — softening raised entropy at T > 1, sharpening lowered it at T < 1. **Ranking preserved: True.** Top-class predictions changed on **0.0%** of fixtures, so accuracy and draw recall are byte-identical between arms (0.5519 and 0.0056).

**Per §17, the distinction matters:** temperature scaling is incapable of changing classification here, so this is purely a calibration/scoring question. On that question it lost on every measure.

---

## 16. Leakage Tests

All 14 required checks pass (118/118 tests, 0 skipped):

| Check | Status |
|---|---|
| T = 1 identity | **PASS** — exact, 0.000e+00 |
| Test labels cannot alter T | **PASS** — rewrote every test label, T unchanged |
| Future labels cannot alter earlier T | **PASS** |
| Future probabilities cannot alter T | **PASS** |
| Calibration rows strictly earlier than validation | **PASS** |
| Outer-test rows never enter selection | **PASS** — `n_train` verified |
| 2025/26 fixtures = 0 | **PASS** |
| Deterministic output | **PASS** — bit-identical |
| Probabilities normalized | **PASS** — max deviation < 1e-12 |
| Class ranking preserved | **PASS** — all grid points |
| No random state | **PASS** — no RNG in module |
| Same data → same T | **PASS** |
| Changing only test labels leaves T unchanged | **PASS** |
| Changing future outcomes leaves calibration unchanged | **PASS** |

`select_temperature`'s signature is `(y_train, P_train, grid, method)` — no parameter exists through which test data could enter.

**Scope discipline verified by AST**, not string matching: the module imports only `numpy`, `dataclasses`, `__future__`. No sklearn, scipy, lightgbm, torch, models or features. No isotonic, Platt, PAVA or sigmoid calibrator defined. Vector temperatures rejected — single global T only.

---

## 17. Determinism

Probabilities identical on repeat: **True**. `apply_temperature`, `select_temperature`, `reliability_bins` and `ece` all bit-identical. No RNG in the module.

---

## 18. T = 1 Identity Control

| Check | Result |
|---|---|
| max \|P − P_T1\| | **0.000e+00** (exact) |
| Probabilities identical | **True** |
| Predictions identical | **True** |
| Log Loss / Brier / RPS / ECE / Accuracy identical | **True** |

Verified on both synthetic and the **real 4,001-fixture market distribution**. Achieved by an explicit short-circuit at `T == 1.0` rather than relying on floating-point round-trip.

---

## 19. Protected MD5

Verified before and after:

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## 20. Git Status / Files

`git status --short` unavailable — not a git repository.

**INTENDED:** `research/temperature_scaling/*` · **UNEXPECTED:** none.

**Created:** `E8_AUDIT_REPORT.md`, `temperature_scaling.py`, `test_temperature_scaling.py`, `run_e8_experiment.py`, `e8_results.json`, `E8_TEMPERATURE_SCALING_REPORT.md`.

**Untouched:** `src/*` (including `calibration.py`), `data/*`, `predict_match.py`, `config/*`, and every other research directory. `e2_results.json` deliberately not modified.

### Note on a test I fixed

One test initially failed: a naive string scan for `"sklearn"` tripped on the module's own docstring line *"No scipy, no sklearn."* I replaced the whole scan with **AST-based import analysis** — checking what the module actually imports and what callables it defines, rather than what its prose mentions. Same class of mistake I made in E7; the fix is more robust than the original in both cases. Deleting the assertion to go green would have been wrong.

---

## 21. Known Limitations

1. **Only 2 folds**, and fold 1 could not use nested inner validation (single training season).
2. **Fold 1's T rests on 522 training fixtures** — thin.
3. **V3 secondary diagnostic not run** (sklearn unavailable). Non-decisive by design.
4. **n = 3,479**, recency-skewed to 2023/24–2024/25.
5. **The base was already well calibrated**, so the experiment had little headroom by construction. This was flagged in the audit *before* running.

---

## 22. Final E8 Decision

# E8: FAIL

**Criterion 1 is mandatory and was not met: temperature scaling made the incumbent worse on the primary metric (−0.00158100).**

| # | Criterion | Result |
|---|---|---|
| 1 | Log Loss improves | **FAIL** — −0.00158100 |
| 2 | ≥2 of Brier/RPS/ECE improve | **FAIL** — 0/3 |
| 3 | Not isolated to one league | **FAIL** — 1/5, and that one by +0.00009 |
| 4 | No leakage | **PASS** |
| 5 | T stable across folds | **FAIL** — 1.10 vs 0.90, opposite directions |
| 6 | T = 1 identity control | **PASS** — exact |
| 7 | No numerical instability | **PASS** |
| 8 | 2025/26 quarantine | **PASS** |
| 9 | Protected files byte-identical | **PASS** |
| 10 | E2 reproduction resolved | **PASS** — CASE B, documented |

Six criteria pass; four fail. **The four failures are the substantive ones.**

### What this means

The result is clean and interpretable, not a defect:

- The market's de-vigged closing probabilities are **already well calibrated** (ECE 0.0078, the best measured in this project). There is nothing for a temperature to fix.
- The two folds wanted **opposite corrections** — soften then sharpen — which is the signature of fitting noise rather than a stable miscalibration.
- Because ranking is preserved and 0.0% of top-class predictions changed, temperature could only ever have helped via calibration, and calibration got **worse** (ECE 0.0078 → 0.0112).

Per §30: *"If T improves ECE but worsens Log Loss: do not PASS."* Here it worsened both.

**The correct action is to keep the uncalibrated market probabilities.**

Per §27, no overselling in either direction: the degradation is small (0.0016 log loss, ~0.17%), roughly a quarter of the natural fold-to-fold spread. It is not catastrophic. It is simply and consistently negative, on every metric, in both folds, in four of five leagues.

---

## 23. Recommendation for E9

**Proceed to E9 (LightGBM Poisson).** E8's failure is self-contained and blocks nothing.

Three points:

1. **Stop calibrating the market; try to beat it.** E8 confirms the market signal is well calibrated as well as accurate. Post-hoc adjustment of it is a closed question — three separate attempts to add to or adjust the market (E7's blend, E7's Arm D, E8's temperature) have now all failed or come out worse. E9 should aim at a genuinely better *base model*, not another wrapper.

2. **Apply the E7 lesson to E9's baseline immediately.** The V3 artifact is fitted on 2020/21–2024/25 and **must never be used to score those seasons**. E9's V3 baseline must be a per-fold refit. E2 got this wrong and it took until E8 to surface; E9 should not repeat it. The honest V3 reference is **0.990435**, not 0.981357.

3. **Set expectations against the real bar.** The number to beat is **0.955905** (market alone). No fitted model in this project has come close — V3 is 0.990435, V3+A/D is 0.991780, the best blend is 0.967628. LightGBM would need roughly a 3.5% log-loss improvement over V3 just to match the market. That is worth attempting, but E9 should be evaluated against 0.955905 rather than against V3, or it risks declaring success while remaining behind the incumbent.

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
E9  — LightGBM Poisson            NEXT
E10 — Match Importance + Squad    QUEUED
E11 — Advanced Tactical Efficiency QUEUED
E12 — Final Champion Selection    PENDING
E13 — Final 5-League Walk-Forward PENDING
E14 — Production Inference        PENDING
E15 — V2 → V4/V5 Promotion        PENDING
E16 — Production Monitoring       PENDING
```

---

**Nothing promoted. Production default unchanged. V2 unmodified. V3 artifact unmodified. No isotonic, no Platt, no per-class or per-league temperature. Research only.**
