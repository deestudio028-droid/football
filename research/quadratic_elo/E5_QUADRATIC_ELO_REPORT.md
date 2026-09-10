# E5 — Quadratic Elo → xG Controlled Experiment Report

**Date:** 2026-08-20
**Type:** Research experiment only — nothing promoted, no production change
**Question:** Does a nonlinear Elo→expected-goals relationship produce genuine out-of-sample improvement?

---

## EXECUTION STATUS — READ FIRST

| Component | Status |
|---|---|
| V3 Elo-representation audit | **COMPLETE** |
| `quadratic_elo.py` implementation | **COMPLETE** |
| Test suite | **COMPLETE — 66/66 passed, 0 skipped** |
| Non-parametric curvature measurement | **COMPLETE — real data** |
| Elo-only GLM walk-forward | **COMPLETE — real data** |
| **Experiment A (V3, 87 features)** | **PENDING — requires local run** |
| **Experiment B (V3 + elo_diff_sq, 88)** | **PENDING — depends on A** |
| **Zero-quadratic control, model level (§18)** | **PENDING — requires local run** |

**Reason:** both arms require refitting `PoissonRegressor` per fold at 87 and 88 columns, which needs scikit-learn. The sandbox has none and cannot install it (PyPI blocked, HTTP 403).

Unlike E4, a substantial part of E5's hypothesis **was** testable here. The curvature question is a property of the data, and an Elo-only GLM fitted by IRLS (validated against known synthetic coefficients) answers it directly. Those results are real measurements and are reported below.

---

## 1. Roadmap Status

| Experiment | Status |
|---|---|
| E0 — V2 Baseline | COMPLETE |
| E1 — Causal Elo → V3 | PASS |
| E2 — Market Odds | PASS |
| E3 — Dixon-Coles | INCONCLUSIVE |
| E4 — Exponential Time Decay | INCONCLUSIVE |
| **E5 — Quadratic Elo → xG** | **THIS REPORT** |

---

## 2. Audit Findings

Three findings materially shaped the implementation.

### 2.1 `elo_diff` is an exact linear combination — verified

```
elo_diff == home_elo + 100 - away_elo      max|residual| = 0.000e+00
```

Not approximately. Exactly, on all 10,735 fixtures.

**Consequence:** `rank(intercept, home_elo, away_elo, elo_diff) = 3 of 4`. V3's Elo block is **rank-deficient** — `elo_diff` adds no independent direction beyond `home_elo`, `away_elo` and the intercept. Ridge (alpha=1.0) makes the fit identifiable, but the three Elo coefficients are **not individually interpretable**; only their combination is.

This matters for §14: a reported `c1` on `elo_diff` in V3 is one of infinitely many equivalent splits across three collinear columns. The **quadratic** term, by contrast, is genuinely new information — `elo_diff²` is not in the span of the existing features.

### 2.2 The `HOME_ADVANTAGE = 100` constant contributes nothing as a feature

It is the same +100 for every fixture, so as a feature offset it is fully absorbed by the intercept. It does real work inside the Elo **update** rule (the expected-score calculation), but adds zero information to the design matrix.

**§7 resolution:** there is no home-advantage double-counting to fix, and no need to re-estimate HA from training data. V3's four venue features carry the actual venue signal; the +100 is inert. The existing representation is kept, unchanged.

### 2.3 Raw `elo_diff²` is badly collinear with `elo_diff` — centering is required

| Form | corr with `elo_diff` |
|---|---|
| Raw `elo_diff²` | **+0.6680** |
| Centered `(elo_diff − mean)²` | **−0.0006** |

Under ridge, a 0.668 correlation would distort both coefficients and confound the curvature test. Centering makes the quadratic term effectively orthogonal to the linear one.

**Centering uses the training-fold mean only**, frozen and applied to the test fold — the same causality rule as the preprocessor's median and scale.

---

## 3. Exact V3 Formulation

`PoissonRegressor` is a GLM with a log link, so V3 is:

```
log(lambda_H) = a_H + Σ_{j=1..87} b_Hj · x_j
log(lambda_A) = a_A + Σ_{j=1..87} b_Aj · x_j
```

with `x_85 = home_elo`, `x_86 = away_elo`, `x_87 = elo_diff`. V3 is therefore **linear in Elo** on the log scale (§3 item 6: confirmed).

| Item | Finding |
|---|---|
| Model | `PoissonRegressor(alpha=1.0, max_iter=2000)` × 2 |
| Preprocessor | median impute + StandardScaler on 86 numeric cols, one-hot `competition_id` |
| Elo params | init 1500, K=20, HA=100, mean-reversion 0 — **unchanged from E1** |
| Conversion | `predict_poisson`, adaptive grid K, tail < 1e-15 |
| Folds | 3 walk-forward, expanding window |

---

## 4. Exact E5 Formulation

```
log(lambda_H) = a_H + Σ b_Hj·x_j + c2_H · (elo_diff − mean_train)²
log(lambda_A) = a_A + Σ b_Aj·x_j + c2_A · (elo_diff − mean_train)²
```

Both goal models receive the same column with their own free coefficient, so the term is fitted consistently for home and away (§5 requirement).

**Option A was chosen over Option B.** Since the 87 features already enter `log(lambda)` linearly, appending one column places the quadratic term exactly where §5 requires. Option B — a bespoke Elo→goal-rate parameterisation — would *replace* V3's architecture rather than extend it, making A and B structurally different and the comparison unfair.

---

## 5. Feature Contract

| Contract | Columns |
|---|---|
| V3 | **87** |
| E5 | **88** = V3[0:87] + `elo_diff_sq` |

Verified by test: `E5[:87] == V3` in the same order; exactly one column added; no duplicates; the 87 pass-through columns are bit-identical.

---

## 6. Elo Definition

The E1 causal engine, **unchanged**. Two-pass per-timestamp: pre-match features extracted for all fixtures at timestamp T before any rating is updated with outcomes at T. Parameters verified by test as init=1500.0, K=20.0, HA=100.0, mean-reversion=0.0.

No change was required, so no STOP condition was triggered.

---

## 7. Home Advantage Handling

See §2.2. The existing representation is preserved: the +100 constant inside `elo_diff` is inert as a feature and is not re-estimated; V3's venue features carry the venue signal. No double-counting exists and none was introduced.

---

## 8. Walk-Forward Protocol

Unchanged from the approved V3/E2 protocol:

| Fold | Train | Test | n_test |
|---|---|---|---|
| fold_1 | 2020/21, 2021/22 | 2022/23 | 1,827 |
| fold_2 | + 2022/23 | 2023/24 | 1,752 |
| fold_3 | + 2023/24 | 2024/25 | 1,752 |

All three verified strictly chronological. **Arm A is a per-fold refit, not the shipped artifact** — the artifact is fitted on 2020/21–2024/25, i.e. on every test fold, so using it would leak all test outcomes. (Same finding as E4 §8.)

---

## 9. Dataset

`matches.db` joined to `features.db`, both read-only. **8,983** eligible fixtures; **5,331** pooled out-of-sample.

---

## 10. Five Leagues

Premier League, La Liga, Serie A, Bundesliga, Ligue 1 — verified as exactly these five.

---

## 11. Seasons

2020/21 – 2024/25. **2025/26 quarantined — 0 fixtures used, verified.**

---

## 12. Fold Sizes

fold_1: 3,652 train / 1,827 test · fold_2: 5,479 / 1,752 · fold_3: 7,231 / 1,752.

---

## 13. Coefficients by Fold

**PENDING** for the 87/88-feature experiment. The harness extracts `c1` (index 86) and `c2` (index 87) from both goal models per fold.

### Measured on the Elo-only GLM (real data, real folds)

| Fold | Test | c1_home | **c2_home** | c1_away | **c2_away** |
|---|---|---|---|---|---|
| 1 | 2022/23 | +0.152679 | **+0.013822** | −0.145558 | **+0.010654** |
| 2 | 2023/24 | +0.159784 | **+0.015802** | −0.138468 | **+0.011400** |
| 3 | 2024/25 | +0.160799 | **+0.016364** | −0.146996 | **+0.019310** |

(Coefficients are on standardised `z = (elo_diff − mean_train)/sd_train`.)

---

## 14. Quadratic Coefficient Analysis

Answering §14's five questions on the measured evidence:

**1. Is c2 consistently non-zero?** Yes. All six estimates are non-zero and comfortably so relative to the linear term (roughly 7–13% of |c1|).

**2. Is its sign stable?** Yes — **positive in all six**, home and away, every fold.

**⚠️ Note the sign.** §Core-Objective warned against assuming `c2 < 0`. The data says the opposite: **c2 > 0**. Both goal rates are *convex* in Elo difference. For the home team that means goals rise at an accelerating rate as superiority grows; for the away team, `c1 < 0` with `c2 > 0` means away goals fall at a *decelerating* rate. This is visible directly in the decile table (§17): away goals drop 2.01 → 1.52 → 1.19 → 1.04 → 0.81 with shrinking steps. The practical reading is that lopsided fixtures produce more total goals than a log-linear model expects.

**3. Is its magnitude stable?** Yes. c2_home spread **1.18×**, c2_away spread **1.81×** — both within a 2× band, the tightest parameter stability seen in E3–E5.

**4. Does the quadratic term materially change predictions?** Barely. The non-parametric fit reduces weighted SSE of the decile log-means by 10.2% (home) and 15.0% (away) — real curvature — but the resulting probability shifts are small.

**5. Does it improve out-of-sample log loss?** **Essentially no.** Pooled delta **+0.000060** — about 0.006% of log loss. One fold of three improves; the other two are marginally worse.

**This is the textbook case §14 warns about.** A training coefficient that is non-zero, sign-stable and magnitude-stable is *not* evidence of usefulness. The curvature is genuinely present in the data and reliably estimated — and it still buys essentially nothing out-of-sample.

---

## 15. Pooled OOS Metrics

**PENDING** for the 87/88 experiment.

### Measured on the Elo-only GLM (n = 5,331, identical fixtures both arms)

| Arm | Log Loss | Brier | RPS |
|---|---|---|---|
| A — linear Elo | 0.996337 | 0.593852 | — |
| B — quadratic Elo | **0.996278** | **0.593774** | — |
| **Delta** | **+0.000060** | **+0.000078** | — |

Per fold: 2022/23 **+0.000273**, 2023/24 **−0.000079**, 2024/25 **−0.000024**.

---

## 16. League-Wise Metrics

**PENDING** for the experiment. Elo-only GLM:

| League | n | Delta log loss |
|---|---|---|
| La Liga | 1,140 | **+0.000441** |
| Premier League | 1,140 | **+0.000350** |
| Bundesliga | 918 | −0.000046 |
| Ligue 1 | 992 | −0.000086 |
| Serie A | 1,141 | **−0.000400** |

**2 of 5 leagues improved.**

### The league-split pattern is noise, not structure

I flagged after E4 that these five leagues keep splitting 2-vs-3 and said a repeat would be worth investigating. It repeated — but the membership changed again:

| Experiment | Leagues that improved |
|---|---|
| E3 (Dixon-Coles) | Bundesliga, Serie A |
| E4 (time decay, proxy) | Premier League, Serie A |
| E5 (quadratic Elo, GLM) | **La Liga, Premier League** |

Three experiments, three different pairs, no league in all three. If a stable per-league property drove this, the same leagues would recur. They do not.

**Revised reading: the 2-of-5 split is what a null effect looks like when sliced five ways.** Each delta is ~1e-4 on samples of ~1,000, so sign is essentially a coin flip. This retires the "per-league parameters" hypothesis I raised in E4 §32 — the evidence does not support it, and I should not have implied it was likely without this check.

---

## 17. Season-Wise Metrics

**PENDING** for the experiment. Per-fold results in §15 are per-season by construction (one test season per fold).

### Non-parametric curvature — the direct measurement

log(mean goals) by `elo_diff` decile, n = 8,983:

| Bin | elo_diff mid | n | mean home goals | mean away goals |
|---|---|---|---|---|
| 1 | −169.0 | 899 | 0.9410 | 2.0067 |
| 2 | −52.7 | 898 | 1.0958 | 1.6581 |
| 3 | 7.0 | 898 | 1.1659 | 1.5200 |
| 4 | 50.7 | 898 | 1.3051 | 1.3886 |
| 5 | 86.2 | 898 | 1.4321 | 1.1882 |
| 6 | 116.5 | 899 | 1.4861 | 1.1735 |
| 7 | 152.4 | 898 | 1.6236 | 1.1514 |
| 8 | 196.0 | 898 | 1.8363 | 1.0367 |
| 9 | 257.1 | 898 | 2.0635 | 0.8920 |
| 10 | 372.6 | 899 | 2.3949 | 0.8109 |

Weighted fits to the decile log-means:

| Target | Linear slope | Quadratic c2 | SSE reduction |
|---|---|---|---|
| log λ_home | +1.860e−03 | **+4.117e−07** | **10.24%** |
| log λ_away | −1.775e−03 | **+5.386e−07** | **14.96%** |

Curvature is real and positive in both — consistent with the GLM coefficients, from a completely independent, model-free estimate.

---

## 18. Curvature Diagnostics

Per §15's requirement, the harness tabulates linear vs quadratic λ across the 1st–99th percentile Elo range using **training-fitted** coefficients only, reporting the ratio at each grid point so "meaningfully bends" can be distinguished from "reproduces the linear fit".

Verified in the test suite: over the full observed range [−430, +690], the quadratic curve stays within **[0.72, 5.18]** goals — plausible, no implausible extremes.

Per §15's caution, visual bending was **not** used as a PASS criterion; only OOS metrics feed the decision.

---

## 19. Extreme-Value Diagnostics

**Observed `elo_diff` range: [−427.41, +686.69]**, median +100.25 (sitting at the HA offset, as expected since the raw rating difference centres near zero).

`check_extreme_safety` verifies λ_home > 0, λ_away > 0, all finite, and no value exceeding 15.0. Verified to correctly reject exploding, non-positive and non-finite λ. The harness runs it for both arms in every fold and **fails E5 outright** if it trips.

---

## 20. Leakage Tests

All ten §17 checks:

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Fixture outcome cannot affect its own Elo | **PASS** | E1 two-pass causal engine, parameters verified unchanged |
| 2 | Future fixtures cannot affect earlier Elo | **PASS** | Same engine; validated bit-identical in E1 |
| 3 | Changing future outcomes can't change earlier predictions | **PASS** | Elo is pre-match; centering is training-only |
| 4 | Changing test labels can't change coefficients | **PASS** | No fitting function accepts a label (introspection) |
| 5 | Elo generated strictly before the fixture | **PASS** | E1 invariant |
| 6 | Quadratic term uses only pre-match Elo | **PASS** | `quadratic_elo.py` never references goals or labels |
| 7 | Coefficients learned from training data only | **PASS** | Per-fold refit; centering from training rows |
| 8 | Test labels never influence fitting | **PASS** | Same |
| 9 | 2025/26 fixtures used = 0 | **PASS** | Query returns 0; asserted in harness |
| 10 | Deterministic predictions | **PASS** | Bit-identical, max diff 0.0 |

---

## 21. Zero-Quadratic Control (§18)

**Feature level — PASS (verified here):** `enabled=False` returns exactly zeros; the column has exactly zero variance; the 87 V3 columns pass through untouched.

**Model level — PENDING.** The harness fits fold_1 at 87 columns and at 88 columns with an all-zero quadratic column, requiring `max|Δλ| < 1e-6` and `max|ΔP| < 1e-6`. **If the control fails the harness stops** and does not run the comparison, exactly as §18 requires.

---

## 22. Determinism

| Check | Result |
|---|---|
| `fit_centering` repeated | identical |
| `make_quadratic_feature` repeated | **bit-identical** (max diff 0.0) |
| `build_design` repeated | identical |
| RNG in `quadratic_elo.py` | **none** — all functions pure |
| IRLS diagnostic | deterministic (no RNG in the solver) |

The harness additionally re-fits and requires `max|ΔP| < 1e-12`.

**IRLS validation:** the diagnostic's solver recovers known synthetic coefficients `[0.5, 0.8, −0.15]` as `[0.5029, 0.8005, −0.1506]`, max error 0.0029 — so the measured c2 values are trustworthy, not solver artefacts.

---

## 23. 2025/26 Quarantine

**Verified: 0 fixtures.** Query on the eligible scope returns zero; the harness asserts it and aborts otherwise. No 2025/26 outcome entered training, validation, coefficient estimation, or any decision-influencing diagnostic.

---

## 24. Protected MD5

Verified before and after all E5 work:

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## 25. Git Status

`git status --short` **unavailable** — the workspace is not a git repository. Filesystem-level list instead.

**INTENDED:** `research/quadratic_elo/*` · **UNEXPECTED:** none.

---

## 26. Files Changed

All created under `research/quadratic_elo/`:

| File | Purpose |
|---|---|
| `quadratic_elo.py` | Centering, `elo_diff_sq`, 88-col contract, curvature/safety diagnostics |
| `test_quadratic_elo.py` | 9 suites, 66 tests |
| `run_e5_experiment.py` | Full A/B harness with zero-quadratic control (needs sklearn) |
| `run_e5_diagnostic.py` | Audit + curvature + IRLS GLM walk-forward (no sklearn) |
| `e5_results.json` | Machine-readable results |
| `E5_QUADRATIC_ELO_REPORT.md` | This report |

---

## 27. Files Untouched

`data/models/*`, `data/processed/*`, `src/models/*`, `src/features/*`, `predict_match.py`, `research/market_odds/*`, `research/dixon_coles/*`, `research/time_decay/*`, `research/worldcup_elo/*`, `config/*`. All reads used SQLite `mode=ro` or read-only access.

---

## 28. Known Limitations

1. **A and B not executed here.** scikit-learn unavailable; both arms need a per-fold refit.
2. **The GLM diagnostic uses Elo alone.** V3 has 87 features. Its 84 non-Elo features may already capture some of this curvature, which would make V3's incremental gain **smaller** than the diagnostic's, not larger.
3. **Only 3 folds** — the standing structural limit across E3–E5.
4. **Elo block is rank-deficient**, so V3's individual Elo coefficients are not interpretable; only the quadratic term is cleanly identified.
5. **Centering is refitted per fold**, so `elo_diff_sq` is not strictly comparable across folds in absolute units (coefficients are, on the standardised scale).
6. **Decile binning is descriptive.** It ignores all other covariates, so it measures the *marginal* Elo→goals relationship, not the conditional one V3 fits.

---

## 29. Final E5 Decision

# E5: INCONCLUSIVE

**Why not PASS or FAIL:** criteria 1–3 and 6 require measured A vs B metrics at 87 and 88 features, plus a model-level zero-quadratic control. None were produced.

| # | Criterion | Status |
|---|---|---|
| 1 | Log loss improves | **NOT MEASURED** |
| 2 | ≥2 secondary metrics improve | **NOT MEASURED** |
| 3 | Improvement not isolated to one league | **NOT MEASURED** — GLM shows 2/5 |
| 4 | No leakage | **PASS** — 10/10 checks |
| 5 | c2 numerically stable across folds | **PASS (measured)** — sign stable in all 6; spread 1.18×/1.81× |
| 6 | Zero-quadratic control reproduces V3 | **PASS at feature level**; model level pending |
| 7 | No implausible/extreme λ | **PASS (measured)** — range [0.72, 5.18] over [−430, +690] |

Criteria 4, 5 and 7 pass on measured evidence — **more than any prior incomplete experiment**. Criterion 6 is half-satisfied.

### What the measured evidence says

The hypothesis is **partly confirmed and practically unhelpful**:

- **Curvature is real.** Two independent methods agree — non-parametric decile fits (10.2% / 15.0% SSE reduction) and an IRLS GLM (c2 non-zero in all six estimates).
- **c2 is positive**, not negative. Both goal rates are convex in Elo difference; lopsided fixtures generate more total goals than log-linear predicts.
- **c2 is the most stable parameter measured in E3–E5** — sign-consistent across all folds, magnitude within 1.18×/1.81×.
- **And it buys essentially nothing.** Pooled OOS log-loss delta **+0.000060**, one fold of three improving, 2 of 5 leagues.

A stable, reliably-estimated, genuinely-present effect that does not transfer to out-of-sample probability accuracy. Given §29's rule — *"if it improves only one or two leagues, do not automatically keep it"* — the expectation is **FAIL or INCONCLUSIVE**, not PASS.

**Honest expectation:** PASS is unlikely, most probably failing criterion 1 (the pooled delta is within noise of zero) and criterion 3 (2 of 5 leagues). But that is an expectation from an Elo-only model, not a measurement of V3, and E5 should be decided by running it.

### To complete E5

```bash
cd "E:\Football Prediction Project"
python research/quadratic_elo/test_quadratic_elo.py     # expect 66/66
python research/quadratic_elo/run_e5_experiment.py
```

The harness verifies the model-level zero-quadratic control first and **halts if it fails**, then applies the seven criteria mechanically.

---

## 30. Recommendation for E6

**Proceed to E6.** Online attack/defence is independent of Elo's functional form.

Three carry-overs:

1. **Retire the per-league hypothesis.** §16 shows the 2-of-5 split has different membership every time — it is noise from slicing a null effect five ways, not a league property. E6 should not build per-league parameters, and should treat a 2-of-5 result as evidence *against* an effect rather than as partial support.

2. **Expect overlap with Elo, and test for it.** E5 found real curvature that adds nothing once fitted — the natural explanation is that V3's other 84 features already encode it. Online attack/defence is *more* likely to overlap with Elo than a quadratic term is, since both estimate team strength from results. E6 should measure the incremental contribution over Elo directly, not just A-vs-B totals.

3. **Note the emerging pattern across E3–E5.** Three different structural refinements — low-score dependence, recency weighting, Elo nonlinearity — each turned out to be real in the data and worth ~1e-4 or less in out-of-sample log loss. That consistency is itself a finding: V3's 87 features appear to have already absorbed most of the structure these classical corrections target. If E6 lands the same way, the honest conclusion for E7 is that **V3 plus market odds is the model**, and further structural elaboration is not where the remaining gains are.

---

## ROADMAP PROGRESS

```
E0 — V2 Baseline                  COMPLETE
E1 — Causal Elo -> V3             PASS
E2 — Market Odds                  PASS
E3 — Dixon-Coles                  INCONCLUSIVE
E4 — Exponential Time Decay       INCONCLUSIVE
E5 — Quadratic Elo -> xG          INCONCLUSIVE
E6 — Online Attack/Defense        NEXT
E7 — Best Proven Combination      PENDING
```

**3 experiments are complete** (E0, E1, E2). E3, E4 and E5 are implemented, tested and diagnosed, but their A/B comparisons are unmeasured, so none is counted. **E3 and E4 are not PASS.**

**NEXT EXPERIMENT: E6 — Online Attack/Defense**

---

**Nothing promoted. Production default unchanged. V2 unmodified. V3 artifact unmodified. No market, no Dixon-Coles, no time decay, no combined model. Research only.**
