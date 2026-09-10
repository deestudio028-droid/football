# E4 — Exponential Time Decay Controlled Experiment Report

**Date:** 2026-08-20
**Type:** Research experiment only — nothing promoted, no production change
**Question:** Does exponential recency weighting of training observations improve V3?

---

## EXECUTION STATUS — READ FIRST

| Component | Status |
|---|---|
| Training-pipeline audit (§9) | **COMPLETE** |
| `time_decay.py` implementation | **COMPLETE** |
| Test suite | **COMPLETE — 69/69 passed, 0 skipped** |
| Signal-decay diagnostic | **COMPLETE — measured on real data** |
| Walk-forward proxy | **COMPLETE — measured on real data** |
| **Experiment A (V3, uniform weights)** | **PENDING — requires local run** |
| **Experiment B (V3 + decay weights)** | **PENDING — depends on A** |
| **Zero-decay control, model level (§19)** | **PENDING — requires local run** |

**Reason:** unlike E3 (a post-processing transform on fixed lambdas), E4 changes **training**. Both arms require refitting `PoissonRegressor` per fold, which requires scikit-learn. The sandbox has none and cannot install it (PyPI blocked, HTTP 403). Writing a hand-rolled Poisson GLM was rejected deliberately: §19 requires the zero-decay control to reproduce *V3's* training behaviour, and a clone that cannot be checked against sklearn would make that control meaningless.

What did run on real data: the signal-decay measurement and a walk-forward proxy. Both are reported with their numbers and their caveats.

---

## 1. Roadmap Status

| Experiment | Status |
|---|---|
| E0 — V2 Baseline | COMPLETE |
| E1 — Causal Elo → V3 | PASS |
| E2 — Market Odds | PASS |
| E3 — Dixon-Coles | INCONCLUSIVE |
| **E4 — Exponential Time Decay** | **THIS REPORT** |

---

## 2. Experiment Objective

Test whether weighting training observations by `w_i = exp(-xi * ΔDays_i)` improves V3's out-of-sample probabilistic accuracy. The weight is a **training weight**, never a feature. The 87-column contract is untouched.

---

## 3. Hypothesis

Recent matches may carry more relevant information than old ones. The experiment is designed to let the evidence decide — the candidate grid includes an explicit **no-decay** option, so the procedure can decline to apply decay at all.

---

## 4. Dataset

`data/processed/matches.db` joined to `data/processed/features.db`, both read-only. No market odds, no Dixon-Coles, no new features.

---

## 5. Fixture Count

**8,983** eligible fixtures. Pooled out-of-sample across the three test folds: **5,331**.

---

## 6. Leagues

Premier League, La Liga, Serie A, Bundesliga, Ligue 1 — verified as exactly these five, no others.

---

## 7. Seasons

2020/21, 2021/22, 2022/23, 2023/24, 2024/25. **2025/26 quarantined — 0 fixtures, verified.**

---

## 8. V3 Baseline Definition — an important audit finding

**Arm A is a per-fold refit, NOT the shipped artifact.**

`v3_poisson_venue_elo_candidate.pkl` is fitted on `FINAL_TRAIN_SEASONS` = 2020/21 through 2024/25 — that is, on **every season the walk-forward folds use as test data**. Scoring fold test sets with it would leak all test outcomes into training.

The V3 that was *validated* in E1 is the per-fold refit performed inside `research/worldcup_elo/run_walk_forward_experiments.py`, which refits the preprocessor and both regressors inside each fold (lines 141–149, inside the fold loop). Arm A reproduces exactly that. The shipped artifact is the separate final fit used only to predict the quarantined 2025/26 season.

Had this not been caught in audit, E4 would have compared a leaked baseline against a clean treatment.

---

## 9. Existing V3 Training Methodology (audit)

| Item | Finding |
|---|---|
| Model | `PoissonRegressor(alpha=1.0, max_iter=2000)` × 2 (home goals, away goals) |
| Target | `label_home_goals`, `label_away_goals` (counts, not 1X2) |
| Features | 87 = 80 base + 4 venue + 3 causal Elo |
| Preprocessor | `LogisticRegressionPreprocessor` — median impute + StandardScaler on 86 numeric columns, one-hot of `competition_id` |
| **sample_weight support** | **Yes** — sklearn's `PoissonRegressor.fit()` accepts `sample_weight` natively. Arm B is a one-argument change; no model surgery needed. |
| Lambdas | `model_home_goals.predict(E)`, `model_away_goals.predict(E)` |
| 1X2 conversion | `src/models/poisson.py::predict_poisson`, adaptive grid K, tail < 1e-15 |
| Fold structure | 3 walk-forward folds by season |
| **Window type** | **Expanding** — fold_1 trains on 2 seasons, fold_2 on 3, fold_3 on 4 |
| Contract impact | None — weighting requires no feature change |

---

## 10. Time-Decay Formulation

```
w_i = exp(-xi * ΔDays_i)          ΔDays_i = T_ref - t_i  >= 0,   xi >= 0
```

`T_ref` is the **prediction-time reference** — the first kickoff of the period being predicted — **never today's real-world date**. A run in 2026 and a run in 2030 produce identical weights. Verified by test.

`xi = 0` gives `w_i = 1` for every observation, reproducing unweighted training exactly.

---

## 11. Candidate xi Grid

Expressed as half-lives in days:

```
60, 90, 120, 180, 240, 365, 540, 730, None (no decay)
```

**Rationale:** a domestic league plays roughly one match per team per week, so a season spans ~280 days of fixtures. 60 days is about a third of a season (very aggressive); 730 days is two full seasons (very mild). The grid spans that range roughly geometrically, plus an explicit no-decay control.

**The World Cup value `xi = 0.001` is not used and is not in the grid** — verified by an explicit test. That value came from international football, a different domain with far sparser fixtures.

---

## 12. Half-Life Conversion

`half_life = ln(2) / xi`. Verified by test: at each grid half-life the weight is exactly 0.5 at that age (to 1e-12), and round-tripping half-life → xi → half-life is exact to 1e-9.

Effective weights by age:

| Half-life | 30d | 90d | 180d | 365d | 730d |
|---|---|---|---|---|---|
| 60d | 0.707 | 0.354 | 0.125 | 0.015 | 0.0002 |
| 180d | 0.891 | 0.707 | 0.500 | 0.246 | 0.060 |
| 365d | 0.944 | 0.842 | 0.710 | 0.500 | 0.250 |
| 730d | 0.972 | 0.917 | 0.842 | 0.708 | 0.500 |
| None | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 13. Hyperparameter Selection Method

**Nested temporal validation, training period only.**

For outer training seasons `(s1, …, sk)`, `build_inner_splits` yields `k-1` inner splits: train on `(s1..sj)`, validate on `s(j+1)`. Each inner split's reference time is the first kickoff of its own validation season. xi is chosen by **mean inner-validation log loss**, then frozen and applied to a refit on the full outer training period.

`select_xi`'s signature is `(inner_splits, fit_predict, half_life_grid)` and `build_inner_splits` receives `train_seasons` only — **neither has any parameter through which outer-test data could enter**. Both verified by signature introspection in the test suite.

Inner split counts: fold_1 → 1, fold_2 → 2, fold_3 → 3.

---

## 14. Walk-Forward Protocol

Unchanged from E1/E3:

| Fold | Train | Test | n_test |
|---|---|---|---|
| fold_1 | 2020/21, 2021/22 | 2022/23 | 1,827 |
| fold_2 | + 2022/23 | 2023/24 | 1,752 |
| fold_3 | + 2023/24 | 2024/25 | 1,752 |

All three verified strictly chronological on real data.

---

## 15. Pooled OOS Results

**PENDING** — requires the local run.

### What was measured instead: does signal actually decay with age?

**First attempt, variable-width bands — and why it was wrong.** Using bands like [0,30) and [240,365), correlation between a team-form differential and the actual match margin appeared *flat or increasing* with age (0.2893 at [0,30) rising to 0.3226 at [540,730)). That result is an artifact: wider bands contain more matches, so the form estimate is less noisy. Age was confounded with sample size.

**Corrected, equal-width 120-day bands:**

| Age band (days) | n | Avg matches/team | Correlation |
|---|---|---|---|
| [0, 120) | 8,670 | 12.5 | **0.3841** |
| [120, 240) | 7,166 | 11.5 | 0.3481 |
| [240, 360) | 5,518 | 12.2 | 0.3162 |
| [360, 480) | 5,066 | 12.8 | 0.2938 |
| [480, 600) | 4,419 | 11.9 | **0.2803** |
| [600, 720) | 3,763 | 12.2 | 0.3048 |

**Corrected, fixed sample size (exactly 10 matches per team, only age varies):**

| Offset (days) | n | Correlation |
|---|---|---|
| 0 | 8,153 | **0.3756** |
| 60 | 7,476 | 0.3600 |
| 120 | 6,800 | 0.3510 |
| 180 | 6,147 | 0.3420 |
| 240 | 5,935 | 0.3262 |
| 365 | 5,247 | 0.3189 |
| 540 | 4,087 | **0.3089** |

**Signal does decay with age — but very gradually.** Correlation falls only ~18% relative over 1.5 years. An exponential fit to the fixed-N curve gives:

```
implied xi        = 0.00036 / day
implied half-life ≈ 1,926 days  (≈ 5.3 years)
```

That is **far milder than every finite half-life in the candidate grid**. It predicts that the selection procedure should favour the long end (730 days) or no decay at all, and that any effect will be small.

---

## 16. League-Wise Results

**PENDING** for A/B. The proxy's league breakdown is in §31.

---

## 17. Season-Wise Results

**PENDING** — the harness reports per-season A/B metrics for all three test seasons.

---

## 18. Selected xi Per Fold

**PENDING** for the real experiment. The proxy's selections are in §20.

---

## 19. Selected Global xi

**PENDING.** Note the design does not use one global xi: each fold selects its own from its own training period. §20 reports whether those selections agree.

---

## 20. Stability Analysis

**PENDING** for the real experiment. The proxy result is a warning sign:

| Fold | Test season | Selected half-life |
|---|---|---|
| fold_1 | 2022/23 | 240 days |
| fold_2 | 2023/24 | 365 days |
| fold_3 | 2024/25 | **730 days** |

**Spread = 3.04× → UNSTABLE** by the preregistered criterion (a 2× band counts as reasonably stable).

The selections are monotonically increasing, which is at least an interpretable pattern rather than noise — as the training window lengthens, the procedure prefers milder decay. But a 3× spread across only three folds means the selected value is not converging, and criterion 6 requires reasonable stability.

---

## 21. Weight Diagnostics

Verified on **real fold_1 data**:

| Quantity | Value |
|---|---|
| Training fixtures | 3,652 |
| Age span at reference | **74.9 to 714.1 days** |
| Weight range at 180d half-life | 0.0639 to 0.7493 |

A consequence worth flagging: because of the summer break, the **newest** training fixture in fold_1 is already 74.9 days old. At a 180-day half-life it receives only 0.749 weight, and the oldest receives 0.064 — an 11.7× spread. Aggressive half-lives effectively discard most of the training set, which for fold_1 means discarding most of a 3,652-fixture sample.

Weight sanity, all verified: xi ≥ 0; all weights finite; all weights > 0; monotonically decreasing in age; older never outweighs newer; w(0) = 1.0 exactly; xi = 0 gives all-ones.

---

## 22. Draw Analysis

**PENDING** for A/B. Context from E3: V3's `draw_predicted_rate = 0.0` — it never makes "draw" its argmax.

Time decay is **not** a plausible fix for this. Decay reweights training observations; it does not change the independent-Poisson structure that makes P(Draw) rarely exceed both P(Home) and P(Away). Decay could shift lambdas slightly and thus nudge P(Draw), but the structural cause is untouched. The harness reports draw metrics for both arms so this can be confirmed rather than assumed.

Per instruction, no Dixon-Coles was added to address this.

---

## 23. Leakage Tests

All ten required checks, verified mechanically:

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Future fixtures cannot affect earlier predictions | **PASS** | Appending a future fixture leaves earlier weights bit-identical |
| 2 | Current fixture outcome cannot affect its own prediction | **PASS** | No weight-path function accepts an outcome parameter (introspection) |
| 3 | Changing future outcomes doesn't change earlier predictions | **PASS** | Weights depend only on timestamps |
| 4 | Changing test labels doesn't affect selected xi | **PASS** | `select_xi` receives only inner splits from training seasons |
| 5 | xi selected using training data only | **PASS** | Signature has no outer-test parameter |
| 6 | Test labels never used for hyperparameter selection | **PASS** | Same |
| 7 | ΔDays always non-negative | **PASS** | Clipped at 0; verified on real data |
| 8 | Reference is fixture timestamp, not today | **PASS** | Reference differs from wall-clock by >30 days; weights reproducible |
| 9 | No 2025/26 fixture enters training or validation | **PASS** | Query returns 0; asserted in harness |
| 10 | Same input produces deterministic weights | **PASS** | Bit-identical, max diff 0.0 |

Plus: `assert_causal` hard-fails if any training fixture is at or after the reference — verified to fire correctly and to accept real fold_1 data.

---

## 24. Zero-Decay Control (§19)

**Weight level — PASS (verified here):** `decay_weights(unix, ref, xi=0)` returns exactly 1.0 for every observation. Not approximately — exactly.

**Model level — PENDING.** The harness fits fold_1 twice, once with `sample_weight=None` and once with an all-ones vector, and requires `max|Δlambda| < 1e-9` and `max|ΔP| < 1e-9`. **If this control fails the harness stops and does not run the experiment**, exactly as §19 requires.

---

## 25. Determinism

| Check | Result |
|---|---|
| `decay_weights` repeated calls | **bit-identical** (max diff 0.0) |
| `xi_from_half_life` repeated calls | identical |
| `build_inner_splits` repeated calls | identical splits |
| RNG inside `time_decay.py` | **none** — all functions pure |
| Seed for synthetic test data | 20260820, explicit |

The harness additionally re-fits a weighted model twice and requires `max|ΔP| < 1e-12`.

---

## 26. Protected MD5 Results

Verified before and after all E4 work:

| File | Expected | Actual | Status |
|---|---|---|---|
| `data/models/v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `data/models/v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `data/models/v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## 27. Git Status

`git status --short` is **unavailable** — the workspace is not a git repository (`fatal: not a git repository`). Filesystem-level change list substituted below.

**INTENDED (all within research scope):** `research/time_decay/*`
**UNEXPECTED:** none.

---

## 28. Files Changed

All created under `research/time_decay/`:

| File | Purpose |
|---|---|
| `time_decay.py` | Weight function, half-life conversion, grid, nested temporal xi selection |
| `test_time_decay.py` | 8 suites, 69 tests |
| `run_e4_experiment.py` | Full A/B harness with zero-decay control (requires sklearn) |
| `run_e4_diagnostic.py` | Signal-decay + proxy analysis (no sklearn) |
| `e4_results.json` | Machine-readable results |
| `E4_TIME_DECAY_REPORT.md` | This report |

---

## 29. Files Untouched

`data/models/*`, `data/processed/*`, `src/models/*`, `src/features/*`, `predict_match.py`, `research/market_odds/*`, `research/dixon_coles/*`, `research/worldcup_elo/*`, `config/*`. All reads used SQLite `mode=ro` or read-only file access.

---

## 30. Known Limitations

1. **A and B not executed here.** scikit-learn unavailable; a hand-rolled GLM was rejected because it would void the §19 control.
2. **Only 3 folds, and fold_1 has just 1 inner split.** xi for fold_1 is chosen from a single inner validation season — thin evidence.
3. **Proxy has no features.** It bounds effect size and reveals stability; it cannot predict V3's numbers.
4. **The summer-break gap.** The newest training fixture is already ~75 days old at the reference, so decay begins from a substantial baseline offset rather than from 1.0.
5. **Expanding window confounds fold comparison.** Later folds have both more data and more inner splits, so the trend in selected half-life (240 → 365 → 730) may partly reflect training-set size rather than a genuine drift in optimal decay.
6. **Correlation is not log loss.** §15's decay curve measures a linear association with match margin, which is a proxy for, not identical to, the probabilistic objective E4 optimises.

---

## 31. E4 Decision

# E4: INCONCLUSIVE

**Why not PASS or FAIL:** criteria 1–3 and 6–7 require measured A vs B metrics and a model-level zero-decay control. None were produced, because both arms require refitting V3 with scikit-learn.

**Criteria status:**

| # | Criterion | Status |
|---|---|---|
| 1 | Log loss improves | **NOT MEASURED** |
| 2 | ≥2 secondary metrics improve | **NOT MEASURED** |
| 3 | Improvement not isolated to one league | **NOT MEASURED** |
| 4 | No leakage | **PASS** — 10/10 checks |
| 5 | xi selection strictly training-only | **PASS** — verified by signature introspection |
| 6 | xi reasonably stable across folds | **NOT MEASURED** — proxy suggests instability |
| 7 | Zero-decay control reproduces V3 | **PASS at weight level**; model level pending |

Criteria 4 and 5 are satisfied on measured evidence. Criterion 7 is half-satisfied.

### Advance signal from the proxy — not a verdict

A recency-weighted goal-rate model run through the exact E4 fold structure and nested temporal xi selection:

| Test season | Selected half-life | A log loss | B log loss | Delta |
|---|---|---|---|---|
| 2022/23 | 240 d | 1.020634 | 1.019840 | **+0.000794** |
| 2023/24 | 365 d | 0.993689 | 0.995620 | **−0.001932** |
| 2024/25 | 730 d | 1.007096 | 1.003163 | **+0.003934** |
| **Pooled** | | **1.007329** | **1.006399** | **+0.000930** |

Leagues improved: **2 of 5** (Premier League +0.003652, Serie A +0.004655). Bundesliga, La Liga and Ligue 1 all degraded, Ligue 1 worst at −0.003486.

Four observations: the pooled gain is **positive but tiny** (+0.00093, roughly 0.09% of log loss); one fold of three is actively **worse**; selection is **unstable at 3.04× spread**, failing criterion 6; and improvement reaches only **2 of 5 leagues**, which strains criterion 3.

**This proxy is not the experiment.** Its lambdas come from a featureless goal-rate model; V3 uses 87 features including causal Elo, which already encodes recency-sensitive team strength. It is quite possible that V3's Elo features **already capture most of the recency signal**, leaving decay less to add — which would make the real effect smaller than the proxy's, not larger.

The independent measurement in §15 points the same way: an implied half-life of ~1,926 days is far milder than anything in the grid, so the most likely selection is the grid's long end or no decay at all.

**Honest expectation:** PASS looks unlikely, most probably failing on criterion 6 (stability) and criterion 3 (breadth). But that is an expectation, not a measurement, and E4 should be decided by running it.

### To complete E4

```bash
cd "E:\Football Prediction Project"
python research/time_decay/test_time_decay.py     # expect 69/69
python research/time_decay/run_e4_experiment.py
```

The harness verifies the model-level zero-decay control first and **halts if it fails**, then applies the seven criteria mechanically.

---

## 32. Recommendation for E5

**Proceed to E5 regardless of E4's outcome.** Quadratic Elo → xG changes how team strength maps to expected goals; it is independent of observation weighting.

Three carry-overs:

1. **Check whether Elo already encodes recency.** V3's causal Elo updates after every match, so it is inherently recency-weighted. If E4 returns a null result, the most likely explanation is that Elo has already absorbed this signal — which is directly relevant to E5, since E5 modifies how Elo is used. Worth testing explicitly in E5 rather than assuming.

2. **Keep the audit-first discipline.** E3 caught the `max_goals=10` versus adaptive-K discrepancy; E4 caught that the shipped V3 artifact is fitted on every test fold and cannot serve as the walk-forward baseline. Both would have silently invalidated their experiments. E5 should establish what the current Elo→lambda path actually is before changing it.

3. **Watch the same two structural weaknesses.** Only 3 folds, and per-league heterogeneity that keeps splitting these five leagues into different camps — Bundesliga/Serie A in E3, Premier League/Serie A in E4's proxy, with no stable grouping. If E5 also shows 2-of-5 league splits, that is a pattern worth investigating in its own right: it may mean a single global parameter is the wrong shape for this problem, which would be a more valuable finding than any individual experiment.

---

## ROADMAP PROGRESS

```
E0 — V2 Baseline                  COMPLETE
E1 — Causal Elo -> V3             PASS
E2 — Market Odds                  PASS
E3 — Dixon-Coles                  INCONCLUSIVE
E4 — Exponential Time Decay       INCONCLUSIVE
E5 — Quadratic Elo -> xG          NEXT
E6 — Online Attack/Defense        PENDING
E7 — Best Proven Combination      PENDING
```

**3 experiments are complete** (E0, E1, E2). E3 and E4 are implemented, tested, and diagnosed, but their A/B comparisons are unmeasured, so neither is counted as complete.

**NEXT EXPERIMENT: E5 — Quadratic Elo → xG**

---

**Nothing promoted. Production default unchanged. V2 unmodified. V3 artifact unmodified. No market, no Dixon-Coles, no combined model. Research only.**
