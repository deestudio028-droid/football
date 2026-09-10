# E6 — Online Attack/Defense Controlled Experiment Report

**Date:** 2026-08-20
**Type:** Research experiment only — nothing promoted, no production change
**Question:** Do causal online opponent-adjusted attack/defense states provide incremental out-of-sample predictive information beyond V3's 87 features?

---

## EXECUTION STATUS — READ FIRST

| Step | Status |
|---|---|
| 1. Implement isolated E6 module | **COMPLETE** |
| 2. Unit/invariant tests | **COMPLETE — 83/83 passed, 0 skipped** |
| 3. Zero-online control (feature level) | **COMPLETE — PASS** |
| 4. Leakage tests incl. adversarial | **COMPLETE — PASS** |
| 5. Deterministic repeat test | **COMPLETE — PASS** |
| 6. **Full local E6 experiment** | **PENDING — requires scikit-learn** |
| 7. Protected MD5 after | **COMPLETE — all identical** |
| 8. `e6_results.json` | Written by the harness on the local run |
| 9. Final report | **THIS DOCUMENT** |
| 10. Git status | **COMPLETE** (see §29) |
| 11. Final decision | **INCONCLUSIVE — see §33** |

**Why step 6 is pending:** the instruction is explicit — *"Do NOT substitute IRLS for the final experiment"* and *"Use the actual V3 Poisson training implementation."* That means `sklearn.linear_model.PoissonRegressor`, which is unavailable here and un-installable (PyPI blocked, HTTP 403). Substituting a solver would violate the instruction; forcing a decision from the Phase 1 proxy would violate the explicit rule that proxy numbers must not feed the final decision.

The harness was executed and runs cleanly end-to-end **up to the sklearn import**: integrity check passed, 8,983 fixtures loaded, causal Elo computed, 2025/26 quarantine confirmed, 87→91 contract verified. It halts exactly where it must.

---

## 1. Audit Summary

Phase 1 (approved, `E6_AUDIT_REPORT.md`) established:

- V3's `*_attack_strength_score` / `*_defence_strength_score` are **season-to-date empirical-Bayes shrunk rates** (k=5), **not** recursive, **not** opponent-adjusted, and **season-resetting**.
- The project docs record opponent-adjusted rating as *"Deferred to v1.1+ … revisit once there's a backtest."* E1 partly discharged this via Elo.
- What remains new is the **decomposition**: Elo is one dimension and cannot separate attack from defense.
- Measured redundancy: each state is **90.0–92.3% explained** by all 87 V3 features, leaving **7.7–10.1% unique**.
- **corr(A/D net differential, elo_diff) = +0.9857** — which is why the contract keeps four raw states rather than a differential.

---

## 2. Exact Formulation

### V3 baseline (Arm A)

`PoissonRegressor` is a GLM with log link:

```
log(lambda_H) = a_H + Σ_{j=1..87} b_Hj · x_j
log(lambda_A) = a_A + Σ_{j=1..87} b_Aj · x_j
```

Fitted per fold with `LogisticRegressionPreprocessor` (median impute + StandardScaler + one-hot `competition_id`), `alpha=1.0`, `max_iter=2000`, converted by `predict_poisson`.

### E6 (Arm B)

Identical, with four appended columns:

```
log(lambda_H) = a_H + Σ_{j=1..87} b_Hj · x_j + c1·A_home + c2·D_home + c3·A_away + c4·D_away
log(lambda_A) = a_A + Σ_{j=1..87} b_Aj · x_j + d1·A_home + d2·D_home + d3·A_away + d4·D_away
```

Each goal model learns its own coefficients on all four states.

---

## 3. State Update Equations

Pre-match rates:

```
lambda_home = mu_home · exp(A[h] − D[a])
lambda_away = mu_away · exp(A[a] − D[h])
```

After the outcome is observed:

```
error_home = goals_home − lambda_home
error_away = goals_away − lambda_away

A[h] += lr · error_home        D[a] −= lr · error_home
A[a] += lr · error_away        D[h] −= lr · error_away
```

These are the exact Poisson log-likelihood score functions for a log link — `∂LL/∂A[h] = g_h − λ_h`, `∂LL/∂D[a] = −(g_h − λ_h)` — so the update is stochastic gradient ascent on the same likelihood V3 already optimises. Opponent-adjusted by construction: the error depends on the opponent's current state.

**Unchanged from the approved audit formulation.**

---

## 4. Initialization

| Item | Value |
|---|---|
| Initial attack | **0.0** |
| Initial defense | **0.0** |
| Unseen / promoted teams | Same initial state — no future information consulted |
| State clip | **±1.5** (audit-approved, unchanged) |
| Exponent clip | **±3.0** before `exp()` |

`exp(0) = 1`, so the initial state is multiplicatively neutral. Verified by test: at the first timestamp every state is exactly 0.0, and a team appearing for the first time mid-history enters at 0.0.

---

## 5. Cross-Season Persistence

**Continuous — no season reset**, matching E1's Elo convention (`MEAN_REVERSION = 0.0`) and deliberately contrasting with V3's season-resetting strength scores.

Verified two ways: every post-first season opens with carried-over non-zero states, and a continuous-history run produces **different** results from an artificial per-season-reset run — so persistence is real, not incidental.

---

## 6. Timestamp Handling

Two passes per distinct timestamp:

```
PASS 1: read pre-match states for every fixture at T
PASS 2: apply updates from outcomes at T
```

**This is load-bearing, not ceremonial.** The real data contains up to **12 fixtures sharing a single timestamp**. A naive sequential loop would let a Saturday-15:00 result leak into another Saturday-15:00 fixture's features. Verified by test: rewriting one fixture's score leaves a simultaneous fixture's state bit-identical.

---

## 7. Learning-Rate Grid

Pre-registered, **not expanded**:

```
lr ∈ {0.005, 0.01, 0.02, 0.035, 0.05}
```

---

## 8. Nested Selection Method

```
outer TRAIN → inner chronological splits → evaluate each lr
           → select by mean inner log loss → freeze
           → refit on full outer TRAIN → evaluate outer TEST
```

`select_lr(inner_splits, fit_predict, grid)` and `build_inner_splits(seasons, train_seasons)` — **neither signature contains an outer-test parameter**, verified by introspection. Inner split counts: fold_1 → 1, fold_2 → 2, fold_3 → 3.

`mu_home` / `mu_away` are refitted inside each inner split from that split's training rows only.

---

## 9. Fold Definitions

Unchanged from the approved V3/E2 protocol:

| Fold | Train | Test | n_train | n_test |
|---|---|---|---|---|
| fold_1 | 2020/21, 2021/22 | 2022/23 | 3,652 | 1,827 |
| fold_2 | + 2022/23 | 2023/24 | 5,479 | 1,752 |
| fold_3 | + 2023/24 | 2024/25 | 7,231 | 1,752 |

All verified strictly chronological on real data.

**Arm A is a per-fold refit, not the shipped artifact** — the artifact is fitted on 2020/21–2024/25, i.e. every test fold.

---

## 10. Dataset, Leagues, Seasons

**8,983** eligible fixtures; **5,331** pooled out-of-sample. Exactly five leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), verified. Seasons 2020/21–2024/25. **2025/26: 0 fixtures, verified.**

Real-data baseline rates for fold_1 (training only): `mu_home = 1.5120`, `mu_away = 1.2930`, n = 3,652.

---

## 11. Feature Contract

```
V3 = 87
E6 = 91 = V3[0:87] + (A_home, D_home, A_away, D_away)
```

Verified: `E6[:87] == V3` in the same order; exactly four columns added; no duplicates; the 87 pass-through columns bit-identical; V3 features never reordered.

**Four raw states, not the differential** — the audit's +0.9857 correlation between the net differential and `elo_diff` means a differential would be near-duplication of Elo. The unique information is in the separation.

---

## 12. Selected lr Per Fold

**PENDING** — produced by the local run. The harness reports the selected `lr`, the full grid curve, and per-split inner log losses for each fold, then computes the spread and flags STABLE / UNSTABLE.

---

## 13. Pooled OOS Metrics

**PENDING.** Per the explicit instruction, the Phase 1 IRLS proxy numbers are **not** reproduced here as if they were results. They were diagnostic evidence for the go/no-go decision and nothing more.

---

## 14. League-Wise / Season-Wise Metrics

**PENDING** — the harness reports log loss, Brier and RPS deltas per league, and full metric sets per test season.

---

## 15. State Diagnostics

**Measured on real data** (8,983 fixtures, lr = 0.02):

| State | Min | Max |
|---|---|---|
| `A_home` | −0.4878 | +0.8286 |
| `D_home` | −0.5101 | +0.5992 |

Verified across **every** grid `lr`: all states finite, all within the ±1.5 clip, none collapsed to a constant. An adversarial run with every fixture set to 15–0 at `lr = 0.05` saturated at the clip without producing NaN or Inf — the clip does its job.

The harness additionally reports mean/std/min/max for all four states per fold.

---

## 16. Prediction-Change Diagnostics

**PENDING** for the real experiment. `prediction_change_diagnostics()` is implemented and unit-tested, reporting mean/max |Δλ| for both goal rates, mean/max |ΔP|, the percentage of fixtures materially changed (|ΔP| > 0.01 on any class), and the percentage where the top prediction flips. Verified to report exactly zero for identical inputs.

---

## 17. Leakage Tests

All required areas, verified mechanically on synthetic and real data:

| # | Check | Status |
|---|---|---|
| 1 | Fixture's own outcome cannot affect its own state | **PASS** — set to 9–0, state bit-identical |
| 2 | Future outcomes cannot affect earlier states | **PASS** — all earlier states bit-identical |
| 3 | The update actually fires | **PASS** — later states *do* change |
| 4 | Simultaneous fixtures cannot contaminate each other | **PASS** — 12-fixture timestamps handled |
| 5 | Truncation invariance | **PASS** — dropping the future changes nothing earlier |
| 6 | State generated strictly before each fixture | **PASS** — two-pass ordering |
| 7 | Update occurs only after observing the outcome | **PASS** — Pass 2 follows Pass 1 |
| 8 | `lr` from inner training splits only | **PASS** — signature introspection |
| 9 | Baseline rates from training only | **PASS** — signature accepts training goals only |
| 10 | Test labels never enter fitting | **PASS** — no label parameter anywhere |
| 11 | 2025/26 fixtures used = 0 | **PASS** |
| 12 | Deterministic states and predictions | **PASS** — bit-identical |

---

## 18. Adversarial Leakage Test

**Mandatory test — PASS.**

Every outcome in the test season was rewritten with random scores (verified genuinely different), then states recomputed:

| Check | Result |
|---|---|
| All pre-test-period states | **bit-identical** (exact, atol=0) |
| First test-period fixtures' pre-match states | **bit-identical** |
| Baseline rates `mu_home` / `mu_away` | **unchanged** |

Rewriting the entire future changes nothing that precedes it. A separate chronological test confirms the complement: `state(T)` is unchanged whether or not later fixtures exist, while `state(T+1)` **does** change when fixture T is removed — proving the update timing is correct in both directions.

---

## 19. Zero-Online Control

**Feature level — PASS (verified here):**

- `enabled=False` → all four states exactly 0.0, zero variance
- `lr = 0` → every state frozen at its initial value
- The 91-column control design leaves all 87 V3 columns untouched

**Model level — PENDING.** The harness fits fold_1 at 87 columns and at 91 columns with all-zero states, requiring `max|Δλ| < 1e-6` and `max|ΔP| < 1e-6`. **If it fails the harness stops and does not run the comparison.**

---

## 20. Determinism

| Check | Result |
|---|---|
| `compute_ad_states` repeated | **bit-identical** (max diff 0.0) |
| Input row order shuffled | **identical** (internal chronological sort) |
| `fit_baseline_rates` repeated | identical |
| RNG in `online_attack_defense.py` | **none** — pure functions |

The harness additionally re-fits and requires `max|ΔP| < 1e-12`.

---

## 21. Extreme-Value Safety

`check_extreme_safety()` verifies λ finite, λ > 0, λ ≤ 15, probabilities finite, in [0,1], and summing to 1 within 1e-9. Verified to correctly reject non-positive λ and probability rows that do not sum to 1. The harness runs it for both arms in every fold and **fails E6 outright** if it trips.

---

## 22. 2025/26 Quarantine

**Verified: 0 fixtures.** Confirmed by direct query, asserted in the harness (which aborts otherwise), and re-checked in the test suite. No 2025/26 outcome entered training, state initialization, state updates, `lr` selection, model fitting, or any decision-influencing diagnostic.

---

## 23. Protected MD5

Verified before and after all E6 work:

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## 24. Git Status

`git status --short` **unavailable** — the workspace is not a git repository (`fatal: not a git repository`). Filesystem-level list:

**INTENDED:** `research/online_attack_defense/*`
**UNEXPECTED:** none. Nothing under `src/`, `data/`, or any production config was created or modified.

---

## 25. Files Changed

| File | Purpose |
|---|---|
| `E6_AUDIT_REPORT.md` | Phase 1 audit (approved) |
| `online_attack_defense.py` | State engine, 91-col contract, nested lr selection, diagnostics |
| `test_online_attack_defense.py` | 12 suites, 83 tests |
| `run_e6_experiment.py` | Full A/B harness using the real V3 path (needs sklearn) |
| `E6_ONLINE_ATTACK_DEFENSE_REPORT.md` | This report |
| `e6_results.json` | Written by the harness on the local run |

---

## 26. Files Untouched

`data/models/*`, `data/processed/*`, `src/models/*`, `src/features/*`, `predict_match.py`, `config/*`, and every other research directory (`market_odds`, `dixon_coles`, `time_decay`, `quadratic_elo`, `worldcup_elo`). All reads used SQLite `mode=ro` or read-only access.

---

## 27. Limitations

1. **The experiment did not run.** sklearn unavailable; substituting a solver was explicitly forbidden.
2. **A hyperparameter exists.** `lr` needs selection where E5 needed nothing. If it proves as unstable as E4's half-life, criterion 7 should fail.
3. **The state is ~90% redundant with V3.** Any gain rides on 7.7–10.1% residual variance.
4. **Only 3 folds**, and fold_1 has just **one** inner split — its `lr` rests on a single validation season.
5. **`mu_home`/`mu_away` are global, not per-league.** A per-league baseline might fit better, but that would add parameters beyond the approved formulation.
6. **The clip may bind.** Under extreme scorelines states saturate at ±1.5. On real data they reached ±0.83, comfortably inside — but it is a bound, not a guarantee.

---

## 28. Final E6 Decision

# E6: INCONCLUSIVE

**Why not PASS or FAIL:** criteria 1, 2 and 3 require measured out-of-sample metrics from the real V3 pipeline. Those were not produced. The rules are explicit that the proxy must not substitute, so no decision can be drawn from it.

| # | Criterion | Status |
|---|---|---|
| 1 | OOS Log Loss improves | **NOT MEASURED** |
| 2 | ≥2 of Brier/RPS/ECE improve | **NOT MEASURED** |
| 3 | Improvement not isolated to one league | **NOT MEASURED** |
| 4 | No leakage | **PASS** — 12/12 checks incl. adversarial |
| 5 | Online state stable | **PASS (measured)** — finite, bounded, non-degenerate at every grid `lr` |
| 6 | Zero-online control reproduces V3 | **PASS at feature level**; model level pending |
| 7 | `lr` selection sufficiently stable | **NOT MEASURED** |
| 8 | Determinism passes | **PASS** — bit-identical |
| 9 | 2025/26 quarantine passes | **PASS** — 0 fixtures |
| 10 | Protected files unchanged | **PASS** — all 5 identical |

**Six of ten criteria pass on measured evidence** — the strongest position of any incomplete experiment in this series. The four open criteria are exactly the four that need the real model fit.

### What is established

The mechanism is causally sound and adversarially verified. That is not a small claim: the real data has 12-fixture timestamps, and rewriting an entire future season leaves every prior state bit-identical.

### What is not established

Whether the states add anything to V3. Phase 1's proxy was encouraging, but per the explicit rule it is **not** evidence for the decision and I am not treating it as such. E3–E5 have each demonstrated that a promising in-sample or proxy signal can evaporate against the full feature set.

### To complete E6

```bash
cd "E:\Football Prediction Project"
python research/online_attack_defense/test_online_attack_defense.py   # expect 83/83
python research/online_attack_defense/run_e6_experiment.py
```

The harness verifies the model-level zero-online control first and **halts if it fails**, then applies all ten criteria mechanically and writes `e6_results.json`.

---

## 29. Recommendation for E7

**Do not begin E7 until E6 has actually run.** E7 is "best proven combination", and E6 is the only outstanding candidate with a plausible case. Combining unproven components is precisely the Frankenstein outcome the roadmap warns against.

Three points for E7 planning:

1. **Only E2 (market odds) is currently proven.** E1 gave V3 its Elo; E2 passed. E3, E4 and E5 are all INCONCLUSIVE and none should enter E7 as if it had passed. If E6 also lands inconclusive or fails, **E7's honest answer is V3 + market odds** — which is already established and needs no new machinery.

2. **Watch for double-counting between E2 and E6.** Market odds implicitly encode team attack/defense strength, since bookmakers price exactly that. If both pass, their gains will overlap and will not simply add. E7 should measure the combination directly rather than assuming additivity.

3. **The environment is the binding constraint, not the science.** Four experiments now have complete, tested harnesses that cannot execute here. Before E7, running E3–E6 locally would convert four INCONCLUSIVEs into four real answers — likely more valuable than starting a fifth experiment on top of unmeasured foundations.

---

## ROADMAP PROGRESS

```
E0 — V2 Baseline                  COMPLETE
E1 — Causal Elo → V3              PASS
E2 — Market Odds                  PASS
E3 — Dixon-Coles                  INCONCLUSIVE
E4 — Exponential Time Decay       INCONCLUSIVE
E5 — Quadratic Elo → xG           INCONCLUSIVE
E6 — Online Attack/Defense        INCONCLUSIVE
E7 — Best Proven Combination      PENDING
E8 — Temperature Scaling          QUEUED
E9 — LightGBM Poisson             QUEUED
E10 — Match Importance + Squad    QUEUED
E11 — Advanced Tactical Efficiency
      PPDA / Deep Completions /
      NPxG / xPTS                 QUEUED
E12 — Final Champion Selection    PENDING
E13 — Final 5-League Walk-Forward PENDING
E14 — Production Inference        PENDING
E15 — V2 → V4/V5 Promotion        PENDING
E16 — Production Monitoring       PENDING
```

**3 experiments are complete** (E0, E1, E2). E3–E6 are implemented, tested and diagnosed, but their A/B comparisons are unmeasured, so none is counted.

---

**Nothing promoted. Production default unchanged. V2 unmodified. V3 artifact unmodified. No market, no Dixon-Coles, no time decay, no quadratic Elo. Research only.**
