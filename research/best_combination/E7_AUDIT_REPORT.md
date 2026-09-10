# E7 — Phase 1 Audit Report (pre-implementation)

**Date:** 2026-08-20
**Status:** Audit only. No experiment code written. No production file touched.
**Per §3:** the ten required audit points are answered before any implementation.

---

## HEADLINE: FOUR FINDINGS THAT CHANGE WHAT E7 CAN CLAIM

§13 warned that if E2's PASS methodology used market probabilities in a blending architecture rather than as V3 regression features, E7 must not silently reinterpret it. **That is exactly the situation.** Three further findings compound it.

| # | Finding | Consequence |
|---|---|---|
| 1 | **E2 is a post-hoc blend, not a regression feature** | Arms B and C are architecturally incompatible; Arm D needs a defined bridge |
| 2 | **E2's learned weight hit w = 1.00 in fold 2** | The validated blend *discarded V3 entirely* in its final fold |
| 3 | **E2 "PASS" was blend-vs-V3; market alone beat the blend** | The proven component is arguably the market, not the blend |
| 4 | **Fixture intersection forces n = 3,479, and only 2 folds** | E6's 5,331-fixture / 3-fold result cannot be reproduced on E7's set |

None of these blocks E7. All of them change what a PASS would mean.

---

## Audit point 1 — Exact V3 feature contract

87 columns: 18 goals rolling + 20 form + 5 shrunk attack/defence + `competition_id` + 18 shots + 18 shots-on-target + 4 venue + 3 causal Elo.

Model: `LogisticRegressionPreprocessor` (median impute + StandardScaler + one-hot `competition_id`) → `PoissonRegressor(alpha=1.0, max_iter=2000)` × 2 → `predict_poisson` (adaptive grid K, tail < 1e-15).

---

## Audit point 2 — Exact market probability derivation

Power de-vig, solving `(1/O_H)^k + (1/O_D)^k + (1/O_A)^k = 1` by bisection (tol 1e-12, max 200 iterations, automatic bracket expansion, final renormalisation). Verified on the real dataset: closing probability sums are 1.000000000000 at both min and max.

Source: `research/market_odds/research_dataset.sqlite`, pre-computed. **Not recomputed in E7** — reused verbatim.

---

## Audit point 3 — Exact bookmaker selection

**Pinnacle only** (`bookmaker_id = 1`), 4,001 / 4,001 fixtures. No fallback was exercised.

---

## Audit point 4 — Closing / opening handling

**Closing is primary.** Opening exists in the dataset and is carried as a robustness diagnostic only. **Peak odds are not used anywhere.**

Causal classification, unchanged from E2: closing odds are `ASSUMED PRE-MATCH` on OddAlerts' definition. There are no independent per-record timestamps (the timestamped `odds/movement` endpoint has 21-day retention and is unusable for these seasons). E7 does not strengthen this assumption.

---

## Audit point 5 — Exact E6 state construction

Two log-scale states per team, initialised at 0.0, persisting across seasons with no reset:

```
lambda_home = mu_home · exp(A[h] − D[a])        lambda_away = mu_away · exp(A[a] − D[h])

A[h] += lr·(g_h − λ_h)    D[a] −= lr·(g_h − λ_h)
A[a] += lr·(g_a − λ_a)    D[h] −= lr·(g_a − λ_a)
```

Poisson score functions for a log link. Two-pass per timestamp (read all, then update all). State clip ±1.5, exponent clip ±3.0. `mu_home` / `mu_away` from training rows only.

**Reused verbatim from `research/online_attack_defense/online_attack_defense.py`.** Not redesigned.

---

## Audit point 6 — Exact E6 learning-rate selection

Pre-registered grid `{0.005, 0.01, 0.02, 0.035, 0.05}`, chosen by mean inner-validation log loss over nested chronological splits inside each outer training period, then frozen.

E6's local run selected **[0.01, 0.02, 0.01]**, spread 2.00×, flagged stable.

**E7 re-runs the selection procedure rather than hardcoding those values**, exactly as §7 requires. Note the caveat in §"Known consequences" below: E7's fold structure differs from E6's, so the selected values may legitimately differ — that would not be a reproduction failure.

---

## Audit point 7 — Exact outer fold structure

Here the two experiments diverge.

**E6 folds** (the approved V3/E1 protocol, 3 folds):

| Fold | Train | Test | n_test |
|---|---|---|---|
| 1 | 2020/21, 2021/22 | 2022/23 | 1,827 |
| 2 | + 2022/23 | 2023/24 | 1,752 |
| 3 | + 2023/24 | 2024/25 | 1,752 |

**E2 folds** (2 folds — market coverage is 0% before 2022/23):

| Fold | Train | Test | n_test |
|---|---|---|---|
| 1 | 2022/23 | 2023/24 | 1,727 |
| 2 | 2022/23, 2023/24 | 2024/25 | 1,752 |

E2 needs at least one season of *market* history to fit the blend weight, so 2022/23 is consumed as training and cannot be a test fold.

**E7 must adopt E2's 2-fold structure.** Using E6's 3 folds would require market probabilities for 2022/23 test fixtures with no market history to train the blend against.

---

## Audit point 8 — Exact inner validation structure

- **E6:** `build_inner_splits(seasons, train_seasons)` → k−1 nested chronological splits inside the outer training period; `select_lr` scans the grid by mean inner log loss. Neither signature accepts an outer-test parameter.
- **E2:** `learn_weight(y_train, p_v3_train, p_mkt_train)` scans a 21-point grid `w ∈ {0.00, 0.05, …, 1.00}` on the outer *training* fold directly — no nested inner split.

Both are training-only. They are not the same mechanism.

---

## Audit point 9 — Exact fixture intersection

| Universe | n |
|---|---|
| V3 / E6 eligible (5 leagues, 2020/21–2024/25) | **8,983** |
| E2 market dataset | **4,001** |
| Intersection (market ∩ V3) | **4,001** |
| Less 2022/23, consumed as E2 blend-training | **−522** |
| **FINAL E7 ELIGIBLE (OOS)** | **3,479** |

Drop accounting:

| Reason | n dropped |
|---|---|
| `dropped_no_market` (2020/21, 2021/22 entirely; 71.4% of 2022/23) | 4,982 |
| `dropped_no_e6` | **0** — E6 states exist for every V3 fixture |
| `dropped_no_v3` | **0** — every market fixture has a V3 feature row |
| `dropped_no_label` | **0** |
| 2022/23 reserved as E2 blend-training | 522 |

Final set by season: 2023/24 = 1,727 · 2024/25 = 1,752.
By league: Premier League 755 · Serie A 755 · La Liga 753 · Ligue 1 610 · Bundesliga 606.

**This is exactly E2's reported OOS set (n = 3,479).** The intersection resolves cleanly with zero ambiguity — no arm can silently receive a different fixture set.

---

## Audit point 10 — Exact causal timing of both signals

| Signal | Causal basis | Verified |
|---|---|---|
| Market | Pre-match closing odds; de-vig is a pure function of `(O_H, O_D, O_A)` with no label parameter | E2 leakage checks |
| Online A/D | Two-pass per timestamp; adversarially tested by rewriting an entire future season | E6, 83/83 tests |

E6's adversarial test is worth restating: rewriting every test-season outcome left all prior states **bit-identical (atol = 0)**, including the first test-period fixtures. Real data has up to 12 simultaneous kickoffs, so the two-pass design is load-bearing.

---

## Finding 1 (detail) — E2 is a blend, not a regression feature

`research/market_odds/run_e2_experiment.py:199`

```python
def blend(p_v3, p_mkt, w):
    """P_final = w * P_market + (1 - w) * P_V3, renormalized."""
```

The market never entered a design matrix. E2 fitted V3 unchanged, produced `P_V3`, then mixed it with `P_market` at the probability level using a weight learned on training folds.

E6, by contrast, appends four columns to the design matrix and refits.

**Arms B and C are therefore architecturally different in kind.** §13 forbids reinterpreting E2 as regression features, so Arm B must stay a blend. That forces a decision about Arm D.

### Proposed Arm D — the only construction that preserves both validated methods

```
Step 1  Fit V3 + four A/D states (91 columns)   -> P_C          [E6 methodology, unchanged]
Step 2  Learn w on the training fold             -> w            [E2 methodology, unchanged]
Step 3  P_D = w · P_market + (1 − w) · P_C       -> P_D
```

Arm D is Arm C's output passed through Arm B's blend. Each validated component keeps its own architecture; neither is reinterpreted. No new hyperparameter is introduced — `w` is E2's existing one, and §15 explicitly says not to create one if it is not necessary.

**Zero-component controls fall out naturally:**

- `w = 0` → Arm D reduces to Arm C
- A/D states disabled → Arm D reduces to Arm B
- both → Arm A

---

## Finding 2 (detail) — E2's learned weight reached 1.00

| Fold | n_train | Learned w |
|---|---|---|
| 1 | 522 | 0.85 |
| 2 | 2,249 | **1.00** |

In fold 2 the training-fold optimum was to **discard V3 entirely** and use the market alone. Fold 1 kept only 15% V3.

This matters directly for E7. If `w` again approaches 1.00, Arm D will approach the market alone, and the four A/D states will be blended almost out of existence regardless of their quality. **A near-null D-vs-B result would then be a property of the blend architecture, not evidence that E6 lacks incremental information.** I will report `w` per fold prominently and interpret D vs B in that light.

---

## Finding 3 (detail) — what E2 actually proved

E2's pooled OOS:

| Arm | Log Loss | Brier | RPS | ECE |
|---|---|---|---|---|
| V3 | 0.981357 | 0.584001 | 0.197090 | 0.013069 |
| **Market alone** | **0.955905** | **0.567798** | **0.189968** | 0.007783 |
| Blend | 0.957130 | 0.568443 | 0.190261 | **0.007647** |

The blend beat V3 comfortably — that is the PASS. But **market alone beat the blend** on log loss, Brier and RPS.

So the genuinely proven component is the *market signal*, not the blending machinery. E7 will therefore report **market-alone as a fifth reference column** alongside A/B/C/D. It is not a new arm — it is the incumbent that any combination must actually beat to be worth its complexity.

---

## Finding 4 (detail) — E6's result cannot be reproduced on E7's set

E6 PASSed on **5,331** fixtures across **3** folds, including all of 2022/23. E7's eligible set is **3,479** fixtures across **2** folds, excluding 2022/23.

§20 requires reproducing E2 and E6 standalone closely enough to prove neither was accidentally altered.

- **E2 reproduction: exact match expected.** Same fixtures, same folds, same method — E7's Arm B should reproduce E2's numbers to floating-point.
- **E6 reproduction: exact match impossible.** Different fixture universe and fold count. E7's Arm C is a *re-run of E6's method on E7's set*, not a reproduction of E6's numbers.

I will report both explicitly and will not treat the expected E6 divergence as a control failure. The genuine control is Arm C's zero-state reduction to Arm A, which must hold to floating-point.

---

## Known consequences for the E7 decision

1. **§26 criterion 3 requires D > B and D > C.** Given Finding 2, D > B may be structurally hard. That is a real result, not a bug.
2. **Only 2 folds.** Thinner than E6's 3. Fold 1 trains the blend on 522 fixtures.
3. **n = 3,479 vs E6's 5,331.** A 35% smaller evaluation set, recency-skewed to 2023/24–2024/25.
4. **E6's selected `lr` may differ from [0.01, 0.02, 0.01]** because the folds differ. Expected, not a failure.
5. **Market-alone is the bar to beat**, not V3.

---

## Protected file integrity

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

---

## Recommendation

**Proceed to implementation** with:

- **Fixture set:** 3,479 (the clean intersection)
- **Folds:** E2's 2-fold structure
- **Arm B:** E2's blend, verbatim
- **Arm C:** E6's 91-column refit, verbatim
- **Arm D:** Arm C's output through Arm B's blend — no new hyperparameter
- **Reference column:** market alone, reported throughout
- **Reporting:** `w` per fold prominently, given Finding 2

The one decision needing your confirmation is **Arm D's construction**. It is the only form I can see that preserves both validated architectures without reinterpreting either — but it is a design choice, and §13 requires it be documented before fitting rather than assumed.

Say the word and I'll build it.
