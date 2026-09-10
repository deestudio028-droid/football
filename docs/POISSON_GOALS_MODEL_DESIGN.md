# Poisson Goals Model — Design Document (V2 candidate)

**Status:** DESIGN ONLY. No code written, no production file touched, nothing trained.
**Date:** 2026-08-19
**Supersedes:** nothing. V1 (`v1_logreg.pkl`, 80-column contract) remains frozen and authoritative.

---

## 1. Why this, and why now

Two independent probes closed off the cheaper options:

| Probe | Question | Result |
|---|---|---|
| STEP 8 | Do the existing 79 features contain signal a nonlinear model could extract? | **No.** Log loss 0.9994 → 1.0891, 0/3 folds improving, Draw-vs-rest AUC 0.5361 → 0.5349 |
| STEP 9 | Does the source data contain new pre-match information worth adding? | **No strong candidate.** 0 STRONG, 2 POSSIBLE (`unix`, `season_progress`), 25 fields excluded |

STEP 8's most decision-relevant finding: the nonlinear probe raised the Draw argmax share from 2.4% to 13.1% **while its Draw discrimination got slightly worse**. It predicted more Draws without knowing which matches were Draws, and log loss paid for it.

That rules out the obvious fix. What remains is not a feature or a hyperparameter — it is the **model form**.

### The structural argument

V1 is a 3-class classifier. Its mean P(D) is 0.2486 against a 0.2519 base rate — the *aggregate* is nearly perfect. The failure is in the *spread*: P(D) has a standard deviation of only 0.0598 and exceeds ⅓ in just 7.1% of matches. A class can only win the argmax if its probability exceeds ⅓, so Draw is structurally suppressed almost everywhere.

A goals model does not have this problem by construction. Draw probability is not a third competing logit — it is `Σ_k P(home=k) · P(away=k)`, which varies naturally with the expected goal totals. For evenly matched low-scoring fixtures it rises well above ⅓; for mismatched high-scoring fixtures it falls. The spread comes for free.

### It also delivers what V1 cannot

The client asked for six things. V1 produces two of them.

| Client requirement | V1 | Poisson goals model |
|---|---|---|
| 1. Which team more likely to win | YES | YES |
| 2. Draw probability | YES | YES |
| 3. Expected goals both teams | **NO** | YES (λ_home, λ_away) |
| 4. Approximate goals per team | **NO** | YES |
| 5. Likely scorelines | **NO** | YES (full score matrix) |
| 6. Confidence / probability distributions | partial | YES |

This is also the methodology the project brief already specifies (stage 4: Poisson simulation). V1 was a reasonable Phase-3 baseline; it was never the target architecture.

---

## 2. Evidence base — measured, not assumed

All figures below were measured read-only on **training seasons only** (2020/21–2024/25, n = 8,983). The 2025/26 final-test season was excluded from every calculation.

```
home goals   mean 1.5345   var 1.7102   var/mean 1.1145
away goals   mean 1.2826   var 1.3749   var/mean 1.0719
home advantage           +0.2518 goals   (ratio 1.1963)
corr(home goals, away goals)   -0.0825
observed draw rate                        0.2539
independent-Poisson draw rate at league means  0.2497
gap                                      +0.0043
```

Four things follow, and each one shapes a design decision:

**(a) Goals are close to Poisson, mildly overdispersed.** Variance/mean ratios of 1.11 and 1.07 are near 1.0. Poisson is defensible as the starting distribution; the mild excess is a reason to *test* Negative Binomial later, not to start there.

**(b) Independent Poisson already reproduces the draw rate to within 0.4 percentage points.** This is the single most important number here. The classic argument for a Dixon–Coles low-score correction is that independent Poisson *under-predicts* draws — **in this dataset it does not**. A DC correction is therefore NOT justified up front. It becomes a validated experiment, not a default.

**(c) Score-level deviations are small but real.** 0-0 runs 7.1% above independent Poisson, 1-1 runs 5.2% above, 2-1 runs 7.5% below. Modest, and exactly the region a DC correction targets — which is why it stays on the candidate list even though (b) removes the aggregate motivation.

**(d) Home and away goals are slightly negatively correlated (−0.0825).** The independence assumption is mildly violated. This motivates bivariate Poisson as a candidate, and means independence must be stated as an assumption rather than glossed over.

---

## 3. Architecture

### 3.1 Target change

| | V1 | V2 candidate |
|---|---|---|
| Target | `label_result` ∈ {H, D, A} | `label_home_goals`, `label_away_goals` (both integers) |
| Estimator | one multinomial LogisticRegression | two Poisson regressions (or one with a home/away indicator) |
| Output | 3 probabilities | λ_home, λ_away → full score matrix → everything else |

Both target columns already exist in `features.db`. **No new data ingestion is required.**

### 3.2 Pipeline

```
79 pre-match features + competition_id
        │
        ├── Poisson GLM (log link) ──► λ_home
        └── Poisson GLM (log link) ──► λ_away
                    │
                    ▼
        score matrix  P(i,j) = P(home=i) · P(away=j),  i,j ∈ 0..10
                    │
    ┌───────────────┼────────────────┬──────────────────┐
    ▼               ▼                ▼                  ▼
P(H)=Σ_{i>j}   P(D)=Σ_{i=j}     P(A)=Σ_{i<j}    top-N scorelines
                    │
              expected goals = λ_home, λ_away
```

The truncation at 10 goals per side captures >99.99% of probability mass at these λ values; the residual is renormalised, and the renormalisation constant gets asserted to be within tolerance of 1.

### 3.3 What is deliberately reused unchanged

- The same 79 pre-match features and `competition_id`
- The same three walk-forward folds
- The same `compute-then-record` leakage discipline
- The same evaluation module (`log_loss`, `brier`, `confusion_matrix`)
- The same `CLASS_ORDER = ['H','D','A']` and explicit probability-column mapping

**Holding the information constant is the point.** If V2 beats V1 on the same features and folds, the gain is attributable to the model form and nothing else. Changing features and form together would make the result uninterpretable.

---

## 4. Assumptions — stated explicitly

| # | Assumption | Status | Risk if wrong |
|---|---|---|---|
| A1 | Goal counts are approximately Poisson | **Verified** (var/mean 1.11, 1.07) | Mild overdispersion → slightly over-confident tails |
| A2 | Home and away goals are conditionally independent given λ | **Violated, mildly** (r = −0.0825) | Score-matrix probabilities biased near the diagonal |
| A3 | A log link with the existing features can estimate λ usefully | **Inferred, untested** | The whole design fails; this is the primary risk |
| A4 | Truncating at 10 goals loses negligible mass | **Verified by construction**, asserted at runtime | None if asserted |
| A5 | Draw probability spread will exceed V1's SD of 0.0598 | **Inferred, untested** | The structural argument in §1 does not materialise |

A3 and A5 are the load-bearing assumptions and **both are untested**. This design should be read as a well-motivated hypothesis, not a predicted improvement.

---

## 5. Risks, honestly

**The largest risk: λ estimation inherits the same weakness.** λ_home and λ_away are predicted from the *same 79 features* that STEP 8 showed to be near their ceiling. If those features cannot separate evenly-matched fixtures, the Poisson model will produce sensible-looking but poorly-discriminating λ values, and P(D) will be better-*shaped* without being better-*informed*. STEP 8's warning applies with full force: **a higher Draw rate is not an improvement unless discrimination improves too.**

**Second risk: V2 may score worse on log loss.** A 3-class classifier optimises the H/D/A objective directly. A goals model optimises goal counts and derives H/D/A as a by-product — a harder route to the same target. It is entirely possible that V2 delivers requirements 3–5, produces far more sensible Draw probabilities, and still loses on pooled log loss. **That trade-off is the client's decision, not a technical one**, and it must be surfaced explicitly rather than resolved by picking whichever metric flatters the result.

**Third risk: scoreline outputs invite overconfidence.** A table of "likely scorelines" reads as more precise than it is. Even a well-calibrated model puts only ~12% on the single most likely score. Any client-facing output must show the probability alongside the score, never the score alone.

**Fourth risk: scope creep.** Dixon–Coles, bivariate Poisson, Negative Binomial, time-decay weighting and team-level random effects are all defensible extensions. Adding them simultaneously would make the result uninterpretable and reintroduce exactly the tuning-without-evidence problem this project has avoided so far.

---

## 6. Validation plan

Non-negotiable, and identical to what V1 was held to.

**Primary metrics** — pooled over the same three walk-forward folds, V2 vs V1 head to head:

1. Log loss on H/D/A (the decisive metric)
2. Multiclass Brier
3. Fold-level deltas and their range — never a standard deviation from three folds, and never a significance claim

**Goals-specific metrics** (V2 only, no V1 comparison exists):

4. MAE and RMSE on home goals and away goals
5. Exact-scoreline accuracy vs the base rate of the most common score
6. Calibration of λ: predicted vs observed mean goals by λ decile

**Draw-specific** — carried forward verbatim from STEP 8 so the two runs are directly comparable:

7. Draw-vs-rest AUC — **this is the one that matters.** Draw argmax share rising without AUC rising is the STEP 8 failure mode repeating.
8. Mean/median/SD of P(D), P(D)>⅓ share, actual-Draw rank-1 share

**Baselines V2 must beat to be interesting:**

- V1 on log loss and Brier
- A constant-λ model (league means, no features) — proves the features contribute anything at all
- The class-prior baseline

**Pre-registered decision rule, fixed before any fitting:**

> V2 is classified BETTER only if pooled log loss improves **and** at least 2/3 folds improve **and** Brier does not materially contradict. Any other pattern is INCONCLUSIVE. Draw metrics are reported but do not on their own promote V2.

**2025/26 is not touched.** It has been used twice and is spent as an evaluation set. Final confirmation waits for 2026/27 data.

---

## 7. Build order

Each step is separately reviewable, and any of them can end the project early without wasted work.

| # | Step | Exit criterion |
|---|---|---|
| 1 | Target audit — verify `label_home_goals`/`label_away_goals` are complete, non-negative, and consistent with `label_result` | zero inconsistencies |
| 2 | Constant-λ baseline (league means only) | reproduces the 0.2497 draw rate |
| 3 | Poisson GLM for λ_home and λ_away on the 79 features | converges; λ in a sane range |
| 4 | Score matrix + H/D/A derivation | rows sum to 1 within 1e-9; independent re-derivation agrees |
| 5 | Walk-forward validation vs V1 | pre-registered rule in §6 |
| 6 | Draw diagnostics vs STEP 8 | AUC compared, not just argmax share |
| 7 | Written diagnosis | BETTER / WORSE / INCONCLUSIVE, no tuning after seeing results |

Dixon–Coles, bivariate Poisson, Negative Binomial and time-decay are **explicitly out of scope** for this pass. They become candidates only after step 7, and only one at a time.

---

## 8. What this does not promise

It does not promise 85–95% accuracy. Nothing will. Bookmakers with far richer data operate near 55%, and V1 currently sits at 47.5% on a 40-match holdout sample.

What this design does offer is (a) a model form whose Draw probabilities are structurally capable of varying, (b) the three client deliverables V1 cannot produce at all, and (c) a validation plan that will honestly report a negative result if that is what the data says — as the last two probes already did.

---

## 9. Integrity

Nothing in this document has been implemented. V1 remains frozen:

```
features.db      e7ebe7fc07040a5927683c35b6371e63
matches.db       fdeed042096fa1c851aaee6c84995247
v1_logreg.pkl    5e504427712b35778bb8a62a8496c7cd
train.py         21425459195311492f49e73f5ae38fe0
LOCKED_INPUTS    13/13 unchanged
```

The measurements in §2 were read-only, on training seasons only, and wrote nothing.
