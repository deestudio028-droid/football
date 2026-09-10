# E6 — Phase 1 Audit Report (pre-implementation)

**Date:** 2026-08-20
**Status:** Audit only. No implementation code written. No production file touched.
**Per the execution rule:** this report answers the six required questions before any experiment is built.

---

## VERDICT UP FRONT

**E6 is NOT redundant. Proceed to implementation.**

This is a different conclusion from E3–E5, and it rests on a measurement rather than an argument: an online attack/defence state carries **7.7–10.1% variance that all 87 V3 features cannot explain**, and that residual translates into a **+0.001507 pooled out-of-sample log-loss gain improving 3/3 folds and 5/5 leagues** — roughly **25× E5's effect**, and the first broad, sign-consistent signal in this experiment series.

Two caveats stated up front: the measurement uses an IRLS GLM, not sklearn's `PoissonRegressor` (unavailable here), and a learning rate must be tuned, which introduces a hyperparameter E5 did not have.

---

## Question 1 — What does V3 already contain?

All 87 features, classified:

| Group | Count | Columns | Nature |
|---|---|---|---|
| Goals for/against/diff | 18 | 1–18 | Rolling last5 / last10 / season, home & away |
| Form (points, W/D/L rates, GD) | 20 | 19–38 | Rolling last5 / last10 |
| **Attack/defence strength** | **5** | **39–43** | **Season-to-date, shrunk** |
| League identifier | 1 | 44 | `competition_id` |
| Shots for/against/diff | 18 | 45–62 | Rolling last5 / last10 / season |
| Shots-on-target for/against/diff | 18 | 63–80 | Rolling last5 / last10 / season |
| Venue splits | 4 | 81–84 | Season, home-at-home / away-at-away |
| Causal Elo | 3 | 85–87 | Recursive, opponent-adjusted, cross-season |

**Answering §3's checklist directly:**

| Item | Present in V3? |
|---|---|
| Goals scored rolling averages | **Yes** — last5, last10, season |
| Goals conceded rolling averages | **Yes** — last5, last10, season |
| Home/away venue splits | **Yes** — columns 81–84 |
| Multi-window form | **Yes** — last5 and last10 throughout |
| Opponent-adjusted strength | **Partially** — via Elo only (see Q2) |
| Scoring/conceding rates | **Yes** — extensively |

---

## Question 2 — Is online Attack/Defense genuinely new?

### V3's existing `*_attack_strength_score` / `*_defence_strength_score`

From `src/features/strength.py`, these are **empirical-Bayes shrunk season-to-date rates**:

```
attack  = (Σ goals_for_season    + k · league_mean) / (n_season + k)
defence = (Σ goals_against_season + k · league_mean) / (n_season + k)      k = SHRINKAGE_K = 5
```

Three properties matter:

1. **Not recursive.** A plain cumulative mean over the season, not an incrementally updated state.
2. **Not opponent-adjusted.** The module's own docstring: *"No iterative opponent-adjusted rating — deliberately deferred, see spec §4 for why."*
3. **Season-resetting.** `_season_matches()` filters to `target_season_id`, so the state restarts every August.

`docs/PHASE2_FEATURE_ENGINEERING_REPORT.md` records the deferral explicitly: *"Full iterative opponent-adjusted rating (Elo-style) — Deferred to v1.1+ … This can be revisited in v1.1+ once there's a backtest to compare it against."*

**We now have that backtest.** E6 is precisely the deferred work.

### But E1 already delivered part of it

Causal Elo **is** an iterative opponent-adjusted rating: recursive, opponent-aware, cross-season persistent. So the deferral was partly discharged by E1.

**What remains genuinely new is the decomposition.** Elo is one dimension — it cannot distinguish a team that scores 3 and concedes 2 from one that scores 1 and concedes 0. Both may carry identical Elo. An online attack/defence state is two dimensions, opponent-adjusted, and cross-season persistent. That combination exists nowhere in V3:

| Property | V3 strength scores | V3 Elo | Online A/D |
|---|---|---|---|
| Recursive / online | No | Yes | **Yes** |
| Opponent-adjusted | No | Yes | **Yes** |
| Cross-season persistent | No | Yes | **Yes** |
| Separates attack from defence | Yes | **No** | **Yes** |

### The measurement

I built a candidate state (Q3) and regressed each component on **all 87 V3 features**:

| State | R² vs V3(87) | Residual std | State std | **Unexplained** |
|---|---|---|---|---|
| `A_home` | 0.9234 | 0.06217 | 0.22456 | **7.7%** |
| `D_home` | 0.8992 | 0.05672 | 0.17866 | **10.1%** |
| `A_away` | 0.9221 | 0.06259 | 0.22421 | **7.8%** |
| `D_away` | 0.8987 | 0.05691 | 0.17883 | **10.1%** |

Largely redundant — 90%+ explained — but **not perfectly so**. Correlations with the nearest analogues:

| Pair | corr |
|---|---|
| `A_home` vs `home_attack_strength_score` | +0.8221 |
| `D_home` vs `home_defence_strength_score` | −0.7744 |
| `A_home` vs `home_elo` | +0.8885 |
| `D_home` vs `home_elo` | +0.8527 |
| **A/D net differential vs `elo_diff`** | **+0.9857** |

That last figure is the one to take seriously. The *net* differential is 98.6% correlated with `elo_diff` — as theory predicts, since both estimate the same opponent-adjusted quantity. **If E6 collapsed the state to a single differential it would be almost pure duplication of Elo.** The information lives in the *separation* of attack from defence, which is exactly the dimension Elo lacks.

**This directly shaped the feature contract in Q5.**

---

## Question 3 — Proposed mathematical formulation

**Derived from the project's own architecture, not copied.** V3's goal models are Poisson GLMs with a log link. The natural online state is a stochastic gradient step on *that same likelihood*.

### State

Every team carries two scalars on the log scale: `A[t]` (attack) and `D[t]` (defence).

### Pre-match rates

```
lambda_home = mu_home · exp( A[h] − D[a] )
lambda_away = mu_away · exp( A[a] − D[h] )
```

`mu_home`, `mu_away` are league baseline goal rates estimated from **training fixtures only**.

### Update (after the outcome is observed)

For Poisson with log link, the exact score functions are:

```
∂LL/∂A[h] =  (g_h − lambda_home)          ∂LL/∂D[a] = −(g_h − lambda_home)
∂LL/∂A[a] =  (g_a − lambda_away)          ∂LL/∂D[h] = −(g_a − lambda_away)
```

giving the update rule:

```
A[h] ← A[h] + lr · (g_h − lambda_home)        D[a] ← D[a] − lr · (g_h − lambda_home)
A[a] ← A[a] + lr · (g_a − lambda_away)        D[h] ← D[h] − lr · (g_a − lambda_away)
```

This is principled rather than arbitrary: it is gradient ascent on the exact likelihood V3 already optimises. It is opponent-adjusted by construction — the error term depends on the opponent's current rating.

### Update timing (§6 sequence, enforced)

Two-pass per timestamp, mirroring the E1 Elo engine:

1. **Pass 1** — read pre-match `A`/`D` for every fixture at timestamp T
2. **Pass 2** — apply updates from outcomes at T

Simultaneous fixtures therefore cannot leak into one another.

### Initialization (§13)

All teams start at `A = 0.0`, `D = 0.0` (multiplicatively neutral, since `exp(0) = 1`). Unseen and promoted teams enter at the same neutral state — no future information is consulted. Values are clipped to ±1.5 on the log scale for numerical safety.

### Cross-season persistence (§22)

**Continuous.** No season reset. This matches the E1 Elo convention (`MEAN_REVERSION = 0.0`) and is the deliberate contrast with V3's season-resetting strength scores. The harness will verify continuity rather than assume it.

### Home/away structure (§7)

**Team-global states, not four separate venue-split states.** Home advantage enters through `mu_home` vs `mu_away`, and V3 already carries four venue features (81–84) plus the `+100` inside `elo_diff`. Creating separate home-attack and away-attack states would quadruple the parameter count, halve the data per state, and re-introduce home advantage a fourth time. §7 warns against exactly this.

### Observed state ranges (real data, 8,983 fixtures)

| lr | A range | D range | Finite |
|---|---|---|---|
| 0.01 | [−0.369, +0.695] | [−0.375, +0.445] | Yes |
| 0.02 | [−0.488, +0.828] | [−0.509, +0.600] | Yes |
| 0.05 | [−0.753, +1.080] | [−0.783, +0.921] | Yes |

Stable, bounded, no explosion or collapse at any candidate rate.

---

## Question 4 — Does any parameter need tuning?

**Yes — the learning rate `lr`. This is E6's main methodological weakness and I am flagging it prominently.**

E5 needed no tuning; E6 does. `lr` controls how fast the state tracks recent form, so it is functionally a recency parameter — and E4 already found recency effects in this data to be weak and its selection unstable across folds.

**Pre-registered grid (fixed before any test-fold evaluation):**

```
lr ∈ {0.005, 0.01, 0.02, 0.035, 0.05}
```

Five values, geometric-ish, spanning slow to fast tracking. Deliberately small per §12's "prefer a small pre-registered grid."

**Selection method:** nested temporal validation inside each outer training period — the identical machinery built and tested in E4 (`build_inner_splits` / `select_xi`), reused rather than reinvented. Train on `(s1..sj)`, validate on `s(j+1)`, choose `lr` by mean inner-validation log loss, freeze, refit on the full outer training period, then predict the outer test fold.

**Stability will be reported per fold and is a PASS criterion.** If `lr` proves as unstable as E4's half-life, that alone should push E6 toward INCONCLUSIVE.

---

## Question 5 — Exact E6 feature contract

```
V3 = 87 features
E6 = 91 features = V3[0:87] + (A_home, D_home, A_away, D_away)
```

**Four raw states, not the two differentials.** Justification, given Q2's measurement:

- The net differential correlates **+0.9857** with `elo_diff`. Adding it alone would be near-duplication of a feature V3 already has.
- The unexplained variance sits in the *separation* of attack from defence — precisely what the four raw components preserve and the differential discards.
- A linear model given four raw features can still recover the differential form by learning equal-and-opposite coefficients; the reverse is not possible. Supplying the differential would impose an untested constraint.
- §9 forbids adding *both* raw and differential forms without justification. Only the raw four are added.

`E6[:87]` is byte-identical to the V3 contract, same order. V3's features are not reordered or altered.

---

## Question 6 — Why the formulation is causally valid

| Requirement | How it is met |
|---|---|
| State depends only on fixtures strictly before T | Two-pass per timestamp; Pass 1 reads before Pass 2 writes |
| Fixture's own outcome cannot affect its own state | Its update happens in Pass 2, after its features are read |
| Future fixtures cannot affect earlier state | Updates are applied in strict chronological order |
| Simultaneous fixtures cannot cross-contaminate | Grouped by identical timestamp; all reads precede all writes |
| `mu_home` / `mu_away` from training only | Estimated inside the fold's training period |
| `lr` from training only | Nested temporal validation, no outer-test exposure |
| Deterministic | No RNG; pure function of ordered history |

The design is a direct structural copy of the E1 causal Elo engine, which passed all 14 of its invariant tests including outcome insulation, truncation invariance and determinism.

---

## Advance measurement — how large is the effect?

Two walk-forward measurements on the real folds, using an IRLS Poisson GLM (validated in E5 against known synthetic coefficients to ±0.003).

### Against Elo alone

| Test season | A (Elo only) | B (Elo + A/D) | Delta |
|---|---|---|---|
| 2022/23 | 0.991910 | 0.993043 | −0.001134 |
| 2023/24 | 0.985681 | 0.980033 | **+0.005647** |
| 2024/25 | 0.990047 | 0.985781 | **+0.004266** |
| **Pooled** | **0.989250** | **0.986381** | **+0.002869** |

5/5 leagues improved.

### Against the full 87-feature V3 — the decisive test

| Test season | A (V3, 87) | B (V3 + A/D, 91) | Delta |
|---|---|---|---|
| 2022/23 | 0.994944 | 0.994529 | **+0.000415** |
| 2023/24 | 0.981837 | 0.979836 | **+0.002001** |
| 2024/25 | 0.984035 | 0.981882 | **+0.002153** |
| **Pooled** | **0.987051** | **0.985544** | **+0.001507** |

Brier: 0.588066 → 0.587083 (**+0.000983**).

**League breakdown — all five positive:**

| League | n | Delta |
|---|---|---|
| La Liga | 1,140 | +0.002433 |
| Serie A | 1,141 | +0.002233 |
| Ligue 1 | 992 | +0.001182 |
| Premier League | 1,140 | +0.000872 |
| Bundesliga | 918 | +0.000596 |
| **Leagues improved** | | **5/5** |

Mean |ΔP| = 0.005033; top prediction changes on 1.28% of fixtures.

### Why this is different from E3–E5

| Experiment | Pooled delta | Folds improved | Leagues improved |
|---|---|---|---|
| E3 Dixon-Coles (proxy) | −0.000165 | 0/3 | 2/5 |
| E4 Time decay (proxy) | +0.000930 | 2/3 | 2/5 |
| E5 Quadratic Elo (GLM) | +0.000060 | 1/3 | 2/5 |
| **E6 Online A/D (GLM)** | **+0.001507** | **3/3** | **5/5** |

E6 is the first with a consistent sign across every fold and every league, and roughly **25× E5's magnitude**. It is also the first where the 2-of-5 league split — which I showed in E5 was noise from slicing a null effect — does not appear.

---

## Honest caveats

1. **This is an IRLS GLM, not sklearn's `PoissonRegressor`.** Same likelihood, same ridge penalty, same log link, but not the same solver or preprocessor. The real A/B must be run locally.
2. **A hyperparameter now exists.** E5 had none. If `lr` selection is unstable across folds, criterion 5 should fail.
3. **The state is 90%+ explained by V3.** The gain comes from ~8–10% residual variance. It is real and broad, but it is a small effect on top of a mostly-redundant feature.
4. **Only 3 folds** — the standing structural limit across E3–E6.
5. **`lr = 0.02` was used for the redundancy measurement** as a mid-grid value. The experiment will select it causally per fold.

---

## Protected file integrity

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

No production file was read except read-only, and none was modified. No implementation code has been written.

---

## Recommendation

**Proceed to E6 implementation** with the formulation in Q3, the 91-column contract in Q5, and the pre-registered `lr` grid in Q4.

Awaiting your go-ahead before writing `online_attack_defense.py`, the test suite, and the harness — per the execution rule that the audit be confirmed internally consistent first.
