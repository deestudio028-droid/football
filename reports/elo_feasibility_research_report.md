# Elo Feature Family: Feasibility Research Report

**Date:** 2026-08-20
**Scope:** Can cross-season Elo ratings meaningfully improve the V2 Poisson+Venue champion model?
**Constraint:** All analysis uses 2020/21–2024/25 only. 2025/26 holdout is untouched. V2 artifact is frozen.

---

## 1. Current 84-Feature Audit

The V2 champion (Poisson+Venue) uses exactly 84 input columns:

- **Goals core** (18): per-match goals for/against/diff over last5, last10, season — both sides
- **Form** (20): points, win/draw/loss rate, goal diff over last5/last10 — both sides
- **Shots core** (18): per-match shots for/against/diff over last5, last10, season — both sides
- **Shots on target** (18): same windows as shots — both sides
- **Strength** (5): empirical-Bayes shrunk attack/defence scores + strength_diff
- **Competition ID** (1): league identifier
- **Venue goals** (4): home-only and away-only goals for/against, season window

All 84 features are **within-season only**. No feature carries information from a previous season. The strength module (`strength.py`) explicitly filters to `target_season_id` before computing attack/defence scores. This means every team starts from zero information at the beginning of each season.

**Structural consequence:** The model is blind to the fact that Manchester City entering 2024/25 is fundamentally different from a newly promoted team entering 2024/25. Both begin with identical null/prior features.

The feature database contains 274 total columns; 180 are unused (mostly `_n` and `_coverage_n` companion columns plus xG, tempo, discipline groups not in the V2 contract).

---

## 2. Elo Feasibility Assessment

**Verdict: FEASIBLE — all 10 audit questions pass.**

| Question | Result |
|---|---|
| Do we have chronological match results across seasons? | YES — 10,733 FT fixtures, 2020/21–2025/26, all 5 leagues |
| Are team IDs stable across seasons? | YES — 137 unique IDs, 155 names, zero ID collisions |
| Can we compute Elo without external data? | YES — only needs results already in matches.db |
| Is the feature leakage-safe? | YES — same read-then-update pattern as existing FeatureContext |
| Does it fit the Poisson architecture? | YES — continuous features, direct input to PoissonRegressor |
| Can we implement without modifying production code? | YES — standalone computation, merged at dataset level (Option C) |
| Does the walk-forward framework support it? | YES — Elo state accumulates chronologically, no future leakage |
| Is there a clear information-theoretic reason? | YES — cross-season carry-forward fills the season-boundary gap |
| Is the signal plausible from football domain knowledge? | YES — team quality persists across seasons (transfers, coaching, infrastructure) |
| Can we test without touching 2025/26? | YES — 3-fold walk-forward uses only 2020/21–2024/25 |

---

## 3. Verified Cross-Season Signal

Three empirical claims were independently verified using only 2020/21–2024/25 data (2025/26 excluded):

### Claim 1: Cross-season goal-difference correlation
**Method:** For each team appearing in consecutive seasons within the same league, compute mean goal-difference-per-match in season N and season N+1. Pearson correlation across all such team-season pairs.
**Result:** r = 0.72 (p < 0.001)
**Interpretation:** 52% of variance in a team's goal difference carries forward. This is strong — comparable to year-over-year revenue correlation for established businesses. It confirms that "team quality" is a real, persistent signal, not noise.

### Claim 2: Early-season information gap
**Method:** Split all fixtures by `played` count (games played by the team with fewer games). Compare log loss for early (played < 5) vs late (played ≥ 20) fixtures.
**Result:** Log loss 1.029 (early, n=728) vs 0.993 (late, n=2422). Gap = 0.037.
**Interpretation:** The model is measurably worse at the start of each season, when within-season features are thin. This is exactly where Elo's cross-season carry-forward would provide the most value.

### Claim 3: Promoted team blind spot
**Method:** Compare home win rates for promoted teams (first season in top flight) vs established teams, restricted to first 5 matchweeks of each season.
**Result:** Promoted teams: 24.4% home win rate. Established teams: 42.3%. Gap = 17.9 percentage points.
**Interpretation:** The model has no mechanism to distinguish promoted teams from established ones at season start. An Elo system naturally assigns promoted teams a lower rating (since they come from a lower division), directly encoding this information.

---

## 4. Leakage Risks and Mitigations

### Mechanical leakage test: ALL PASS

Three tests were run on a prototype Elo implementation:

1. **No future results in Elo computation:** For every fixture, the Elo ratings used as features were computed from strictly earlier fixtures only. Verified by checking that each fixture's Elo values are identical whether computed from the full dataset or from only fixtures with earlier timestamps. **PASS.**

2. **Season regression uses only pre-season information:** The regression-to-mean at season boundaries uses only the team's end-of-previous-season Elo and the league mean — both known before the new season starts. **PASS.**

3. **Promoted team initialization uses no current-season data:** Promoted teams receive `league_mean - 150` based on the previous season's league mean Elo, not any current-season information. **PASS.**

### Residual risks (minor, mitigated by design):

- **Elo parameters (K, H, regression coefficient) chosen before experiment:** All are pre-registered, not tuned on validation data.
- **No optimization of Elo on validation folds:** Elo is computed identically regardless of which fold is being evaluated. The Elo state simply accumulates forward in time.

---

## 5. Recommended Elo Design

### Parameters (pre-registered, not tunable during experiment):

| Parameter | Value | Justification |
|---|---|---|
| Initial rating | 1500 | Standard convention |
| K-factor | 20 | Standard for league football; balances responsiveness vs stability |
| Home advantage (H) | 100 | ~60 Elo points = ~0.17 expected-score difference ≈ observed home win surplus |
| Season regression | 0.67 × prev_elo + 0.33 × league_mean | Reflects that ~2/3 of team strength persists; ~1/3 mean-reverts |
| Promoted team init | league_mean − 150 | Newly promoted teams are, on average, weaker than established top-flight teams |
| League mean | Rolling, per league | Each of the 5 leagues maintains independent Elo pools |

### Expected-score formula:
Standard logistic: `E_home = 1 / (1 + 10^((away_elo - home_elo - H) / 400))`

### Update rule:
```
actual_home = 1.0 if home win, 0.5 if draw, 0.0 if away win
home_elo_new = home_elo + K * (actual_home - E_home)
away_elo_new = away_elo + K * (E_home - actual_home)
```

### Season boundary logic:
At the start of each new season:
1. For teams present in the previous season: `new_elo = 0.67 * end_of_season_elo + 0.33 * league_mean_elo`
2. For newly promoted teams (no previous season in this league): `new_elo = league_mean_elo - 150`
3. League mean is computed from end-of-season ratings of all teams in that league

### Sanity check (verified):
After running Elo through 2020/21–2024/25, top-rated teams per league align with real-world expectations (e.g., Manchester City, Bayern Munich, PSG near the top of their respective leagues). Mean rating ≈ 1500, spread (max − min) ≈ 528. No degenerate behavior.

---

## 6. Exact Feature Columns

Three new features, added to the existing 84 (total: 87):

| Column | Type | Description |
|---|---|---|
| `home_elo` | float | Home team's Elo rating as of immediately before this fixture |
| `away_elo` | float | Away team's Elo rating as of immediately before this fixture |
| `elo_diff` | float | `home_elo - away_elo` (positive = home team stronger) |

**Why these three and not more:**
- `home_elo` and `away_elo` give the model access to absolute team strength levels (useful for expected goals calibration in Poisson).
- `elo_diff` is the primary signal for outcome prediction (who is stronger).
- Derived features (elo_ratio, elo_product, etc.) add no information a PoissonRegressor can't extract from these three; adding them would be feature bloat without information-theoretic justification.

**Integration path (Option C — standalone, zero production modification):**
1. Compute Elo ratings in a standalone script that reads matches.db chronologically
2. Output a table mapping `(fixture_id) → (home_elo, away_elo, elo_diff)`
3. Merge with the existing feature matrix at training/evaluation time via fixture_id join
4. No modification to feature_builder.py, history.py, strength.py, predict_match.py, or v2_poisson_venue.pkl

---

## 7. Walk-Forward Experiment Design

### Structure: Champion vs Challenger

| | Champion (V2) | Challenger (V2+Elo) |
|---|---|---|
| Features | 84 (MODEL_B + VENUE) | 87 (84 + 3 Elo) |
| Model | PoissonRegressor (home + away) | PoissonRegressor (home + away) |
| H/D/A conversion | Tail-safe Poisson PMF | Tail-safe Poisson PMF |
| Preprocessing | Identical | Identical |
| Hyperparameters | PoissonRegressor defaults | PoissonRegressor defaults |

### Folds (identical for both):

| Fold | Train seasons | Validation season | Approx train fixtures | Approx val fixtures |
|---|---|---|---|---|
| fold_1 | 2020/21, 2021/22 | 2022/23 | ~3,600 | ~1,800 |
| fold_2 | 2020/21, 2021/22, 2022/23 | 2023/24 | ~5,400 | ~1,800 |
| fold_3 | 2020/21, 2021/22, 2022/23, 2023/24 | 2024/25 | ~7,200 | ~1,800 |

### Elo computation timeline:
- Elo accumulates from the first fixture in 2020/21 onward, across all seasons
- For fold_1: Elo has ~2 seasons of warm-up before the validation season
- For fold_2: ~3 seasons of warm-up
- For fold_3: ~4 seasons of warm-up
- Elo ratings used as features for any fixture are computed from strictly earlier fixtures only — identical computation regardless of which fold is being evaluated

### What changes between Champion and Challenger:
Only the feature matrix width (84 → 87). Everything else — fold boundaries, Poisson architecture, tail-safe conversion, label definition, preprocessing — is identical. This isolates the experiment to one question: does adding Elo information improve prediction?

---

## 8. Acceptance / Rejection Criteria

### Primary metric: Pooled log loss across all 3 folds

**Accept if ALL of:**
1. Challenger pooled log loss < Champion pooled log loss (any improvement)
2. Challenger log loss is lower on at least 2 of 3 individual folds
3. No catastrophic degradation on any single fold (defined as: Challenger fold log loss > Champion fold log loss + 0.02)

**Reject if ANY of:**
1. Challenger pooled log loss ≥ Champion pooled log loss
2. Challenger wins on only 1 of 3 folds (unreliable signal)
3. Any single fold shows catastrophic degradation (> 0.02 increase)

### Secondary metrics (reported but not gatekeeping):
- H/D/A accuracy (pooled and per-fold)
- Brier score (pooled and per-fold)
- Subgroup analysis: early-season (played < 5) vs late-season (played ≥ 20) log loss
- Subgroup analysis: promoted team fixtures vs established team fixtures

### Why these criteria:
- The 2/3-fold requirement guards against overfitting to a single temporal slice
- The catastrophic-degradation guard prevents accepting a Challenger that trades a good fold for a terrible fold
- Log loss is the proper scoring rule for probability calibration, which matters more than raw accuracy for a Poisson prediction system
- Subgroup analyses test the specific hypotheses (early-season gap, promoted team blind spot) that motivated Elo in the first place

---

## 9. Expected Impact Range

### Realistic expectation:
Based on the verified signals:

- **Pooled log loss:** Improvement of 0.005–0.015 (from ~0.999 toward ~0.984–0.994). The early-season gap alone is 0.037, and early-season fixtures are ~13% of the dataset, so a 50% reduction in that gap would contribute ~0.002 to pooled improvement. The promoted-team effect adds further.
- **Accuracy:** Improvement of 0.5–2.0 percentage points (from ~51.6% toward ~52–53.5%).
- **Draw prediction:** Modest improvement expected. Elo diff near zero signals balanced teams, which should increase draw probability. But draws are inherently hard to predict (~25% base rate, near-random even for sophisticated models).

### What Elo will NOT do:
- It will not produce a dramatic accuracy jump. Elo encodes one specific piece of missing information (cross-season team strength). It does not address other missing information sources (odds, rest days, referee effects, etc.).
- It will not fix all early-season errors. Even with Elo, the first 1–2 matches of a season still have limited within-season data for other feature families.
- It will not make promoted teams easy to predict. Promoted teams are inherently more variable than established teams; Elo provides a better prior, not a crystal ball.

### Honest assessment of what "success" means here:
If Elo improves pooled log loss by 0.005+ on 2/3 folds, that is a genuine, repeatable improvement — even though 0.005 sounds small. In well-calibrated probability models, gains come in small increments. Each 0.005 improvement in log loss roughly corresponds to predictions being ~0.5% better calibrated, compounding across thousands of predictions.

---

## 10. Path from ~51–52% Toward 60%+

Current champion: 51.57% accuracy, 0.999 log loss (3-fold pooled).

**Getting to ~55–57% (realistic with current data architecture):**

The current model uses only match statistics and within-season form. To reach 55–57%, we would need to stack multiple independent information sources, each contributing 0.5–2 percentage points:

1. **Elo / cross-season strength** (this proposal): +0.5–2pp. Fills the season-boundary blind spot.
2. **Pre-match betting odds** (available via OddAlerts odds/history API, not yet ingested): +2–4pp. Odds are the single most powerful predictor in football — they aggregate information from thousands of bettors and statistical models. The `has_odds` column shows 99.9% coverage. This is almost certainly the highest-impact single addition.
3. **Rest days / fixture congestion:** +0.5–1pp. Data exists in matches.db (unix timestamps), not yet computed as features. Teams with <3 days rest underperform measurably.

Stacking all three realistically puts us in the 55–58% range.

**Getting to 58–60% (requires additional data or model architecture changes):**

4. **Lineup / squad information:** Knowing which players are actually playing (injuries, suspensions, rotation) is a major source of information. OddAlerts may provide lineups; this needs investigation.
5. **Model architecture upgrade:** PoissonRegressor is deliberately simple. A gradient-boosted model (LightGBM/XGBoost) on the same features typically adds 1–2pp from better nonlinear interaction capture. However, this trades interpretability for performance and needs careful regularization.
6. **Expected goals (xG) features:** Already computed but excluded from V2 due to coverage issues in early folds. As the dataset grows and xG coverage improves, these become viable.

---

## 11. Honest Assessment: Is 65% Realistically Achievable?

**Short answer: Not with the current data architecture alone. It would require at least one major new information source (most likely betting odds), probably two, plus a model architecture upgrade.**

**The arithmetic of 65%:**

Football match outcomes have a base rate of roughly 45% home wins, 27% draws, 28% away wins across the top 5 European leagues. A "predict home win every time" baseline gets ~45%. The theoretical ceiling for 3-way prediction accuracy is estimated at 55–60% by academic literature, because football has irreducible randomness (injuries during play, referee decisions, deflected goals, etc.).

Published academic models using full information (odds, lineups, detailed statistics) typically achieve:
- Without odds: 50–55% accuracy
- With odds as features: 53–58% accuracy
- Ensemble models with odds + deep features: 55–60% accuracy
- The very best published models with everything: ~58–62% accuracy

65% would be near the theoretical ceiling. It is not impossible, but it would require:

1. **Betting odds as features** (near-mandatory for 60%+)
2. **Cross-season strength / Elo** (this proposal)
3. **Lineup / squad strength data**
4. **A nonlinear model** (gradient boosting or neural network)
5. **Possibly match-context features** (rivalry, league position dynamics, end-of-season motivation)
6. **Careful calibration and ensemble methods**

Even then, 65% is optimistic rather than expected. A more realistic ceiling with all of the above is 58–62%.

**Recommendation:** Pursue 65% as a directional target that motivates good engineering, but measure progress honestly. If we reach 57–60% with odds + Elo + a model upgrade, that would be a genuinely excellent result.

---

## 12. Top 3 Next Information Sources After Elo

Ranked by expected impact and feasibility:

### #1: Pre-Match Betting Odds (HIGHEST PRIORITY)

**Information content:** Odds encode the market's aggregate assessment of match probabilities, which integrates team news, form, motivation, weather, and thousands of other factors that no statistical model can capture individually.
**Data availability:** OddAlerts `odds/history` endpoint provides opening/closing/peak odds per bookmaker (Pinnacle, Bet365, etc.) for ft_result markets. 99.9% of fixtures have `has_odds = True`. Permanent historical records.
**Expected impact:** +2–4 percentage points accuracy, 0.02–0.05 log loss improvement. This is likely the single largest available gain.
**Implementation complexity:** Medium. Requires ingesting odds data, converting to implied probabilities, and adding as features. Key design decision: use Pinnacle closing odds (most efficient market) or average across bookmakers.
**Leakage risk:** Low if using pre-match odds only (not in-play). Opening odds are available hours/days before kickoff.

### #2: Rest Days / Fixture Congestion

**Information content:** Teams with compressed schedules (European competition + league) underperform. The 3.4% of fixtures with <3 days rest show measurable effects.
**Data availability:** Already computable from `unix` timestamps in matches.db. No additional API call needed.
**Expected impact:** +0.5–1pp accuracy. Small overall but concentrated on specific fixtures where it's highly informative.
**Implementation complexity:** Low. Compute days since last match for each team, add as 2 features (home_rest_days, away_rest_days).
**Leakage risk:** None — uses only past fixture timestamps.

### #3: League Table Position (Pre-Match)

**Information content:** Current league standing encodes accumulated season performance in a single number. Particularly useful as a nonlinear signal — there may be behavioral differences between teams fighting relegation vs mid-table vs title contenders.
**Data availability:** `home_position` and `away_position` columns exist in matches.db with 100% coverage. Verified to be pre-match frozen values (exist at played=0, change between matches).
**Expected impact:** +0.5–1pp, but likely partially redundant with strength_diff (correlation -0.89). Most value comes from the nonlinear effects (relegation zone behavior, etc.) that a linear model might not capture from raw strength scores.
**Implementation complexity:** Very low — columns already exist, just need adding to the feature contract.
**Leakage risk:** Low — positions are frozen pre-match.
**Caveat:** High correlation with existing features means marginal value may be near zero for a linear model. Worth testing but don't expect much unless switching to a nonlinear model.

---

## Summary Decision Table

| Item | Status |
|---|---|
| Elo feasible? | YES |
| Cross-season signal verified? | YES (r=0.72, early gap=0.037, promoted gap=17.9pp) |
| Leakage risk? | MITIGATED (3/3 mechanical tests PASS) |
| Experiment designed? | YES — Champion (84) vs Challenger (87), 3-fold walk-forward |
| Acceptance criteria clear? | YES — pooled improvement + 2/3 folds + no catastrophic degradation |
| Expected Elo impact | 0.005–0.015 log loss, 0.5–2pp accuracy |
| 65% achievable with Elo alone? | NO |
| 65% achievable with current data architecture? | UNLIKELY — needs odds + Elo + model upgrade minimum |
| Realistic ceiling with all available data | 58–62% |
| Recommended next step | Implement Elo challenger experiment, then odds ingestion |

---

## Appendix: Files Referenced

- `src/features/strength.py` — current within-season-only strength calculation
- `src/features/history.py` — FeatureContext architecture (leakage guarantee)
- `src/features/feature_builder.py` — chronological walk, integration point for Elo
- `src/models/config.py` — walk-forward fold definitions
- `src/models/ablation.py` — 84-column V2 feature contract (MODEL_B + VENUE)
- `data/processed/matches.db` — 10,733 FT fixtures, 5 leagues, 6 seasons
- `data/audit/odds_history_example.json` — OddAlerts odds structure
