# Phase 3 — V1 Model Spec (finalized contract)

**Status: FROZEN.** This is the finalized V1 contract, written after real implementation and evaluation (`PHASE3_MODEL_DEVELOPMENT_REPORT.md`), mirroring how `PHASE2_FEATURE_SPEC.md` relates to `PHASE2_FEATURE_ENGINEERING_REPORT.md` in this project. Any change to anything below is a new model version, not a V1 patch.

## Model identity

- `MODEL_VERSION = "v1.0"` (matches `config.MODEL_VERSION`)
- `REQUIRED_FEATURE_VERSION = "v1.0"` (matches `config.REQUIRED_FEATURE_VERSION`) — hard-fails if `feature_rows` ever contains more than one `feature_version`, or a version other than `"v1.0"`.
- Algorithm: **scikit-learn `LogisticRegression`**, trained on a **global pool of all 5 leagues** (English Premier League, La Liga, Bundesliga, Serie A, Ligue 1), with `competition_id` supplied as a feature — not 5 separate per-league models.
- Calibration: **none**. Raw `predict_proba()` output is used as-is.

## Target

3-class classification: **H** (home win), **D** (draw), **A** (away win) — `ALLOWED_LABELS = ("H", "D", "A")`, class order fixed as `CLASS_ORDER = ["H", "D", "A"]` everywhere probabilities are reported.

## Training data

- Source: `data/processed/features.db`, table `feature_rows`, filtered to `label_result IS NOT NULL`.
- **10,734 labeled rows** as of this freeze (6 complete seasons, 2020/21–2025/26, 5 leagues).
- Fixture **`420450481`** (Nantes vs Toulouse, Ligue 1 2025/26, ABANDONED) is permanently excluded from training and evaluation — it has no valid label and must never be coerced into one.
- Row filter is exactly `WHERE label_result IS NOT NULL` — never a blanket `dropna()` across feature columns (a feature column may legitimately be null for a row that still has a valid label).

## Feature contract — exactly 15 columns

No other column from `feature_rows` may be added to V1's `X` without a version bump. Verified to exist verbatim; any future absence must fail loudly, never be silently substituted.

| # | Column | Description |
|---|---|---|
| 1 | `home_goals_for_per_match_season` | Home team's goals scored per match, current season to date |
| 2 | `home_goals_against_per_match_season` | Home team's goals conceded per match, current season to date |
| 3 | `away_goals_for_per_match_season` | Away team's goals scored per match, current season to date |
| 4 | `away_goals_against_per_match_season` | Away team's goals conceded per match, current season to date |
| 5 | `home_goals_for_per_match_last5` | Home team's goals scored per match, last 5 matches |
| 6 | `away_goals_for_per_match_last5` | Away team's goals scored per match, last 5 matches |
| 7 | `home_points_last5` | Home team's points won, last 5 matches |
| 8 | `away_points_last5` | Away team's points won, last 5 matches |
| 9 | `home_attack_strength_score` | Home team's shrinkage-adjusted attacking strength |
| 10 | `home_defence_strength_score` | Home team's shrinkage-adjusted defensive strength |
| 11 | `away_attack_strength_score` | Away team's shrinkage-adjusted attacking strength |
| 12 | `away_defence_strength_score` | Away team's shrinkage-adjusted defensive strength |
| 13 | `strength_diff` | Combined home-vs-away strength differential |
| 14 | `league_home_advantage_season` | League-level home-advantage adjustment, current season |
| 15 | `competition_id` | League identity (categorical: `{200, 419, 423, 477, 499}`) |

**Explicitly excluded from V1**: xG (all 54 xG-related columns), shots, shots on target, possession, corners, fouls, cards, events, lineups, and every other Phase 2 feature not listed above. This exclusion is structural (`config.APPROVED_FEATURE_COLUMNS_V1`, enforced by `run_experiments._select_approved_features()`), not incidental.

**Also excluded from `X` regardless of feature set** (`config.X_EXCLUDED_COLUMNS`): `label_home_goals`, `label_away_goals`, `label_result`, `fixture_id`, `generated_at`, `feature_version`, `season_id`, `unix`, `home_id`, `away_id`.

## Preprocessing contract

- **Imputation**: `SimpleImputer(strategy="median")`, fit on the training partition only.
- **Scaling**: `StandardScaler`, fit on the training partition only.
- **`competition_id` encoding**: one-hot against categories observed in the training partition; an unseen category at inference time produces an all-zero encoding (never invents a new column, never errors).
- No zero-filling of any feature. No blanket `dropna()`.

## Validation protocol (frozen)

Walk-forward, 3 folds, never randomized:

| Fold | Train | Validate |
|---|---|---|
| fold_1 | 2020/21 – 2021/22 | 2022/23 |
| fold_2 | 2020/21 – 2022/23 | 2023/24 |
| fold_3 | 2020/21 – 2023/24 | 2024/25 |
| **Final test (locked)** | 2020/21 – 2024/25 | **2025/26** |

Model and calibration selection use only the mean across folds 1–3. 2025/26 is used exactly once, after every selection decision is frozen.

## Selection rule (as applied)

Primary: lowest mean validation log loss across folds 1–3. Brier score is the tie-breaker within 0.005 log loss. Disqualifying condition: a candidate producing malformed probabilities (not summing to ~1, containing NaN/inf, or fewer than 3 classes) is rejected regardless of its metrics.

**Result of applying this rule: LogisticRegression selected (mean validation log loss 1.0043186935714623) over HistGradientBoostingClassifier (mean validation log loss 1.0891458453243126).**

## Locked V1 performance (final, 2025/26, untouched during selection)

| Metric | Value |
|---|---|
| Log loss | `1.012643607907126` |
| Brier score | `0.6059500599589396` |
| Accuracy | `0.500856653340948` |
| Macro-F1 | `0.3712753915810083` |
| Balanced accuracy | `0.4321797560315324` |
| ECE | `0.01731559581035984` |

These numbers are the V1 contract's performance baseline. Any future change to features, model, or calibration must be compared against them, not silently assumed to be an improvement.

## Output contract

- `predict_proba`-style output: an array/row of exactly 3 floats, in `CLASS_ORDER = ["H", "D", "A"]` order, each finite, non-negative, summing to ~1 (`atol=1e-6`) — enforced by `evaluate.validate_probabilities()` everywhere probabilities are produced or consumed.
- No calibration transform is applied to this output in V1.

## What this spec does NOT cover (explicitly deferred)

- Feature ablation (Model A/B/C/D: goals+form / +shots / +xG / full feature set) — designed in the Phase 3 design report §12, not executed.
- Hyperparameter tuning of either candidate.
- xG-inclusive or shots-inclusive model variants.
- Poisson / Monte Carlo match simulation, expected-goals-to-scoreline conversion.
- Betting logic or recommendations.
- A production prediction API or UI.

None of the above may begin under this spec; each requires its own explicit scope and, if it changes any contract item above, a new model version.
