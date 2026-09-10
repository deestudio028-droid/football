# 02 — Information Clock & Causal Leakage Audit

## 1. Information-Clock Principle

For every feature $f_{i, t}$ computed for fixture $i$ scheduled at kickoff time $t$:
$$t_{\text{information}}(f_{i, t}) \le t_{\text{kickoff}}(i)$$

No feature or parameter may access:
- The actual match outcome $y_i \in \{H, D, A\}$
- Actual goals scored $g_{i, H}, g_{i, A}$
- Post-match statistics, cards, or substitutions
- Future matches $j > i$ or future seasons
- Prospective 2026 live match results

---

## 2. Feature-by-Feature Causal Timing Audit

| Feature Group | Features | Computation Timing | Causal Verification |
|:---|:---|:---|:---:|
| **Rolling Match Form** | `home_points_last5`, `away_points_last5`, `home_goals_for_per_match_last5`, etc. | Computed strictly over historical finished matches with $t_{\text{finish}} < t_{\text{kickoff}}$ | **PASS (Causal)** |
| **Season-to-Date Stats**| `home_goals_for_per_match_season`, `league_home_advantage_season`, etc. | Cumulative season stats strictly prior to match $i$ | **PASS (Causal)** |
| **Team Strength Scores**| `home_attack_strength_score`, `away_defence_strength_score`, `strength_diff` | Pre-match relative strength ratios | **PASS (Causal)** |
| **E1 Causal Elo Engine**| `home_elo`, `away_elo`, `elo_diff` | Updated sequentially fixture-by-fixture using historical results prior to kickoff; initial rating 1500, $K=20$, home advantage $100$ | **PASS (Causal)** |
| **E6 Online Attack/Defense**| `A_home`, `D_home`, `A_away`, `D_away` | Two-pass causal state engine; fixture $i$ reads state updated only on matches strictly preceding $t_i$ | **PASS (Causal)** |
| **Contextual Categorical**| `competition_id` | Static competition identifier known at scheduling | **PASS (Causal)** |

---

## 3. Training & Prospective Boundaries

- **Historical Training Window**: `2020-08-21T17:00:00.000000Z` to `2026-05-24T19:45:00.000000Z` (All 6 seasons).
- **Prospective Test Boundary**: All matches commencing in the **2026 Live Testing period**.
- **Leakage Status**: ZERO prospective 2026 live matches entered the training set, feature engineering, baseline fitting, or imputation.
