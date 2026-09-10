# 03 — Orthogonal Feature Dictionary

## 1. Executive Summary

This feature dictionary documents the 40 causal pre-match features constructed in Experiment $E_{12}$ ([`data/research/e12_orthogonal_features.parquet`](file:///e:/Football%20Prediction%20Project/data/research/e12_orthogonal_features.parquet)).

All features are strictly zero-leakage ($t_{\text{past}} < t_{\text{match}}$).

---

## 2. Feature Definitions by Hypothesis Group

### Group A: Finishing & Defensive Residuals (9 Features)
- `h_fin_ewma5`: Home team EWMA finishing residual ($\sum w_i (\text{goals} - xG)_i / \sum w_i$, $t_{1/2} = 5$).
- `h_def_ewma5`: Home team EWMA defensive concession residual ($\sum w_i (GA - xGA)_i / \sum w_i$, $t_{1/2} = 5$).
- `h_net_luck_ewma5`: Net luck index (`h_fin_ewma5 - h_def_ewma5`).
- `a_fin_ewma5`, `a_def_ewma5`, `a_net_luck_ewma5`: Corresponding Away team metrics.
- `match_finishing_residual_diff`: `h_fin_ewma5 - a_fin_ewma5`.
- `match_defensive_residual_diff`: `h_def_ewma5 - a_def_ewma5`.
- `match_net_luck_diff`: `h_net_luck_ewma5 - a_net_luck_ewma5`.

### Group B: xG Momentum & Trends (5 Features)
- `h_xg_trend`: Short-term ($t_{1/2}=3$) vs. Long-term ($t_{1/2}=10$) offensive $xG$ momentum.
- `h_xga_trend`: Short-term vs. Long-term defensive concession trend.
- `a_xg_trend`, `a_xga_trend`: Away team trend metrics.
- `match_xg_momentum_diff`: `(h_xg_trend - h_xga_trend) - (a_xg_trend - a_xga_trend)`.

### Group C: Venue-Specific Specialization (7 Features)
- `h_venue_xg_ewma`: Home team offensive $xG$ computed exclusively in prior home matches ($t_{1/2} = 5$).
- `h_venue_xga_ewma`: Home team defensive $xGA$ in prior home matches.
- `h_venue_xgd`: `h_venue_xg_ewma - h_venue_xga_ewma`.
- `a_venue_xg_ewma`, `a_venue_xga_ewma`, `a_venue_xgd`: Away team metrics computed exclusively in prior away matches.
- `venue_spec_xgd_diff`: `h_venue_xgd - a_venue_xgd`.

### Group D: Matchup-Level Relative Intensity (4 Features)
- `exp_matchup_home_xg`: Pre-match expected Home $xG$ from relative attacking and defending strengths.
- `exp_matchup_away_xg`: Pre-match expected Away $xG$.
- `xg_matchup_delta`: `exp_matchup_home_xg - exp_matchup_away_xg`.
- `xg_matchup_total`: `exp_matchup_home_xg + exp_matchup_away_xg`.

### Group E: Schedule Fatigue Interactions (7 Features)
- `rest_days_diff`: Home rest days minus Away rest days.
- `a_congestion_7d`, `h_congestion_7d`: Count of matches in the previous 7 days.
- `a_congestion_14d`: Count of matches in the previous 14 days.
- `away_fatigue_x_def`: Away 7-day match congestion $\times$ Away defensive concession strength.
- `away_fatigue_x_att`: Away 7-day match congestion $\times$ Away attacking strength.
- `rest_diff_x_quality`: Rest differential $\times$ Home team net quality.

### Group F: Elo $\times$ xG Orthogonal Discrepancy (1 Feature)
- `xg_implied_dominance`: $xG$-based expected dominance proportion $\frac{\text{exp\_matchup\_home\_xg}}{\text{exp\_matchup\_home\_xg} + \text{exp\_matchup\_away\_xg}} \in [0, 1]$.
