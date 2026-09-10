# 03 — Causal Information Clock & Timing Assertions

## 1. Information Lifecycle

```
    Historical State t < t_cutoff            Kickoff t = 0
 [------------------------------------] | [---------------------------]
       1. Base V4 Lambdas Extracted     |
       2. E10 Probabilities Generated   |
       3. Reliability Features Built    |
       4. Confidence Bands Assigned     |      Actual Match Played
       ----------------------------------       Actual Outcome Observed
       (Strictly Pre-Kickoff Boundary)
```

---

## 2. Feature-by-Feature Causal Classification

| Feature Name | Source | Timestamp Cutoff | Verification Status |
|:---|:---|:---|:---:|
| `max_prob_e10`, `min_prob_e10`, `entropy` | E10 Prediction Engine | $t \le t_{\text{cutoff}}$ | **Verified Causal** |
| `margin_p1_p2`, `prob_concentration` | E10 Prediction Engine | $t \le t_{\text{cutoff}}$ | **Verified Causal** |
| `v4_lam_h`, `v4_lam_a`, `total_lam` | Frozen V4 GLMs | $t \le t_{\text{cutoff}}$ | **Verified Causal** |
| `home_elo`, `away_elo`, `abs_elo_diff` | Pre-match Elo Engine | $t \le t_{\text{cutoff}}$ | **Verified Causal** |
| `season_match_num`, `min_team_samples` | Historical Fixture Index | $t \le t_{\text{cutoff}}$ | **Verified Causal** |
| `js_divergence_e10_e13` | Causal Model Predictions | $t \le t_{\text{cutoff}}$ | **Verified Causal** |
