# 06 — Real Prospective Testing Protocol & Test Lock

## 1. Prospective Testing Principles

The **2026 live matches** serve as the **True Unseen Prospective Validation Set**.

To maintain scientific validity:
1. **Zero Retraining During Test Run**:
   `v4_1_prospective_candidate_2025_26.pkl` is locked and immutable. No model parameters, scalers, imputers, or baselines may be updated using live 2026 match outcomes during the test period.
2. **Strict Pre-Kickoff Predictions**:
   Predictions for any 2026 fixture must be computed and logged **strictly before kickoff timestamp ($t_{\text{pred}} < t_{\text{kickoff}}$)**.
3. **Immutable Prediction Records**:
   For every prospective match, log:
   - `fixture_id`
   - `prediction_timestamp`
   - `P(H)`, `P(D)`, `P(A)`
   - `model_version = v4.1-champion-dc-elo-stacking-2025-26-trained`
   - `model_file_md5 = 145f918d933eb343c0f63ca342b10289`
4. **Post-Match Evaluation**:
   After the fixture status becomes `FT` / final, append the actual result and compute prospective evaluation metrics.

---

## 2. Prospective Evaluation Metrics

### Primary Scoring Rules:
- **Ranked Probability Score (RPS)**:
  $$\text{RPS} = \frac{1}{2} \left[ (P_H - \mathbf{1}_{\{y=H\}})^2 + ((P_H + P_D) - \mathbf{1}_{\{y \in \{H, D\}\}})^2 \right]$$
- **Multi-Class Log-Loss**:
  $$\text{Log-Loss} = -\ln(P_y)$$

### Secondary Quality Metrics:
- **Draw Expected Calibration Error ($ECE$)**: 10-bin reliability curve for draw probability.
- **Top-1 Pick Win Rate (%)**
- **Multi-Class Brier Score**
- **Cumulative RPS & Log-Loss Tracking Curves**
- **League-Wise Disaggregation (5 Leagues)**
