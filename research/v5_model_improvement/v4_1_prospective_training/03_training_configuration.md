# 03 — V4.1 Training Configuration & Hyperparameters

## 1. Candidate Identity & Architecture

- **Model Identifier**: `v4_1_prospective_candidate`
- **Model Version**: `v4.1-champion-dc-elo-stacking-2025-26-trained`
- **Governance Status**: `PROSPECTIVE_TEST_CANDIDATE`
- **Target File**: `data/models/v4_1_prospective_candidate_2025_26.pkl`

---

## 2. Training Scope & Dataset Geometry

- **Input Dimension ($N_{\text{features}}$)**: 91 features (87 V3 contract features + 4 E6 Online Attack/Defense features).
- **Training Matches ($N_{\text{train}}$)**: **10,734 matches**
- **Training Seasons**: `['2020/2021', '2021/2022', '2022/2023', '2023/2024', '2024/2025', '2025/2026']`
- **Target Variables**:
  - `label_home_goals` (Mean: $1.5348$)
  - `label_away_goals` (Mean: $1.2740$)

---

## 3. Preprocessing & Encoding Pipeline

- **Preprocessor**: `LogisticRegressionPreprocessor`
  - Numeric columns (90 columns): Median Imputation (`SimpleImputer(strategy="median")`) followed by Z-Score standardization (`StandardScaler()`).
  - Categorical column (`competition_id`): One-Hot Encoded into 5 binary indicators for leagues `[200, 419, 423, 477, 499]`.
  - Encoded Design Matrix Dimension: $(10734, 95)$

---

## 4. Regressors & Conversion Hyperparameters

- **Home Goal Regressor**: `PoissonRegressor(alpha=1.0, max_iter=2000, fit_intercept=True)`
- **Away Goal Regressor**: `PoissonRegressor(alpha=1.0, max_iter=2000, fit_intercept=True)`
- **Dixon-Coles Score Matrix Transformation**:
  - Correlation parameter $\rho = -0.08$
  - Truncation grid: $[0, 10] \times [0, 10]$ goals
  - Low-score probability adjustments:
    $$\tau(0, 0) = 1 - \lambda_H \lambda_A \rho$$
    $$\tau(1, 0) = 1 + \lambda_A \rho$$
    $$\tau(0, 1) = 1 + \lambda_H \rho$$
    $$\tau(1, 1) = 1 - \rho$$
- **1X2 Simplex Conversion**:
  $$P(H) = \sum_{i > j} M_{i, j}, \quad P(D) = \sum_{i = j} M_{i, j}, \quad P(A) = \sum_{i < j} M_{i, j}$$
