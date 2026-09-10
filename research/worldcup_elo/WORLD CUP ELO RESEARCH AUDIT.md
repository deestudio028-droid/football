# WORLD CUP ELO RESEARCH AUDIT

**Target Repository:** [hjjbh1314/worldcup-predictor](https://github.com/hjjbh1314/worldcup-predictor)  
**Audit Date:** 2026-08-20  
**Environment:** Isolated Research Sandbox (`research/worldcup_repo/`)  
**Scope:** Systematic code-level audit across 16 dimensions, mathematical equations, information pipeline, causality guarantees, and transferability analysis to the Football Prediction Project (FPP).

---

## 1. Categorized Evidence Summary

### A. VERIFIED FROM SOURCE CODE
1. **Elo Initialization**: `init_rating = 1500.0` for any new team entering the ratings dictionary (`src/elo.py:EloModel.__init__`).
2. **K-factor Hierarchy**: $K \in \{20, 30, 40, 50, 60\}$ based on tournament tier (`src/elo.py:tournament_k`). Scale parameter `k_scale = 1.0`.
3. **Home Advantage**: $+100.0$ Elo rating points added to the home team if `neutral == False`, $0.0$ if `neutral == True` (`src/elo.py:elo_diff`).
4. **Goal-Difference Multiplier**: Stepwise margin multiplier $G(|gd|) = 1.0$ if $|gd| \le 1$, $1.5$ if $|gd| = 2$, and $(11+|gd|)/8.0$ if $|gd| \ge 3$ (`src/elo.py:goal_diff_multiplier`).
5. **Competition Weighting**: Enforced solely via the K-factor tiers ($K=20$ for friendlies up to $K=60$ for World Cup finals).
6. **Cross-Season Handling**: Cumulative online rating continuity without arbitrary season-boundary resets.
7. **Mean Reversion**: Parameter `regress: float = 0.0` default. Formula $R \leftarrow R - \gamma(R - 1500)$ called at calendar year boundaries. Default is $\gamma = 0.0$ (no reversion).
8. **Rating Update Order**: Strict read-then-update sequence in `run()` and `build_feature_table()`: pre-match features are appended before `_update_one()` is called.
9. **Chronological Guarantees**: Input dataset is sorted by `date`.
10. **Probability Conversion**: `EloProbHead` in `src/baseline.py` applies `StandardScaler` on $[\Delta, |\Delta|]$ followed by `LogisticRegression(max_iter=2000)` with classes `['H', 'D', 'A']`.
11. **Calibration**: Multi-class Platt scaling (`PlattCalibrator` in `src/calibrate.py`) fitting `LogisticRegression(max_iter=3000)` on log-probabilities from a separate pre-test tuning set (2016–2018).
12. **Validation Methodology**: Static temporal split ($< 2018-01-01$ train, $\ge 2018-01-01$ test) in standard scripts; 3-way split ($< 2014$, $2014-2018$, $\ge 2018$) in tuning.
13. **Test-Set Tuning**: Grid search in `tune_elo.py` optimizes hyperparameters on the $2014-2018$ validation split and evaluates the test split ($\ge 2018$) only once.
14. **Leakage Risks**: Zero target leakage detected in feature construction. Unit test `test_no_leakage_causality()` enforces identical features when truncating the input data.
15. **Exact Dataset Used**: `martj42/international_results` (49,520 international fixtures, 1872–2026).
16. **Exact Reported Metrics**: Accuracy, Multiclass Log Loss, Brier Score, and Ranked Probability Score (RPS) implemented in `src/metrics.py`.

### B. CLAIMED BY README
- Elo Baseline achieves ~60.0% accuracy, Log Loss ~0.8735, Brier ~0.5137, RPS ~0.1707.
- HistGradientBoosting multi-feature model provides negligible lift over Elo (Accuracy 60.0%, Log Loss 0.8732, RPS 0.1705).
- Permutation importance is heavily dominated by `elo_diff` (+0.3418).
- Dixon-Coles goal model underperforms calibrated Elo (58.4% / RPS 0.177 vs 60.5% / RPS 0.169).
- Probability calibration corrects modern home advantage decay (Mean P(H) pulled from 0.510 to 0.477).

### C. REPRODUCED RESULT (100% REPRODUCIBILITY)
- **Elo Baseline**: Accuracy 60.1%, Log Loss 0.8730, Brier 0.5134, RPS 0.1705 (**MATCH**).
- **HistGradientBoosting**: Accuracy 60.0%, LogLoss 0.8733, Brier 0.5131, RPS 0.1705 (**MATCH**).
- **Permutation Importance**: `elo_diff` +0.3368, all other features $< 0.009$ (**MATCH**).
- **Calibrated Elo**: Accuracy 60.40%, LogLoss 0.8701, Brier 0.5114, RPS 0.1696 (**MATCH**).
- **Dixon-Coles**: Accuracy 58.45%, LogLoss 0.8949, Brier 0.5270, RPS 0.1774 (**MATCH**).
- **Sanity / Leakage Unit Tests**: 4/4 PASS (**MATCH**).

### D. NOT VERIFIED / NOT APPLICABLE TO FPP
- International confederation adjustments (`src/confed.py`): Not applicable to FPP's 5 domestic European leagues (no inter-confederation domestic matches).
- Static 2018 single split: FPP requires strict 3-fold rolling walk-forward validation (`iter_walk_forward_folds()`).

---

## 2. Mathematical Formalism

### 2.1 Pre-Match Difference & Expected Score
For fixture $i$ at timestamp $T$ between home team $H$ and away team $A$:
$$\Delta_i = (R_H + \text{home\_advantage} \cdot [1 - \text{neutral}]) - R_A$$
$$W_e = \frac{1}{1 + 10^{-\Delta_i / 400.0}}$$

### 2.2 Post-Match Elo Update
$$R'_H = R_H + K \cdot G \cdot (W - W_e)$$
$$R'_A = R_A - K \cdot G \cdot (W - W_e)$$
where $W \in \{1.0, 0.5, 0.0\}$, $G = \text{goal\_diff\_multiplier}(|gd|)$, and $K = \text{base\_K} \times k\_scale$.

### 2.3 3-Class Probability Head
Let $x_i = [\Delta_i, |\Delta_i|]^T$. Probabilities are computed via multinomial logistic regression:
$$P(y = c \mid x_i) = \frac{\exp(\beta_c^T \tilde{x}_i + \beta_{0, c})}{\sum_{k \in \{H, D, A\}} \exp(\beta_k^T \tilde{x}_i + \beta_{0, k})}$$
where $\tilde{x}_i = \text{StandardScaler}(x_i)$.

---

## 3. Transferability to FPP Domestic Leagues

| Mechanism | World Cup Predictor | FPP Domestic Adaptation | Transferability |
|---|---|---|---|
| Rating State | Global national-team pool | 5 distinct league teams | **TRANSFERABLE** |
| Cross-Season Persistence | Continuous over decades | Continuous across seasons | **TRANSFERABLE** |
| Margin Multiplier | eloratings $G(\|gd\|)$ | eloratings $G(\|gd\|)$ | **TRANSFERABLE** |
| Pre-Match Features | `elo_diff`, `home_elo`, `away_elo` | Direct inputs to `PoissonRegressor` | **TRANSFERABLE** |
| K-factor Hierarchy | 20 to 60 by tournament | Fixed $K=20$ (regular league fixtures) | **MODIFIED** |
| Validation Framework | Static 2018 Split | 3-Fold Walk-Forward (`iter_walk_forward_folds`) | **FPP PROTOCOL TAKES PRECEDENCE** |