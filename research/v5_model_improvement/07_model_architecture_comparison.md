# 07 — Model Architecture Comparison: 15 Mathematical & ML Paradigms

## 1. Executive Summary

This document presents a rigorous comparative analysis of **15 mathematical, statistical, and machine learning model architectures** for association football 1X2 match forecasting.

Our core finding is that **no single pure model dominates across all dimensions**:
- **Statistical / Poisson models** provide superior probabilistic structure, scoreline interpretability, and exact mathematical coherence, but struggle with non-linear feature interactions.
- **Tree-based gradient boosters (LightGBM, CatBoost)** excel at capturing complex tabular interactions (e.g., fatigue $\times$ travel $\times$ team depth), but produce uncalibrated raw probabilities that require post-hoc calibration.
- **Ensembles and Mixture-of-Experts** combining a structural Poisson core with non-linear residual gates achieve the highest Ranked Probability Score ($RPS$) and optimal draw sensitivity.

---

## 2. Deep Comparative Architectural Profiles

### Model A: Current Production $V_4$ Poisson
- **Mathematical Form**: Two independent Poisson GLMs estimating expected goals:
  $$\ln \lambda_{\text{home}} = \beta_0^H + \sum \beta_j^H x_j, \quad \ln \lambda_{\text{away}} = \beta_0^A + \sum \beta_j^A x_j$$
  Mapped to 1X2 probabilities via a $15 \times 15$ Tail-Safe Bivariate Poisson grid.
- **Strengths**: 100% mathematically coherent; generates full scoreline probability distributions ($P(x, y)$); fast, robust, and completely deterministic; zero overfitting on small samples.
- **Weaknesses**: Linear in the log-link; cannot capture non-linear feature thresholds (e.g., severe fatigue cliffs) without explicit manual feature engineering; underestimates draws without external gate.
- **Feature Requirements**: 91 continuous and scaled features.
- **Training Complexity**: Extremely low ($\approx 2\text{ seconds}$).
- **Calibration**: High on Home and Away; systematically conservative on Draws ($P(D) \approx 25\text{–}27\%$).
- **Draw Performance**: Low recall on raw argmax ($0.0\%$ argmax share); requires shadow gate ($V_{4.6}$).
- **Interpretability**: Very High (GLM coefficients directly represent percentage goal multiplier).
- **Role in $V_5$**: **The Invariant Production Baseline**.

---

### Model B: Dixon-Coles Model
- **Mathematical Form**: Modifies Poisson joint probabilities for low scores using interaction parameter $\rho$:
  $$P(X=x, Y=y) = \tau_{\lambda, \mu}(x, y) \frac{e^{-\lambda}\lambda^x}{x!} \frac{e^{-\mu}\mu^y}{y!}$$
  where $\tau(0,0) = 1 - \lambda \mu \rho$, $\tau(1,0) = 1 + \mu \rho$, $\tau(0,1) = 1 + \lambda \rho$, $\tau(1,1) = 1 - \rho$.
- **Strengths**: Directly repairs the structural Poisson deficiency for $0-0$ and $1-1$ scorelines; preserves scoreline distribution; backed by 25+ years of academic validation.
- **Weaknesses**: Maximum likelihood estimation (MLE) over all team parameters requires non-convex optimization; sensitive to initializations; assumes constant $\rho$ across diverse leagues.
- **Feature Requirements**: Team identities, match dates, goals scored/conceded.
- **Training Complexity**: Moderate (BFGS/L-BFGS optimization across $2N + 2$ parameters).
- **Calibration**: Excellent on low-scoring draws; significantly reduces draw Brier score.
- **Draw Performance**: Substantially higher draw accuracy and $RPS$ improvement over independent Poisson.
- **Interpretability**: Very High ($\rho$ directly represents low-score dependence).
- **Role in $V_5$**: **Primary Candidate for Score-Distribution Transformation**.

---

### Model C: Bivariate Poisson (Karlis-Ntzoufras)
- **Mathematical Form**:
  $$X = X_1 + X_3, \quad Y = Y_1 + X_3 \quad \text{where } X_1 \sim \text{Pois}(\lambda_1), Y_1 \sim \text{Pois}(\lambda_2), X_3 \sim \text{Pois}(\lambda_3)$$
  $\text{Cov}(X, Y) = \lambda_3 \ge 0$.
- **Strengths**: Explicitly models positive correlation between team scores (capturing high-scoring vs low-scoring match tempos).
- **Weaknesses**: Forces positive correlation across *all* scorelines; inflates $4-4$ and $3-3$ probabilities unrealistically; computationally heavier than Dixon-Coles.
- **Feature Requirements**: Team attack/defense and match pace indicators.
- **Training Complexity**: Moderate (EM algorithm).
- **Calibration**: Moderate; over-predicts high-scoring draws.
- **Draw Performance**: Improves draw recall but harms log-loss on blowout matches ($4-0, 5-1$).
- **Interpretability**: High ($\lambda_3$ = global match pace).
- **Role in $V_5$**: Secondary benchmark.

---

### Model D: Skellam Distribution Regression
- **Mathematical Form**: Models the goal difference $Z = X - Y$ directly as a Skellam distribution:
  $$P(Z=k) = e^{-(\lambda_1 + \lambda_2)} \left(\frac{\lambda_1}{\lambda_2}\right)^{k/2} I_{|k|}(2\sqrt{\lambda_1 \lambda_2})$$
  where $I_k$ is the modified Bessel function of the first kind.
- **Strengths**: Directly targets the margin of victory $k$; $k > 0 \implies H$, $k = 0 \implies D$, $k < 0 \implies A$; avoids independent scoring assumptions.
- **Weaknesses**: Discards total goal intensity information (cannot distinguish a $0-0$ match from a $3-3$ match, which have vastly different underlying dynamics); Bessel function computation is slower.
- **Feature Requirements**: Differential features ($\Delta \text{Elo}, \Delta xG, \Delta \text{Strength}$).
- **Training Complexity**: Moderate.
- **Calibration**: Moderate on 1X2; poor for Over/Under total goals.
- **Draw Performance**: Naturally outputs $P(Z=0) = P(D)$, providing clean mathematical draw calibration.
- **Interpretability**: Moderate.
- **Role in $V_5$**: Specialized draw-validation benchmark.

---

### Model E: Bayesian Hierarchical Poisson Model
- **Mathematical Form**: Hierarchical prior structure on team attack $\alpha_i \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)$ and defense $\beta_i \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)$ with league-level hyperpriors.
- **Strengths**: Natural shrinkage of newly promoted and low-sample teams toward the league prior; outputs full posterior predictive distributions and parameter uncertainty.
- **Weaknesses**: MCMC sampling (via Stan or PyMC) is computationally intensive (several minutes to hours per season); unsuitable for instant real-time live dashboard inference.
- **Feature Requirements**: Team hierarchies, match results, home advantage hyperpriors.
- **Training Complexity**: Very High (MCMC sampling).
- **Calibration**: State-of-the-art out-of-sample calibration.
- **Draw Performance**: Strong due to posterior uncertainty integration.
- **Interpretability**: Exceptional (full credible intervals for every team and parameter).
- **Role in $V_5$**: Offline research gold standard for promoted-team shrinkage priors.

---

### Model F: Ordered Logistic Regression
- **Mathematical Form**: Models 1X2 as an ordinal variable ($A < D < H$) with latent variable $y^* = \beta^T x + \epsilon$ and learned cutoffs $\kappa_1, \kappa_2$:
  $$P(A) = \sigma(\kappa_1 - y^*), \quad P(D) = \sigma(\kappa_2 - y^*) - \sigma(\kappa_1 - y^*), \quad P(H) = 1 - \sigma(\kappa_2 - y^*)$$
- **Strengths**: Naturally respects the ordinal geometry of football ($H \leftrightarrow D \leftrightarrow A$); guarantees that draw probability peaks when $y^* \approx 0$ (team parity).
- **Weaknesses**: Proportional odds assumption is frequently violated in football (the factors driving home wins vs draws differ from those driving draws vs away wins).
- **Feature Requirements**: Differential features ($\Delta \text{Elo}, \Delta \text{Form}, \Delta xG$).
- **Training Complexity**: Very Low ($< 1\text{ second}$).
- **Calibration**: High on 1X2 probabilities.
- **Draw Performance**: Strong baseline for parity matches.
- **Interpretability**: Very High ($\kappa_1, \kappa_2$ define the universal draw corridor).
- **Role in $V_5$**: High-value fast parity benchmark.

---

### Model G: Multinomial Logistic Regression
- **Mathematical Form**: Direct 3-class softmax modeling:
  $$P(Y=k) = \frac{e^{\beta_k^T x}}{\sum_{j \in \{H, D, A\}} e^{\beta_j^T x}}$$
- **Strengths**: Unconstrained separate parameter vectors for Home, Draw, and Away; allows distinct features to drive Draw probability independently from Win probabilities.
- **Weaknesses**: Ignores the structural constraint that goals are generated by physical scoring events; requires strong L2 regularization to prevent overfitting on 90+ features.
- **Feature Requirements**: Scaled tabular features.
- **Training Complexity**: Very Low.
- **Calibration**: Moderate (often overconfident on extremes).
- **Draw Performance**: Moderate; often suppresses draw probabilities in non-parity matches.
- **Interpretability**: Moderate.
- **Role in $V_5$**: Standard linear baseline.

---

### Model H: XGBoost (Extreme Gradient Boosting)
- **Mathematical Form**: Ensembles of regularized regression trees trained on multiclass log-loss objective with second-order Taylor expansion approximations.
- **Strengths**: Captures complex non-linear feature interactions (e.g. `rest_diff < -3` AND `is_away == 1` AND `elo_diff < 50`); robust to collinearity and monotonic feature transformations.
- **Weaknesses**: Highly vulnerable to overfitting on noisy match outcomes; probabilities often uncalibrated near boundaries; treats 1X2 as arbitrary discrete labels without goal structure.
- **Feature Requirements**: Tabular feature matrix.
- **Training Complexity**: Low/Moderate ($5\text{–}30\text{ seconds}$).
- **Calibration**: Poor out-of-the-box (requires post-hoc isotonic regression or Platt scaling).
- **Draw Performance**: Tends to severely under-predict draws because draws are the minority class (~26%) with the highest entropy.
- **Interpretability**: Moderate (SHAP values, feature importance).
- **Role in $V_5$**: Strong tabular benchmark for non-linear interactions.

---

### Model I: LightGBM Poisson Dual-Objective Regressor
- **Mathematical Form**: Dual LightGBM regressors trained with native Poisson objective ($\text{loss} = \hat{y} - y \ln \hat{y}$) predicting $\lambda_{\text{home}}, \lambda_{\text{away}}$, then converted via bivariate score matrix.
- **Strengths**: Combines non-linear tree power with structural Poisson scoreline mapping; leaf-wise tree growth captures subtle goal-rate variations; faster and more memory-efficient than XGBoost.
- **Weaknesses**: Requires careful regularization (`min_child_samples`, `colsample_bytree`, `learning_rate`) to avoid overfitting on high-scoring blowout matches.
- **Feature Requirements**: Tabular feature matrix.
- **Training Complexity**: Very Low ($1\text{–}5\text{ seconds}$).
- **Calibration**: High (inherits bivariate Poisson probability structure).
- **Draw Performance**: High when paired with Dixon-Coles or Physical Gate post-processing.
- **Interpretability**: High (outputs interpretable $\lambda_H, \lambda_A$ + SHAP explanations).
- **Role in $V_5$**: **Top-Tier Primary Challenger for $V_4$ Poisson Replacement**.

---

### Model J: CatBoost
- **Mathematical Form**: Gradient boosting on decision trees with symmetric (oblivious) trees and native ordered boosting to eliminate target leakage during training.
- **Strengths**: Best-in-class handling of categorical features (`competition_id`, `venue`, `referee_id`, `formation`); ordered boosting provides superior resistance to overfitting on small tabular sports datasets.
- **Weaknesses**: Slightly slower training than LightGBM; default hyperparameters can be overly conservative.
- **Feature Requirements**: Tabular features (supports raw categoricals without manual one-hot encoding).
- **Training Complexity**: Moderate ($10\text{–}60\text{ seconds}$).
- **Calibration**: Better native probability calibration than XGBoost.
- **Draw Performance**: Moderate/High.
- **Interpretability**: Moderate/High.
- **Role in $V_5$**: Top-tier candidate for mixed categorical/numerical feature expansions.

---

### Model K: Deep Neural Network (MLP / TabNet)
- **Mathematical Form**: Multi-layer perceptron with residual connections, batch normalization, dropout, and softmax output.
- **Strengths**: Capable of learning arbitrary continuous representations; can ingest multi-modal embeddings (e.g. text news + player embeddings).
- **Weaknesses**: Severely overfits on tabular football data where $N \approx 1,800\text{ matches/year}$; lacks inductive bias for football physics; requires extensive hyperparameter tuning.
- **Feature Requirements**: Standardized continuous features.
- **Training Complexity**: High ($1\text{–}10\text{ minutes}$, GPU preferred).
- **Calibration**: Poor (deep networks suffer from overconfidence calibration drift).
- **Draw Performance**: Very Poor (draws are routinely washed out by high-variance gradients).
- **Interpretability**: Very Low (black box).
- **Role in $V_5$**: Not recommended for core production (proven inferior to GBDT on tabular sports data).

---

### Model L: Stacked Meta-Learner Ensemble
- **Mathematical Form**: Level-0 Base Learners (V4 Poisson, LightGBM, Ordered Logit, Market Odds) outputting probability vectors $P_m \in \mathbb{R}^3$, combined by a Level-1 regularized meta-regressor (e.g., L2 Logistic Regression).
- **Strengths**: Combines the structural stability of Poisson models with the non-linear interaction power of tree boosters and market intelligence; achieves lowest overall out-of-sample $RPS$.
- **Weaknesses**: High pipeline complexity; strict cross-validation stacking required to prevent Level-1 meta-learner from overfitting on Level-0 training predictions.
- **Feature Requirements**: Outputs of diverse Level-0 models.
- **Training Complexity**: Moderate.
- **Calibration**: Exceptional when trained with strictly proper scoring loss functions.
- **Draw Performance**: State-of-the-art.
- **Interpretability**: Moderate.
- **Role in $V_5$**: **Top-Tier Candidate for Ultimate $V_5$ Production Architecture**.

---

### Model M: Probability-Level Convex Blending
- **Mathematical Form**: Convex combination of calibrated probability outputs:
  $$P_{V_5}(H/D/A) = w_1 P_{\text{Poisson}}(H/D/A) + w_2 P_{\text{LightGBM}}(H/D/A) + w_3 P_{\text{Market}}(H/D/A)$$
  where $\sum w_i = 1, w_i \ge 0$, optimized via Nelder-Mead or SLSQP to minimize out-of-sample $RPS$.
- **Strengths**: Zero risk of catastrophic failure; guarantees that output probabilities remain valid distributions ($\sum P = 1, P \ge 0$); trivial to audit and verify.
- **Weaknesses**: Static global weights $w_i$ do not adapt to match-specific regimes (e.g., parity matches vs lopsided matches).
- **Feature Requirements**: Calibrated probability outputs from sub-models.
- **Training Complexity**: Extremely Low ($< 1\text{ second}$).
- **Calibration**: Highest reliability.
- **Draw Performance**: High.
- **Interpretability**: Very High (transparent percentage blending).
- **Role in $V_5$**: **Core Ensemble Baseline ($E_7$ Expansion)**.

---

### Model N: Residual Draw Correction Architecture
- **Mathematical Form**: Base Model generates $(P_{\text{base}}(H), P_{\text{base}}(D), P_{\text{base}}(A))$. A secondary specialized binary classifier predicts the probability of residual draw underestimation $\Delta_{\text{draw}}$, applying a constrained probability transfer:
  $$P_{V_5}(D) = P_{\text{base}}(D) + \Delta_{\text{draw}}, \quad P_{V_5}(H) = P_{\text{base}}(H) - \frac{P_{\text{base}}(H)}{P_{\text{base}}(H)+P_{\text{base}}(A)}\Delta_{\text{draw}}$$
- **Strengths**: Preserves baseline model strength on decisive win/loss matches while specifically intervening on high-risk draw fixtures (exactly matching the philosophy of $V_{4.6}$ and Historical Candidate H).
- **Weaknesses**: Requires strict margin bounds to prevent over-allocation of draw probability in high-scoring games.
- **Feature Requirements**: Parity features ($RPI, xG_{\text{sum}}, \text{abs\_elo\_diff}, \Delta \text{rest}$).
- **Training Complexity**: Low.
- **Calibration**: High on draws.
- **Draw Performance**: Exceptional.
- **Interpretability**: Very High.
- **Role in $V_5$**: **Primary Candidate for Draw-Layer Evolution**.

---

### Model O: Mixture-of-Experts (MoE) Architecture
- **Mathematical Form**: A Gating Network $G(x) \in \Delta^3$ routes the match feature vector $x$ across three specialized Expert sub-models:
  1. **Expert 1 (Heavy Mismatch Expert)**: Optimized for lopsided matches ($|\Delta \text{Elo}| > 150$).
  2. **Expert 2 (Tactical Parity Expert)**: Optimized for balanced mid-table clashes ($|\Delta \text{Elo}| \le 80, xG_{\text{sum}} < 2.70$).
  3. **Expert 3 (High-Variance / Volatile Expert)**: Optimized for early-season, promoted-team, or cup-fatigued matches.
  $$P(Y) = \sum_{k=1}^3 G_k(x) \cdot P_{\text{Expert}_k}(Y)$$
- **Strengths**: Solves the fundamental tension in football forecasting where one model cannot simultaneously optimize for blowout games and tight $0-0$ defensive grinds.
- **Weaknesses**: Requires sufficient sample size in each partition ($N \ge 1,000$ per regime) to train experts reliably.
- **Feature Requirements**: Full candidate feature matrix.
- **Training Complexity**: Moderate.
- **Calibration**: Very High.
- **Draw Performance**: Optimal (Expert 2 focuses exclusively on draw mechanics).
- **Interpretability**: High (regime gate is explicitly observable).
- **Role in $V_5$**: Long-term structural candidate.

---

## 3. Master Architecture Comparison Matrix

| Model Code | Model Name | Model Class | Training Speed | Primary Loss Function | Native Calibration | Draw Sensitivity | Interpretability | Recommended Role |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **A** | $V_4$ Poisson | Linear GLM + Matrix | $< 2\text{s}$ | Poisson NLL | High | Low (Needs Gate) | Very High | **Invariant Baseline** |
| **B** | Dixon-Coles | Bivariate Poisson + $\rho$ | $15\text{s}$ | Bivariate NLL | Very High | High | Very High | **Primary Transformation** |
| **C** | Bivariate Poisson | Karlis-Ntzoufras | $30\text{s}$ | Bivariate NLL | Moderate | Moderate | High | Secondary Benchmark |
| **D** | Skellam Regression | Goal Diff GLM | $5\text{s}$ | Skellam NLL | Moderate | High | Moderate | Draw Validation |
| **E** | Bayesian Hierarchical | MCMC Poisson | $30\text{m}$ | Posterior Log-Lik | Gold Standard | High | Very High | Offline Prior Discovery |
| **F** | Ordered Logistic | Ordinal Linear | $< 1\text{s}$ | Ordered Log-Loss | High | High | Very High | Fast Parity Benchmark |
| **G** | Multinomial Logistic | Softmax Linear | $< 1\text{s}$ | Cross-Entropy | Moderate | Low | Moderate | Linear Direct Benchmark |
| **H** | XGBoost | Gradient Boosted Trees | $20\text{s}$ | Multiclass Log-Loss | Poor | Very Low | Moderate | Non-linear Benchmark |
| **I** | LightGBM Poisson | Dual GBDT + Matrix | $3\text{s}$ | Dual Poisson NLL | High | High | High | **Top-Tier Challenger** |
| **J** | CatBoost | Ordered Tree Ensemble | $45\text{s}$ | Multiclass Log-Loss | Moderate | Moderate | High | Categorical Specialist |
| **K** | Neural Network | Deep MLP / TabNet | $5\text{m}$ | Cross-Entropy | Poor | Very Low | Very Low | **Rejected (Overfits)** |
| **L** | Stacked Ensemble | Level-0 + Level-1 | $15\text{s}$ | Meta Log-Loss | State-of-the-Art | State-of-the-Art | Moderate | **Top-Tier Architecture** |
| **M** | Probability Blending | Convex Optimization | $< 1\text{s}$ | $RPS$ Minimization | Very High | High | Very High | **Core Ensemble Baseline** |
| **N** | Residual Draw Correction | Base + Residual Gate | $3\text{s}$ | Constrained $RPS$ | Very High | State-of-the-Art | Very High | **Top Draw Architecture** |
| **O** | Mixture-of-Experts | Gated Regime Experts | $10\text{s}$ | Joint Gated NLL | Very High | State-of-the-Art | High | Structural Long-Term |

---

## 4. Architectural Synthesis for $V_5$

The research points unequivocally toward a **Three-Layer Hybrid Architecture** for $V_5$:
1. **Layer 1 (Core Intensity Engine)**: Dual LightGBM Poisson Regressors with $xG$ and Elo feature inputs estimating $(\lambda_H, \lambda_A)$.
2. **Layer 2 (Score Distribution Mapping)**: Vectorized Dixon-Coles bivariate transformation applying calibrated $\rho$ parameters and low-score dependency.
3. **Layer 3 (Residual Parity Gate)**: Continuous Rating Parity ($RPI$) and Match Load fatigue correction tuning draw probability before final output.
