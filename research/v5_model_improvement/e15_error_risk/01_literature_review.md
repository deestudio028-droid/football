# 01 — Pre-Match Error Risk & Selective Forecasting: Literature Review

## 1. Executive Summary

This literature review investigates selective prediction, forecast difficulty modeling, abstention mechanisms, conformal prediction, and risk-coverage curves in machine learning and sports probabilistic forecasting.

---

## 2. Comprehensive Methodological Audit

### 1. Selective Classification & Abstention (Geifman & El-Yaniv 2017)
- **Methodology**: Augments a base classifier $f(x)$ with a selection / rejection function $g(x) \in \{0, 1\}$:
  $$(f, g)(x) = \begin{cases} f(x) & \text{if } g(x) = 1 \\ \text{ABSTAIN} & \text{if } g(x) = 0 \end{cases}$$
- **Relevance**: Evaluates risk-coverage curves $\mathcal{R}(f, g)$ subject to coverage constraint $\Phi(g) = \mathbb{E}[g(X)] \ge c$.
- **Finding**: In highly stochastic domains like sports, selective abstention reliably purges high-entropy derbies, reducing conditional loss.

### 2. Meta-Learning for Error Prediction (Brazdil et al. 2008)
- **Methodology**: Trains a secondary meta-model $\hat{L}(x)$ to predict the loss $\mathcal{L}(f(x), y)$ incurred by the primary model before the outcome is realized.
- **Finding**: Prediction entropy, probability margin ($P_{(1)} - P_{(2)}$), and Elo parity are strong linear predictors of expected multi-class loss.

### 3. Conformal Prediction & Set-Valued Forecasting (Vovk et al. 2005, Shafer & Vovk 2008)
- **Methodology**: Generates prediction sets $\Gamma^{\epsilon}(x) \subseteq \{H, D, A\}$ guaranteeing coverage probability $P(Y \in \Gamma^{\epsilon}(X)) \ge 1 - \epsilon$.
- **Relevance**: When set size $|\Gamma^{\epsilon}(x)| = 3$, the match is flagged as unresolvable / high-risk prior to kickoff.

### 4. Calibration Under Selection (Platt 1999, Guo et al. 2017)
- **Methodology**: Selection functions must preserve marginal probability calibration on the retained subset. Subsetting on predicted confidence can induce mild post-selection probability skew if uncalibrated.
