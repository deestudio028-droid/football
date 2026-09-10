# Phase 8 — Draw Champion Freeze & Prospective Validation Protocol Report

**Date:** 2026-08-21
**Status:** Champion Frozen & Prospective Protocol Established. **Zero Production Files Modified.**

## 1. Executive Summary

- **Research Program Climax:** Successfully froze the exact equations, parameters, and causal contracts for the draw research champion (**V4 + Expanding-Window Dixon–Coles + Elo Draw Stacking**).
- **Reclassification of Reused 300 OOS:** Formally reclassified the existing 300-match Fresh-Extended OOS dataset as `REUSED_HISTORICAL_RESEARCH_OOS` (evaluated across 6 prior phases). It is retained strictly as historical diagnostic evidence and will **not** count toward prospective confirmation.
- **Prospective Protocol Established:** Established an immutable prospective validation protocol requiring a fresh, untouched future cohort of **$N \ge 1,050$ fixtures** ($\ge 80\%$ statistical power at $\alpha=0.05$) to resolve the remaining uncertainty.
- **Pre-Registered Promotion Gate:** Promotion consideration requires clearing all 7 pre-declared gates (Sample $N \ge 1,050$, $\Delta \le -0.0010$, $95\%$ CI strictly excluding zero, secondary metric stability).
- **Final Phase 8 Verdict:** **A. PROTOCOL FROZEN — READY FOR PROSPECTIVE DATA.** Zero production files modified.

## 2. Frozen Champion Mathematical Specification

### Primary Model Formulation
$$\text{logit}(P(D)_{\text{new}}) = 0.1130 + 0.6037 \cdot \text{logit}(P(D)_{\text{DC}}) + 0.4812 \cdot \text{logit}(P(D)_{\text{Elo}})$$

### Proportional Odds Simplex Redistribution
$$P(H)_{\text{new}} = P(H)_{\text{V4}} \cdot \frac{1 - P(D)_{\text{new}}}{1 - P(D)_{\text{V4}}}, \quad P(A)_{\text{new}} = P(A)_{\text{V4}} \cdot \frac{1 - P(D)_{\text{new}}}{1 - P(D)_{\text{V4}}}$$

### Mathematical Invariants
1. $\sum_{c \in \{H,D,A\}} P(c)_{\text{new}} = 1.0$ (Strict Simplex Normalization)
2. $P(c)_{\text{new}} \ge 0 \quad \forall c$ (Strict Non-Negativity)
3. $\frac{P(H)_{\text{new}}}{P(A)_{\text{new}}} = \frac{P(H)_{\text{V4}}}{P(A)_{\text{V4}}}$ (Invariance of Conditional Home/Away Relative Odds)

## 3. Exact Frozen Parameters

| Component | Parameter | Frozen Value | Source / Methodology |
|---|---|---|---|
| **V4 Baseline** | Artifact MD5 | `06841f0c03c8597b2b8cd8f8ab064864` | Production V4 Poisson + Elo + Online A/D |
| **Dixon–Coles** | Bundesliga $\rho$ | `-0.0768` | Shrunk League Empirical Bayes (Phase 2) |
| **Dixon–Coles** | Ligue 1 $\rho$ | `-0.0583` | Shrunk League Empirical Bayes (Phase 2) |
| **Dixon–Coles** | Serie A $\rho$ | `-0.0468` | Shrunk League Empirical Bayes (Phase 2) |
| **Dixon–Coles** | La Liga $\rho$ | `-0.0387` | Shrunk League Empirical Bayes (Phase 2) |
| **Dixon–Coles** | Premier League $\rho$ | `-0.0163` | Shrunk League Empirical Bayes (Phase 2) |
| **Dixon–Coles** | Global Fallback $\rho$ | `-0.0560` | Shrunk League Empirical Bayes (Phase 2) |
| **Elo Draw Curve** | Intercept $a_0$ | `+0.2227` | Bivariate Logistic Calibrator (Phase 3) |
| **Elo Draw Curve** | Slope $a_1$ ($	ext{logit}(P_D)$) | `+1.1278` | Bivariate Logistic Calibrator (Phase 3) |
| **Elo Draw Curve** | Slope $a_2$ ($|d\text{Elo}|/100$) | `-0.1652` | Bivariate Logistic Calibrator (Phase 3) |
| **Stacking Layer** | Intercept | `+0.1130` | Regularized Logistic Stacking ($L_2=10.0$) |
| **Stacking Layer** | Weight DC | `+0.6037` | Regularized Logistic Stacking ($L_2=10.0$) |
| **Stacking Layer** | Weight Elo | `+0.4812` | Regularized Logistic Stacking ($L_2=10.0$) |

## 4. Reused 300 OOS Classification

> [!IMPORTANT]
> **Formal Classification:** `REUSED_HISTORICAL_RESEARCH_OOS`  
> The 300-match Fresh-Extended dataset (`fresh_extended_fixture_ids.json`) was queried across Phases 2–7. While strict pre-registration prevented outcome tuning, repeated confirmation creates statistical exposure. It is now classified as historical diagnostic data and is **strictly barred from counting toward prospective confirmation**.

## 5. Prospective Validation Protocol & Promotion Decision Rule

| Protocol Requirement | Specification | Rationale |
|---|---|---|
| **Sample Size** | **$N \ge 1,050$ fixtures** (Preferred $N \ge 1,500$) | Guarantees $\ge 80\%$ statistical power at $\alpha=0.05$ |
| **Data State** | Untouched prospective future matches | Zero prior evaluation, zero tuning |
| **Primary Metric** | Multiclass Cross-Entropy (Log Loss) vs V4 Base | Project primary loss function |
| **Uncertainty Test** | Two-tailed paired bootstrap (10,000 resamples, seed 20260820) | 95% CI must strictly exclude zero |
| **Practical Threshold** | $\Delta \text{Log Loss} \le -0.0010$ | Substantive performance gain required |
| **Secondary Checks** | Brier Score, RPS, ECE, Draw Calibration Bias | Prevents degradation in calibration or ordinal accuracy |
| **Bucket Stability** | Chronological 50- and 100-match rolling buckets | Confirms improvement is not localized to one burst |

## 6. Audit & Gates Summary

| Verification Gate | Result | Notes |
|---|---|---|
| **INTEGRITY** | **PASS** | All 20 protected assets bit-identical pre- and post-flight |
| **DETERMINISM** | **PASS** | Bit-identical repeat execution (\(\Delta = 0.000\text{e}{+}00\)) |
| **NO OOS ACCESS** | **PASS** | Zero new OOS evaluation performed in Phase 8 |
| **PRODUCTION ISOLATION** | **PASS** | Zero production models (V2/V3/V4) modified |

## 7. Final Phase 8 Research Verdict

**VERDICT: A. PROTOCOL FROZEN — READY FOR PROSPECTIVE DATA**

**NO PRODUCTION CHANGE. CHAMPION IS FROZEN AND READY FOR FUTURE PROSPECTIVE VALIDATION.**