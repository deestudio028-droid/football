# 01 — Phase 41 (E16) Audit & Scientific Baseline

## 1. Executive Summary

This document reviews the methodological findings of Experiment E16 and establishes the scientific foundation for the Phase 42 (E17) Forecast Status Engine.

---

## 2. Key Audited Findings from E16

1. **Continuous Expected Loss Predictability**:
   - Pre-match forecast reliability is highly predictable ($r = +0.4078, p = 0.0000$).
   - Replaced binary tail-upset modeling with continuous expected loss $E[\text{RPS} \mid X]$.
2. **Prediction Geometry Dominance**:
   - Normalized entropy, max probability, and probability margin account for the vast majority of forecast reliability signal ($r > 0.42$).
3. **Generalization Across Folds & Leagues**:
   - Survived all 4 walk-forward folds including 2025/26 Blind Holdout ($r = 0.4194$).
   - Generalizes identically across all 5 major European leagues.
4. **Foundation for E17**:
   - E17 builds directly on E16's continuous calibrated reliability scores to establish an actionable 4-tier decision taxonomy (`STRONG`, `LEAN`, `CAUTION`, `AVOID`).
