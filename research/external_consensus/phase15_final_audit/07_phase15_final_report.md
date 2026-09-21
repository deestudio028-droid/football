# Phase 15 Final Results Audit Report

**Cohort:** 2026-09-18 → 2026-09-21 | **Leagues:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1  
**Generated:** 2026-09-21 03:06 UTC  
**V4 SHA256:** `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`  
**Status:** ANALYSIS ONLY — V4 model NOT modified

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Fixtures completed | 48 / 48 |
| 1X2 Accuracy | **56.2%** (27/48) |
| Exact Score Accuracy | **8.3%** (4/48) |
| Mean Brier Score | 0.5785 |
| Mean Log Loss | 0.9732 |
| High-Confidence Accuracy | 100.0% (3/3) |

---

## A. 1X2 Accuracy

**27/48 = 56.2%** correct 1X2 predictions.

**Actual outcome distribution:**

| Outcome | Actual | Predicted |
|---------|--------|-----------|
| Home Win (H) | 27 | 36 |
| Draw (D) | 10 | 0 |
| Away Win (A) | 11 | 12 |

## B. Exact Score Accuracy

**4/48 = 8.3%** exact score hits.

**Exact score hits:**

| Fixture | Score | League |
|---------|-------|--------|
| Osasuna vs Rayo Vallecano | **1-1** | La Liga |
| Fiorentina vs Napoli | **1-1** | Serie A |
| Fulham vs Manchester United | **1-1** | Premier League |
| Deportivo A Coruña vs Real Betis | **1-1** | La Liga |

## C. Brier Score

Mean Brier: **0.5785** (range: 0.0786–0.8824)

> Lower is better. Perfect prediction = 0. Random 3-class baseline ≈ 0.667.

## D. Log Loss

Mean Log Loss: **0.9732** (range: 0.2530–1.4305)

## E. Draw Risk Analysis

| Tier | Fixtures | Actual Draws | Draw Rate |
|------|----------|-------------|-----------|
| LOW | 6 | 0 | 0.0% |
| MEDIUM | 32 | 8 | 25.0% |
| HIGH | 10 | 2 | 20.0% |

## F. 2-0 Signal Analysis

**2 fixture(s)** flagged as `2-0_PROFILE`:

- Outcome accuracy: **100.0%** (2/2)
- Exact score accuracy: **0.0%** (0/2)

| Fixture | λ_H | λ_A | P(H) | Pred | Actual | Outcome ✓ | Exact ✓ |
|---------|-----|-----|------|------|--------|-----------|---------|
| FC Bayern München vs FC Union Berlin | 2.31 | 0.53 | 77.6% | 2-0 | 7-0 | ✅ | ❌ |
| Manchester City vs Sunderland | 2.16 | 0.73 | 70.5% | 2-0 | 5-3 | ✅ | ❌ |

## G. Strong Home Signal Analysis

**3 fixture(s)** with `strong_home_profile=True`:
- Outcome accuracy: **100.0%** (3/3)

| Fixture | Pred | Actual | Score Pred | Score Actual | ✓ |
|---------|------|--------|------------|-------------|---|
| FC Bayern München vs FC Union Berlin | H | H | 2-0 | 7-0 | ✅ |
| Manchester City vs Sunderland | H | H | 2-0 | 5-3 | ✅ |
| Bayer 04 Leverkusen vs RB Leipzig | H | H | 2-1 | 2-0 | ✅ |

## H. High-Confidence Predictions (>65%)

**3 fixtures**, accuracy: **100.0%** (3/3)

## K. League Breakdown

| League | N | 1X2 Acc | Exact Acc | Brier | Log Loss |
|--------|---|---------|-----------|-------|----------|
| Bundesliga | 9 | 66.7% | 0.0% | 0.5281 | 0.9037 |
| La Liga | 10 | 60.0% | 20.0% | 0.6044 | 1.0085 |
| Ligue 1 | 9 | 77.8% | 0.0% | 0.5370 | 0.9106 |
| Premier League | 10 | 40.0% | 10.0% | 0.6040 | 1.0088 |
| Serie A | 10 | 40.0% | 10.0% | 0.6100 | 1.0210 |

## L. Biggest Misses (Top 15 by Confidence)

| Fixture | Pred | Actual | P(pred) | Score Pred | Score Actual | League |
|---------|------|--------|---------|------------|-------------|--------|
| Athletic Club vs Deportivo Alavés | **H** | D | 51.5% | 1-0 | 0-0 | La Liga |
| Osasuna vs Rayo Vallecano | **H** | D | 48.9% | 1-1 | 1-1 | La Liga |
| Frosinone vs Como | **A** | H | 46.3% | 1-1 | 2-0 | Serie A |
| Leeds United vs Crystal Palace | **H** | D | 45.3% | 1-1 | 0-0 | Premier League |
| Nottingham Forest vs Coventry City | **H** | A | 44.9% | 1-1 | 0-1 | Premier League |
| Bologna vs Torino | **H** | D | 44.3% | 1-0 | 1-1 | Serie A |
| Espanyol vs Elche | **H** | A | 44.1% | 1-1 | 1-3 | La Liga |
| VfB Stuttgart vs Borussia Dortmund | **H** | A | 43.3% | 1-1 | 0-1 | Bundesliga |
| Eintracht Frankfurt vs SC Freiburg | **A** | D | 43.0% | 1-1 | 2-2 | Bundesliga |
| Brighton & Hove Albion vs Arsenal | **A** | H | 42.7% | 1-1 | 3-0 | Premier League |
| Venezia vs Lazio | **H** | A | 41.9% | 1-1 | 0-2 | Serie A |
| Schalke 04 vs Elversberg | **H** | D | 40.9% | 1-1 | 0-0 | Bundesliga |
| Roma vs Inter | **H** | D | 39.8% | 1-1 | 2-2 | Serie A |
| Udinese vs Cagliari | **H** | A | 39.5% | 1-1 | 0-1 | Serie A |
| Nice vs LOSC Lille | **A** | H | 39.1% | 1-1 | 2-1 | Ligue 1 |

## N. Phase 14 Canonical Fix Validation

- Fixtures using `canonical_predicted_score`: **48**
- Predictions of `2-0`: **2**
- Exact `2-0` hits: **0**

## Q. Model Change Decision

> **HOLD — analysis only. No retraining warranted from a single cohort.**

With 48 fixtures, accuracy=56.2%. Systematic analysis of multiple cohorts required before any model update.

---

## Appendix — Frozen V4 Model

```
model_v4_sha256: 1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5
```

*This audit is read-only. V4 model was NOT modified, retrained, or re-parameterized.*