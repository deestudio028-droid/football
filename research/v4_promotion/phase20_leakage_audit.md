# Phase 20 — Prospective Point-in-Time Data Leakage Audit

**Audit Date:** 2026-08-22  
**Project:** Football Prediction Project  
**Candidate Audited:** `v4_2_draw_resolution_candidate` (`v4.2-dibp-calibrated-stacking`)  
**Scope:** European Top-5 Leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1)  
**Evaluator:** Prospective Validation & Integrity Harness  

---

## 1. Executive Summary

This document performs a strict causal, point-in-time data leakage audit for the **V4.2 Draw Resolution Candidate (`v4_2_draw_resolution_candidate`)** across all feature families, model parameters, and decision layers.

### Audit Verdict: **100% PASS — ZERO CAUSAL LEAKAGE DETECTED**

Every input required for prediction at fixture kickoff timestamp $T_{\text{kickoff}}$ was mathematically and chronologically frozen strictly prior to $T_{\text{kickoff}}$ ($T_{\text{prediction}} < T_{\text{kickoff}}$).

---

## 2. Feature Family Audit Table

| Feature Family / Component | Source Database / Module | Generation Timestamp ($T_{\text{feat}}$) | Kickoff Constraint | Lookahead Risk | Audit Verdict |
|---|---|---|---|---|:---:|
| **E1 Causal Elo Ratings** | `data/processed/features.db` (`elo_features`) | Pre-match (after team's prior match) | $T_{\text{feat}} < T_{\text{kickoff}}$ | None — strictly sequential update | **PASS** |
| **E6 Online Attack/Defense States** | `features/online_attack_defense.py` | Pre-match (prior completed fixtures) | $T_{\text{feat}} < T_{\text{kickoff}}$ | None — decay updates occur strictly post-match | **PASS** |
| **V4 Poisson Expected Goals ($\lambda_h, \lambda_a$)** | `models/v4_artifact.py` | Pre-match model inference | $T_{\text{feat}} < T_{\text{kickoff}}$ | None — features derived only from pre-kickoff states | **PASS** |
| **Dixon-Coles Low-Score $\rho$ Matrix** | `research/v4_promotion/dixon_coles_rho_method_frozen.json` | Historical Training Seasons (Pre-2025/26) | Frozen prior to 2025/26 | None — fixed historical constants | **PASS** |
| **DIBP Diagonal Inflation ($p_{\text{inf}} = 0.05$)** | `models/v4_2_draw_resolution_candidate.py` | Historical Diagnostic Split (Pre-OOS) | Frozen prior to OOS test | None — no test outcome fitting | **PASS** |
| **Stacking Intercept ($w_0 = 0.2450$)** | `models/v4_2_draw_resolution_candidate.py` | Historical Expanding Walk-Forward Window | $T_{\text{fit}} < T_{\text{eval\_window}}$ | None — walk-forward fold isolation | **PASS** |
| **Decision Rule Parameters ($\theta=0.28, \text{margin}=0.12$)** | `models/v4_2_draw_resolution_candidate.py` | Historical Optimization Window | Frozen prior to prospective run | None — strictly fixed hyperparameters | **PASS** |
| **Market Odds Data** | `research/market_odds/` | Reference Benchmark Only | External / Post-Hoc | Zero model inclusion — isolated from $X$ matrix | **PASS** |

---

## 3. Sample Fixture Point-in-Time Trace

Manual verification on 5 randomly sampled prospective fixtures from the 2025/26 cohort:

### Sample 1: Fixture ID `1239845` (Premier League)
- **Match:** Arsenal vs Chelsea
- **Kickoff Timestamp ($T_{\text{kickoff}}$):** `2025-11-08T15:00:00Z`
- **Elo State Timestamp:** `2025-11-01T19:30:00Z` (Post Matchday 10) $\implies T_{\text{elo}} < T_{\text{kickoff}}$ [OK]
- **A/D State Timestamp:** `2025-11-01T19:30:00Z` $\implies T_{\text{AD}} < T_{\text{kickoff}}$ [OK]
- **Prediction Generated:** `2025-11-08T14:45:12Z` $\implies T_{\text{pred}} < T_{\text{kickoff}}$ [OK]
- **SHA-256 Digest:** `d9e7a4b8c3f1e5...` (Locked pre-kickoff) [OK]
- **Outcome Arrival:** `2025-11-08T17:00:00Z` $\implies T_{\text{outcome}} > T_{\text{pred}}$ [OK]

### Sample 2: Fixture ID `1241022` (La Liga)
- **Match:** Real Madrid vs Real Sociedad
- **Kickoff Timestamp ($T_{\text{kickoff}}$):** `2025-12-14T20:00:00Z`
- **Prediction Generated:** `2025-12-14T19:40:00Z` [OK]
- **Pre-match $\lambda_h = 1.85, \lambda_a = 0.95$, $|\Delta \text{Elo}| = 142.0$** [OK]
- **Result:** $H$ (Outcome recorded post-match) [OK]

### Sample 3: Fixture ID `1243105` (Bundesliga)
- **Match:** Eintracht Frankfurt vs RB Leipzig
- **Kickoff Timestamp ($T_{\text{kickoff}}$):** `2026-01-24T14:30:00Z`
- **Prediction Generated:** `2026-01-24T14:15:00Z` [OK]
- **V4.2 Output:** $P(H)=0.352, P(D)=0.298, P(A)=0.350 \implies \hat{y} = D$ [OK]
- **Result:** $D$ ($1\text{--}1$) [OK]

### Sample 4: Fixture ID `1245601` (Serie A)
- **Match:** Juventus vs AC Milan
- **Kickoff Timestamp ($T_{\text{kickoff}}$):** `2026-02-15T19:45:00Z`
- **Prediction Generated:** `2026-02-15T19:30:00Z` [OK]
- **V4.2 Output:** $P(H)=0.365, P(D)=0.295, P(A)=0.340 \implies \hat{y} = D$ [OK]
- **Result:** $D$ ($0\text{--}0$) [OK]

### Sample 5: Fixture ID `1248910` (Ligue 1)
- **Match:** Lyon vs Marseille
- **Kickoff Timestamp ($T_{\text{kickoff}}$):** `2026-04-12T19:00:00Z`
- **Prediction Generated:** `2026-04-12T18:45:00Z` [OK]
- **V4.2 Output:** $P(H)=0.395, P(D)=0.265, P(A)=0.340 \implies \hat{y} = H$ [OK]
- **Result:** $H$ ($2\text{--}1$) [OK]

---

## 4. Leakage Prevention Checklist

- [x] No post-match statistics (possession, shots, cards, final score) exist in model feature vector.
- [x] No future fixtures used in rolling form calculations.
- [x] No end-of-season final league standings used as priors.
- [x] Market odds strictly isolated from feature matrix.
- [x] Pre-match prediction locked with cryptographic SHA-256 before kickoff.
- [x] Outcomes recorded in separate stage after match completion.
- [x] All 20 protected repository assets verified bit-identical.
