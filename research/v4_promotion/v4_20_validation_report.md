# V4 20-Match Validation Report
## Retrospective Evaluation: V2 vs V3 vs V4 vs Pinnacle Market
**Execution Date:** 2026-08-20T15:35:15.594194+00:00  
**Status:** VALIDATION COMPLETE — STRICT EVALUATION ONLY (DO NOT PROMOTE)

---

## Executive Summary

The V4 candidate model (`v4_poisson_venue_elo_online_ad.pkl`, MD5 `06841f0c03c8597b2b8cd8f8ab064864`), which adds E6 Online Attack/Defense features to V3's causal Elo Poisson architecture (91 total features), was evaluated across the fixed, deterministic opening-weekend 20-fixture sample from the 2025/26 season.

Evaluation benchmark includes:
1. **V2 Production Champion:** Poisson + Venue (84 features)
2. **V3 Candidate:** Poisson + Venue + Causal Elo (87 features)
3. **V4 Candidate:** Poisson + Venue + Causal Elo + Online A/D (91 features)
4. **Pinnacle Market:** De-vigged closing 1X2 market probabilities (reference signal)

---

## 20-Match Scorecard

| Model | Correct | Accuracy (↑) | Log Loss (↓) | Brier Score (↓) | RPS (↓) | Draw Recall (↑) | ECE (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2** | 9/20 | 0.4500 | 1.05545 | 0.63504 | 0.22411 | 0.0000 | 0.0326 |
| **V3** | 10/20 | 0.5000 | 1.02912 | 0.61516 | 0.21488 | 0.0000 | 0.0435 |
| **V4** | 10/20 | 0.5000 | 1.01890 | 0.60749 | 0.21166 | 0.0000 | 0.0483 |
| **Market** | 8/20 | 0.4000 | 0.99146 | 0.58613 | 0.20427 | 0.0000 | 0.1438 |

*(↑) indicates higher is better; (↓) indicates lower is better.*

---

## Pairwise Model Comparisons

*Convention: Positive delta indicates V4 is better; negative indicates V4 is worse.*

### V4 vs V2 (Champion)
- **Log Loss Delta:** +0.036545 (V4 better)
- **Brier Delta:** +0.027549 (V4 better)
- **RPS Delta:** +0.012455 (V4 better)
- **ECE Delta:** -0.015747
- **Accuracy Delta:** +0.0500 (+1 matches)

### V4 vs V3 (Elo Candidate)
- **Log Loss Delta:** +0.010221 (V4 better)
- **Brier Delta:** +0.007671 (V4 better)
- **RPS Delta:** +0.003224 (V4 better)
- **ECE Delta:** -0.004824
- **Accuracy Delta:** +0.0000 (+0 matches)

### V4 vs Pinnacle Market (Reference)
- **Log Loss Delta:** -0.027439 (Market has lower log loss by 0.02744)
- **Brier Delta:** -0.021362 (Market has lower Brier by 0.02136)
- **RPS Delta:** -0.007388 (Market has lower RPS by 0.00739)
- **Accuracy Delta:** +0.1000 (V4 correct on 10/20 vs Market 8/20)

---

## Prediction Shifts

- **V2 → V4 Flips:** 1 fixtures
  - Fixture #08 [342863844] Tottenham Hotspur vs Burnley: V2 `A` → V4 `H` (Actual: `H`)

- **V3 → V4 Flips:** 0 fixtures

---

## Per-League Analysis

| League | Sample Size | V2 Acc | V3 Acc | V4 Acc | Market Acc | V4 Log Loss | Market Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ligue 1** | 6 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 1.11071 | 1.09946 |
| **La Liga** | 6 | 0.6667 | 0.6667 | 0.6667 | 0.5000 | 0.94561 | 0.97519 |
| **Premier League** | 8 | 0.5000 | 0.6250 | 0.6250 | 0.5000 | 1.00501 | 0.92267 |

> [!NOTE]
> **League Scope Limitation:** The 20 validation fixtures represent only Premier League (8), La Liga (6), and Ligue 1 (6). Bundesliga (477) and Serie A (499) have 0 observations in this opening-weekend sample.

---

## Case-by-Case Breakdown

| # | Match | Actual | V2 | V3 | V4 | Market | Key Dynamic |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :--- |
| 01 | Girona vs Rayo Vallecano (1-3) | **A** | A | A | **A** | H | V4 beat Market |
| 02 | Rennes vs Olympique Marseille (1-0) | **H** | A | A | **A** | A | All models wrong |
| 03 | Liverpool vs AFC Bournemouth (4-2) | **H** | H | H | **H** | H | Consensus |
| 04 | Villarreal vs Real Oviedo (2-0) | **H** | H | H | **H** | H | Consensus |
| 05 | Aston Villa vs Newcastle United (0-0) | **D** | A | A | **A** | H | All models wrong |
| 06 | Brighton & Hove Albion vs Fulham (1-1) | **D** | H | H | **H** | H | All models wrong |
| 07 | Sunderland vs West Ham United (3-0) | **H** | H | H | **H** | A | V4 beat Market |
| 08 | Tottenham Hotspur vs Burnley (3-0) | **H** | A | H | **H** | H | V4 improved over V2 |
| 09 | Lens vs Olympique Lyonnais (0-1) | **A** | H | H | **H** | H | All models wrong |
| 10 | Wolverhampton Wanderers vs Manchester City (0-4) | **A** | A | A | **A** | A | Consensus |
| 11 | Monaco vs Le Havre (3-1) | **H** | H | H | **H** | H | Consensus |
| 12 | Mallorca vs FC Barcelona (0-3) | **A** | A | A | **A** | A | Consensus |
| 13 | Nice vs Toulouse (0-1) | **A** | H | H | **H** | H | All models wrong |
| 14 | Deportivo Alavés vs Levante (2-1) | **H** | H | H | **H** | H | Consensus |
| 15 | Valencia vs Real Sociedad (1-1) | **D** | H | H | **H** | H | All models wrong |
| 16 | Nottingham Forest vs Brentford (3-1) | **H** | H | H | **H** | H | Consensus |
| 17 | Chelsea vs Crystal Palace (0-0) | **D** | H | H | **H** | H | All models wrong |
| 18 | Brest vs LOSC Lille (3-3) | **D** | H | H | **H** | H | All models wrong |
| 19 | Celta de Vigo vs Getafe (0-2) | **A** | H | H | **H** | H | All models wrong |
| 20 | Angers SCO vs Paris (1-0) | **H** | A | A | **A** | A | All models wrong |

---

## Key Diagnostic Categories

1. **V4-Only Wins (vs V2 & V3):** 0
2. **V4-Only Losses (vs V2 & V3):** 0
3. **V4 and Market Both Correct:** 8
4. **Market Correct while V4 Wrong:** 0
5. **V4 Correct while Market Wrong:** 2
6. **All Models Wrong (inc. Market):** 10

---

## Opening-Weekend Context & Sample Caveat

These 20 fixtures represent the opening weekend of the 2025/26 season across Premier League, La Liga, and Ligue 1. Because these are earliest-season matches, teams have zero within-season rolling history for 2025/26. Rolling form features are at their lowest informational density, relying heavily on prior-season carryover via E1 causal Elo and E6 Online Attack/Defense. This is structural context for the sample and must not be used to selectively filter or tune the validation results.

---

## Validation Integrity & Causality Verification

- **Protected Files Integrity:** All 5 core model and data files verified byte-identical.
- **E2 Research Artifacts:** Both E2 SQLite databases verified byte-identical.
- **V4 Artifact Checksum:** Verified exact build hash (`06841f0c03c8597b2b8cd8f8ab064864`).
- **Information Cutoff:** Verified 0 rows of 2025/26 in training fit.
- **Causality & Two-Pass Engine:** Verified score-rewrite invariance.
- **Determinism:** Bit-identical predictions on repeated runs (`max diff = 0.000e+00`).

---

## Promotion Decision

> [!CAUTION]
> **DO NOT PROMOTE TO PRODUCTION AUTOMATICALLY.**
> This report is an isolated validation harness. No production files (`predict_match.py`, `data/models/v2_poisson_venue.pkl`) have been altered. Any promotion of V4 requires human review and explicit authorization.
