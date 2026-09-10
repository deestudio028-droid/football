# Phase 33 — Promoted Team Pre-Match Feature Initialization Report

## 1. Executive Summary

In Phase 33, a mathematically rigorous, causal pre-match feature initialization engine ([`src/features/promoted_team_initializer.py`](file:///e:/Football%20Prediction%20Project/src/features/promoted_team_initializer.py)) was integrated into the Football Prediction Lab inference pipeline.

This mechanism resolves the promoted-team gap without modifying frozen models, mutating data artifacts, or re-introducing generic static fallbacks (such as $1.45/1.15$ or $44/26/30\%$).

All 17 discovered live fixtures for **2026-08-22** across the 5 top European leagues now receive distinct, fixture-specific probability vectors derived directly through the frozen production $V_4$ Poisson + Venue + Elo + Online AD preprocessor and regressors.

---

## 2. Promoted Team Initialization Architecture

### A. Causal Discovery & Profile Construction
When a match involves a newly promoted club entering the top flight with $N=0$ current-season matches in `matches.db`:
1. **Prior Top-Flight Match Check**: The database is queried for historical top-flight matches in prior seasons.
   - If historical matches exist, pre-match Elo rating and Online Attack/Defense states are recovered up to the historical cutoff.
2. **Conservative Baseline Initialization**: If no prior top-flight matches exist in the historical database:
   - Initial Elo: $R_0 = 1500.0$ (`INIT_RATING`)
   - Initial Online Attack/Defense: $(A_0, D_0) = (0.0, 0.0)$ (league average baseline)
   - Rolling Form / Match Counts: Initialized to missingness-encoded indicator state (`sample_size_matches_n = 0`, `insufficient_history = 1`, `matches_played = 0`).
3. **Strict Interaction with Opponent Context**:
   - The promoted team's causal baseline is coupled with the opponent's true top-flight pre-match state (e.g. Manchester United: $\text{Elo} = 1640.8, A = 0.242, D = 0.149$; Brest: $\text{Elo} = 1516.2, A = 0.051, D = 0.082$).
   - The exact 91-feature contract vector is processed through `v4.preprocessor.transform(X)` and fed into `v4.model_home_goals` and `v4.model_away_goals`.

### B. Fail-Closed Guarantee
If an invalid, empty, or unresolvable team is supplied, the pipeline immediately returns:
$$\text{"Prediction unavailable — required pre-match features unavailable"}$$
Zero synthetic or generic default probabilities are ever generated.

---

## 3. Live Matchday Validation (2026-08-22 OddAlerts Feed)

| Fixture ID | Competition | Home Team | Away Team | $P(\text{Home})$ | $P(\text{Draw})$ | $P(\text{Away})$ | $V_4$ Dec | $V_{4.6}$ Dec | Prediction Source |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **420581845** | Premier League | **Hull City** 🔰 | Manchester United | **34.6%** | **25.4%** | **40.1%** | **A** | **A (Retained)** | **Promoted Init** |
| 420582048 | Premier League | Nottingham Forest | Leeds United | 44.7% | 25.4% | 29.9% | H | H (Retained) | Top Flight |
| 420582049 | Premier League | Ipswich Town | Sunderland | 31.0% | 25.8% | 43.2% | A | A (Retained) | Top Flight |
| 420582050 | Premier League | Everton | Crystal Palace | 43.8% | 25.9% | 30.3% | H | H (Retained) | Top Flight |
| 420582189 | La Liga | Athletic Club | Sevilla | 45.1% | 25.9% | 29.1% | H | H (Retained) | Top Flight |
| 420582220 | Ligue 1 | Lens | Auxerre | 59.2% | 21.8% | 18.9% | H | H (Retained) | Top Flight |
| 420582254 | Serie A | Udinese | Como | 29.8% | 26.1% | 44.1% | A | A (Retained) | Top Flight |
| 420582255 | Serie A | Inter | Monza | 74.1% | 16.2% | 9.7% | H | H (Retained) | Top Flight |
| 420582256 | Premier League | Brentford | Tottenham Hotspur | 41.5% | 25.8% | 32.7% | H | H (Retained) | Top Flight |
| 420582269 | La Liga | Valencia | Celta de Vigo | 47.4% | 25.5% | 27.2% | H | H (Retained) | Top Flight |
| 420582305 | Ligue 1 | Nice | Lorient | 38.8% | 25.9% | 35.3% | H | H (Retained) | Top Flight |
| **420582306** | Ligue 1 | **Le Mans** 🔰 | Brest | **46.6%** | **25.1%** | **28.3%** | **H** | **H (Retained)** | **Promoted Init** |
| 420582304 | Ligue 1 | Toulouse | Olympique Lyonnais | 34.9% | 25.1% | 40.0% | A | A (Retained) | Top Flight |
| 420582303 | Ligue 1 | Troyes | Paris | 22.5% | 23.0% | 54.5% | A | A (Retained) | Top Flight |
| 420582302 | Serie A | Genoa | Napoli | 27.5% | 25.9% | 46.5% | A | A (Retained) | Top Flight |
| 420582301 | Serie A | Parma | Cagliari | 35.4% | 27.0% | 37.6% | A | A (Retained) | Top Flight |
| 420582312 | La Liga | Espanyol | Real Madrid | 25.3% | 24.5% | 50.2% | A | A (Retained) | Top Flight |

- **Total Live Fixtures Available:** 17 / 17 (100% Coverage)
- **Total Distinct Probability Vectors:** 17 / 17 (Zero identical fallbacks)

---

## 4. Promoted Team Details

### 1. Hull City (Premier League) vs Manchester United
- **Fixture ID:** `420581845`
- **Home Team:** Hull City (ID: `9770`)
- **Away Team:** Manchester United (ID: `10659`)
- **Initialization Method:** `Conservative Promotion Baseline (Initial Elo 1500.0 + League Neutral AD)`
- **Pre-Match Metrics:**
  - $\text{Home Elo} = 1500.0, \quad \text{Away Elo} = 1640.8, \quad \Delta\text{Elo} = -40.8$
  - $\text{Home Attack/Defense} = (0.000, 0.000), \quad \text{Away Attack/Defense} = (0.242, 0.149)$
  - $\lambda_{\text{home}} = 1.316, \quad \lambda_{\text{away}} = 1.439$
- **Probabilities:** $P(H) = 34.6\%, \quad P(D) = 25.4\%, \quad P(A) = 40.1\%$
- **$V_4$ Decision:** Away Win (`A`)

### 2. Le Mans (Ligue 1) vs Brest
- **Fixture ID:** `420582306`
- **Home Team:** Le Mans (ID: `5722`)
- **Away Team:** Brest (ID: `8674`)
- **Initialization Method:** `Conservative Promotion Baseline (Initial Elo 1500.0 + League Neutral AD)`
- **Pre-Match Metrics:**
  - $\text{Home Elo} = 1500.0, \quad \text{Away Elo} = 1516.2, \quad \Delta\text{Elo} = +83.8$
  - $\text{Home Attack/Defense} = (0.000, 0.000), \quad \text{Away Attack/Defense} = (0.051, 0.082)$
  - $\lambda_{\text{home}} = 1.547, \quad \lambda_{\text{away}} = 1.147$
- **Probabilities:** $P(H) = 46.6\%, \quad P(D) = 25.1\%, \quad P(A) = 28.3\%$
- **$V_4$ Decision:** Home Win (`H`)

---

## 5. Governance & Immutability Verification

### Protected Repository Baseline Hashes (20/20 Bit-Identical)
1. `data/models/v4_poisson_venue_elo_online_ad.pkl`: `06841f0c03c8597b2b8cd8f8ab064864` ✅
2. `data/models/v3_poisson_venue_elo_candidate.pkl`: `a2850a7687822a5916663301f5ccc96c` ✅
3. `data/models/v2_poisson_venue.pkl`: `25935b4e93fc4074f67f16e3181ed4df` ✅
4. `data/models/v1_logreg.pkl`: `5e504427712b35778bb8a62a8496c7cd` ✅
5. `data/processed/matches.db`: `fdeed042096fa1c851aaee6c84995247` ✅
6. `data/processed/features.db`: `e7ebe7fc07040a5927683c35b6371e63` ✅
7. `research/market_odds/odds_history.sqlite`: `0be31e8b59d739b72c3fb48e555d9fd8` ✅
8. `research/market_odds/research_dataset.sqlite`: `bdab370ffdfe5bbf8ff3a8a26e64471c` ✅
9. `research/v4_promotion/promotion_market_odds.sqlite`: `f8a41b79cd33afb412ccd9ae2892a196` ✅
10. `research/v4_promotion/fresh_100_market_odds.sqlite`: `2cb80b79d772fbedd4f3707b39a32c13` ✅
11. `research/v4_promotion/fresh_100_fixture_ids.json`: `761ad5cc571643e6985e671bd9c3d83a` ✅
12. `research/v4_promotion/fresh_extended_fixture_ids.json`: `0526bfd6980dd51dae59c6f6aadab2f5` ✅
13. `research/v4_promotion/fresh_extended_market_odds.sqlite`: `b4889d1791723ea653057af51ca00f8e` ✅
14. `research/v4_promotion/dixon_coles_rho_method_frozen.json`: `822e742dcc82e5e96445b31c14c0c604` ✅
15. `research/v4_promotion/elo_draw_curve_method_frozen.json`: `65dc2cf762f3d78abcf1a617ef23fe00` ✅
16. `research/v4_promotion/full_score_matrix_method_frozen.json`: `cd44e1da88a50ac45e8383557ad5271f` ✅
17. `research/v4_promotion/market_calibration_method_frozen.json`: `550a0e1f1358a8359d7141b422521dd9` ✅
18. `research/v4_promotion/draw_complementarity_method_frozen.json`: `d4f7dc75785df076c105a6ebfc0a4d6e` ✅
19. `research/v4_promotion/temporal_regime_method_frozen.json`: `4a4f72e1d288d2547272c9b30b0368df` ✅
20. `research/v4_promotion/statistical_power_uncertainty_method_frozen.json`: `68d55b30789d40440a0c14cbfe225c7f` ✅

---

## 6. Unit & Regression Test Verification

| Test Suite | Total Tests | Status |
| :--- | :---: | :---: |
| `tests/test_phase33_promoted_team_initialization.py` | 10 | **10 / 10 PASSED** |
| `tests/test_phase32_prediction_specificity.py` | 8 | **8 / 8 PASSED** |
| `tests/test_phase32_dashboard_fixture_api.py` | 12 | **12 / 12 PASSED** |
| `tests/test_phase31_dashboard.py` | 6 | **6 / 6 PASSED** |
| `tests/test_phase30_historical_draw_retraining.py` | 4 | **4 / 4 PASSED** |
| `tests/test_v46_fixture_feed_integration.py` | 7 | **7 / 7 PASSED** |
| `tests/test_v46_operational_collection.py` | 4 | **4 / 4 PASSED** |
| `tests/test_v46_live_prospective_collection.py` | 6 | **6 / 6 PASSED** |
| `tests/test_v46_outcome_reconciliation.py` | 3 | **3 / 3 PASSED** |
| `tests/test_v4_6_physical_draw_gate.py` | 5 | **5 / 5 PASSED** |
| `tests/test_v4_5_causal_draw_meta.py` | 4 | **4 / 4 PASSED** |
| `tests/test_v4_4_robust_draw_override.py` | 5 | **5 / 5 PASSED** |
| `tests/test_v4_3_accuracy_preserving_draw_override.py` | 5 | **5 / 5 PASSED** |
| `tests/test_v4_2_prospective_integration.py` | 4 | **4 / 4 PASSED** |
| **TOTAL** | **83** | **83 / 83 PASSED (100%)** |
