# 06 — Production Promotion Report: V4.1 Prospective Production Candidate

## 1. Executive Summary & Governance Transition

As of **2026-08-24T03:58:00Z**, the **V4.1 Candidate** (`v4_1_prospective_candidate`) has been formally promoted to the Football Prediction Lab Dashboard as the **Active Frozen Production Model**.

- **Previous Production Model:** `v4_0_draw_champion` (Version: `v4.0-champion-dc-elo-stacking`)
- **New Production Model:** `v4_1_prospective_candidate` (Version: `v4.1-champion-dc-elo-stacking-2025-26-trained`)
- **Benchmark Baseline Model:** `v4_0_draw_champion` (Version: `v4.0-champion-dc-elo-stacking` — **PRESERVED AS FROZEN BASELINE**)
- **Prospective Test Set:** 2026 Live Matches (Active Ongoing Prospective Validation — **STRICTLY UNSEEN**)

> [!IMPORTANT]
> **Governance Invariant:** V4.1 has **NOT** yet been declared statistically superior to V4.0 because the 2026 live prospective evaluation is currently active and ongoing. V4.0 remains frozen, immutable, and accessible in the dashboard as a benchmark reference.

---

## 2. Cryptographic Provenance & Hash Verification

| Model Identity | Role | Version Tag | Artifact Path | MD5 Hash | Status |
|:---|:---:|:---:|:---|:---:|:---:|
| **`v4_1_prospective_candidate`** | **PRODUCTION** | `v4.1-champion-dc-elo-stacking-2025-26-trained` | `data/models/v4_1_prospective_candidate_2025_26.pkl` | **`145f918d933eb343c0f63ca342b10289`** | **FROZEN PRODUCTION** |
| **`v4_0_draw_champion`** | **BENCHMARK** | `v4.0-champion-dc-elo-stacking` | `data/models/v4_poisson_venue_elo_online_ad.pkl` | **`06841f0c03c8597b2b8cd8f8ab064864`** | **FROZEN BENCHMARK** |

---

## 3. Files Changed During Promotion

1. [`src/dashboard/model_registry.py`](file:///e:/Football%20Prediction%20Project/src/dashboard/model_registry.py):
   - Registered `v4_1_prospective_candidate` as the primary production model.
   - Updated `get_production_model()` to return V4.1.
   - Preserved `v4_0_draw_champion` as the benchmark baseline model (`get_baseline_model()`).
   - Added strict startup fail-closed MD5 verification against `145f918d933eb343c0f63ca342b10289`.
2. [`src/dashboard/prediction_service.py`](file:///e:/Football%20Prediction%20Project/src/dashboard/prediction_service.py):
   - Added V4.1 candidate loading and strict MD5 verification.
   - Switched production inference pipeline to V4.1 (`preprocessor` + `model_home_goals` + `model_away_goals` + Dixon-Coles $\rho = -0.08$).
   - Enriched `SingleMatchPredictionResult` and `DashboardMatchPrediction` with explicit V4.1 metadata fields.
   - Retained V4.0 predictions in response as benchmark reference.
3. [`src/dashboard/app.py`](file:///e:/Football%20Prediction%20Project/src/dashboard/app.py):
   - Updated sidebar, Overview, Single Match, and Today's Matches tabs to display V4.1 production probabilities, decision, confidence, and provenance metadata.
4. [`tests/test_v4_1_production_promotion.py`](file:///e:/Football%20Prediction%20Project/tests/test_v4_1_production_promotion.py):
   - Automated test suite verifying V4.1 default routing, metadata exposure, deterministic inference, simplex satisfaction, and benchmark preservation.

---

## 4. Protected Repository Assets & Prospective Ledger Integrity

- **20 Protected Baseline Assets:** **20/20 Verified 100% Bit-Identical**.
- **99 Locked 2026 Prospective Predictions:** [`02_live_forecast_ledger.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/v4_1_prospective_test/02_live_forecast_ledger.csv) is **100% UNTOUCHED** and permanently references `145f918d933eb343c0f63ca342b10289`.
- **Zero Retraining:** Zero model weights, feature transformations, or parameters were altered.

---

## 5. Verification Test Suite Summary

```
============================= test session starts =============================
tests/test_v4_1_production_promotion.py ......                           [ 30%]
tests/test_v4_1_live_forecast_lock.py .......                            [ 65%]
tests/test_v4_1_information_clock.py ....                                [ 85%]
tests/test_v4_1_prospective_training.py ...                              [100%]
======================== 20 passed in 60.09s ==================================
```

---

## 6. Final Production Governance State

$$\begin{aligned}
\mathbf{Production \; Model:} & \quad \mathbf{v4\_1\_prospective\_candidate} \\
\mathbf{Production \; Version:} & \quad \mathbf{v4.1\text{-}champion\text{-}dc\text{-}elo\text{-}stacking\text{-}2025\text{-}26\text{-}trained} \\
\mathbf{Production \; Status:} & \quad \mathbf{FROZEN \; PRODUCTION \; — \; PROSPECTIVE \; LIVE \; EVALUATION \; ACTIVE} \\
\mathbf{Baseline \; Model:} & \quad \mathbf{v4\_0\_draw\_champion} \\
\mathbf{Baseline \; Version:} & \quad \mathbf{v4.0\text{-}champion\text{-}dc\text{-}elo\text{-}stacking} \\
\mathbf{Baseline \; Status:} & \quad \mathbf{FROZEN \; BASELINE \; / \; BENCHMARK} \\
\mathbf{2026 \; Live \; Matches:} & \quad \mathbf{UNSEEN \; PROSPECTIVE \; TEST \; SET}
\end{aligned}$$
