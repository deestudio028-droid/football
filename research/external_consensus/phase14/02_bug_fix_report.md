# Phase 14 — Bug Fix Report: Production 2-0 / Strong Home Pathway

## 1. Overview & Objective
The objective of Phase 14 is to safely eliminate all scoreline flattening, rounding distortions, and heuristic shortcuts in the production prediction pathway without modifying the frozen V4 model binary, model parameters, or probability distributions.

This document details the engineering modifications made across the codebase.

---

## 2. File-by-File Technical Modifications

### 2.1 `src/dashboard/prediction_snapshot_store.py`
- **Problem**: Pre-kickoff locked snapshots stored $\lambda_h, \lambda_a$ and probabilities, but omitted explicit canonical scoreline and signal classification fields.
- **Modifications**:
  1. Added typed attributes to `PredictionSnapshot`:
     - `canonical_predicted_score: Optional[str] = None`
     - `predicted_score: Optional[str] = None`
     - `modal_scoreline_probability: Optional[float] = None`
     - `strong_home_profile: bool = False`
     - `is_2_0_profile: bool = False`
     - `signal_profile: Optional[str] = None`
  2. Implemented `__post_init__` hook:
     - Automatically derives `canonical_predicted_score` using `modal_scoreline(lambda_home, lambda_away)` if not explicitly provided.
     - Automatically derives `strong_home_profile` ($P(H) \ge 0.60 \land \text{Draw Risk} == \text{'LOW'}$).
     - Automatically derives `is_2_0_profile` ($\text{canonical\_score} == \text{'2-0'} \land \text{Draw Risk} == \text{'LOW'}$).
     - Assigns `signal_profile = "2-0_PROFILE"` or `"STRONG_HOME_PROFILE"` or `None`.
  3. **Backward Compatibility**: Fully backward compatible with all pre-existing serialized JSON ledgers and CSV files.

### 2.2 `src/dashboard/prediction_service.py`
- **Problem**: Neither `SingleMatchPredictionResult` nor `DashboardMatchPrediction` propagated canonical scorelines, causing downstream callers to invent rounding heuristics.
- **Modifications**:
  1. Imported `modal_scoreline` and `_grid_size` from `models.poisson`.
  2. Added canonical scoreline and signal profile fields to `SingleMatchPredictionResult` and `DashboardMatchPrediction`.
  3. In `predict_matchup()`:
     - For V4.0 inference, computed exact bivariate Poisson mode:
       ```python
       K_v4 = _grid_size(float(max(lh_v4, la_v4)), 1e-4)
       sh_v4, sa_v4, sp_v4 = modal_scoreline(np.array([lh_v4]), np.array([la_v4]), K_v4)
       v4_modal_score = f"{int(sh_v4[0])}-{int(sa_v4[0])}"
       ```
     - Computed profile flags: `strong_home_profile`, `is_2_0_profile`, `signal_profile`.
     - Injected these fields into the return object.
  4. In `predict_dashboard_fixture()`:
     - Updated Path B (historical/evaluated fixtures): extracted canonical scoreline and profile flags from snapshot.
     - Updated Path A (upcoming live fixtures): propagated canonical scoreline and profile flags from `predict_matchup()`.
  5. Added `get_prediction_service()` singleton getter for thread-safe shared usage.

### 2.3 `src/dashboard/fixture_service.py`
- **Problem**: Lines 840–848 rounded expected goals (`int(round(lh))`, `int(round(la))`), turning `2-0` into `2-1` whenever $\lambda_a \in [0.50, 1.00)$.
- **Modifications**:
  1. Replaced rounding block with:
     ```python
     pred_score = getattr(snap, "canonical_predicted_score", None) or getattr(snap, "predicted_score", None)
     if not pred_score:
         K = _grid_size(float(max(lh, la)), 1e-4)
         sh, sa, _ = modal_scoreline(np.array([lh]), np.array([la]), K)
         pred_score = f"{int(sh[0])}-{int(sa[0])}"
     ```
  2. Extracted signal classification attributes:
     - `is_sh = bool(p_h >= 0.60 and d_tier == "LOW")`
     - `is_20 = bool(pred_score == "2-0" and d_tier == "LOW")`
     - `sig_prof = "2-0_PROFILE" if is_20 else ("STRONG_HOME_PROFILE" if is_sh else None)`
  3. Added `"canonical_predicted_score"`, `"strong_home_profile"`, `"is_2_0_profile"`, `"signal_profile"`, and `"Signal"` to records dictionary.

### 2.4 `src/dashboard/app.py`
- **Problem**: Lines 636–644 applied UI-level rounding of expected goals.
- **Modifications**:
  1. Prioritized `pred.canonical_predicted_score` and `pred.predicted_score`.
  2. Added fallback to `modal_scoreline(pred.lambda_home, pred.lambda_away)`.
  3. Added `"Signal Profile"` column to fixture display table (`🟢 2-0 Signal` / `🔵 Strong Home`).

### 2.5 `src/dashboard/performance_monitor_service.py`
- **Problem**: Derived baseline scores via heuristic: `base_score = "2-1" if p_h < 0.60 else "2-0"`.
- **Modifications**:
  1. Replaced heuristic with inspection of `canonical_predicted_score` or `predicted_score`.
  2. Added fallback to `modal_scoreline(lambda_home, lambda_away)` if $\lambda$ values are present.

### 2.6 `src/dashboard/batch_prediction_service.py` (New Service)
- **Problem**: Previous batch runners lacked standard architecture and defaulted to flattened scorelines (`HOME -> 2-1`, `AWAY -> 1-2`).
- **Modifications**:
  1. Implemented `BatchPredictionService` with mandatory SHA256 verification of frozen V4 binary.
  2. Generates canonical prediction records containing full Poisson modes, probabilities, draw risk metrics, and signal classifications.
  3. Supports atomic persistence to JSONL ledgers and immutable snapshot stores.

---

## 3. Invariant Verification

| Invariant | Target Value | Verified Status |
|---|---|---|
| Frozen V4 Model Binary | `data/models/v4_poisson_venue_elo_online_ad.pkl` | Untouched |
| Frozen V4 SHA256 Hash | `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5` | Verified Bit-Identical |
| Model Retraining / Parameter Tuning | None | 100% Zero Retraining |
| Probability Distributions | Pure V4 Inference | 100% Unmodified |
| Snapshot Store Backward Compatibility | All historical snapshots load cleanly | Verified |

---

## 4. Verification Summary
All downstream prediction pathways now query the canonical mathematical bivariate Poisson mode. `2-0` predictions for heavy home favorites are fully restored and consistently propagated across UI, evaluation tables, performance monitoring, and batch generation ledgers.
