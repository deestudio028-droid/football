# Phase 27 — V4.6 Live Prospective Collection Operational Status

**Execution Date:** 2026-08-22  
**Project:** Football Prediction Project  
**Authoritative Production Model:** `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`) [100% FROZEN]  
**Authoritative Shadow Model:** `v4_6_physical_draw_gate` (`v4.6-physical-draw-gate`) [100% FROZEN]  
**Live Database:** `research/v4_promotion/live_v46_prospective.sqlite`  
**Target Mandatory Power Gate:** $N \ge 1,050$ Fresh Prospective Fixtures  
**Current Live Progress:** $N = 0$ Completed / Reconciled Fixtures  
**Operational Status:** **ACTIVE — RUNNERS OPERATIONAL & IDLE (WAITING FOR FUTURE FIXTURES)**  

---

## 1. Operational Architecture & Execution Commands

```
                               UPCOMING FIXTURES FEED
                                         │
                                         ▼
                   STEP 1: RUN PRE-KICKOFF LIVE COLLECTION
                 python src/monitoring/run_v46_live_collection.py
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
    PRE-FLIGHT GATES                                        PRE-MATCH PREDICTION
 - Reject historical IDs (1,751 known)                     - V4 Poisson Probabilities
 - Enforce t_pred <= t_kickoff - 15m                       - V4.2 DIBP Draw Prob
 - Enforce V4.6 config freeze                              - V4.6 Physical Gate Decision
             │                                                       │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                             DETERMINISTIC SHA-256 HASH
                         hashlib.sha256(canonical_json)
                                         │
                                         ▼
                         IMMUTABLE ISOLATED SQLITE STORE
                 (research/v4_promotion/live_v46_prospective.sqlite)
                        Table: `live_predictions` (Status: LOCKED)
                                         │
 ════════════════════════════════════════╪═══════════════════════════════════════
                              [ MATCH PLAYED / FT ]
 ════════════════════════════════════════╪═══════════════════════════════════════
                                         │
                                         ▼
                  STEP 2: RUN POST-MATCH OUTCOME RECONCILIATION
                python src/monitoring/run_v46_outcome_reconciliation.py
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
    TAMPER DETECTION GATE                                   OUTCOME RECORDING
 - Re-compute SHA-256 digest                               - Final score (H-A)
 - Compare with stored hash                                - Actual outcome (H/D/A)
 - FAIL CLOSED if mismatch                                 - Table: `match_outcomes`
             │                                                       │
             └───────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                       STEP 3: VIEW LIVE VALIDATION METRICS
                     python -m monitoring.v46_live_metrics
```

---

## 2. Command Execution Guide

### A. Run Daily Fixture Collection
```bash
python src/monitoring/run_v46_live_collection.py --safety-buffer-minutes 15
```
- Discovers future scheduled fixtures from football data sources.
- Rejects any historical fixture, past-kickoff match, or duplicate fixture.
- Locks predictions with SHA-256 digests in `live_v46_prospective.sqlite`.

### B. Run Post-Match Outcome Reconciliation
```bash
python src/monitoring/run_v46_outcome_reconciliation.py
```
- Queries locked predictions whose matches have reached full-time (`FT`).
- Verifies prediction hash integrity.
- Ingests final scores and updates live accuracy/draw metrics.

---

## 3. Live Monitoring Dashboard

```
================================================================================================
V4.6 LIVE PROSPECTIVE MONITORING DASHBOARD
================================================================================================
Cohort Progress:          N = 0 / 1050 (0.0% toward Statistical Power Gate)
Fresh Total Locked:       0 fixtures
Completed / Reconciled:   0 fixtures
Pending Kickoff/Outcome:  0 fixtures
Remaining to N=1,050:     1050 fixtures
Current Status:           COLLECTING (N=0 / 1050)

ACCURACY & SCORECARD:
  V4 Baseline Accuracy:     0.00% (0 / 0)
  V4.6 Candidate Accuracy:  0.00% (0 / 0)
  Delta Accuracy:          +0.00%
  Macro F1 Score:         0.0000

DRAW METRICS:
  Draw Predictions:       0
  Correct Draws:          0
  Draw Precision:            0.0%
  Draw Recall:               0.0%
  Draw F1 Score:          0.0000

ERROR ECONOMICS:
  Good Draw Overrides:    0 (Free Draw Wins)
  Bad Draw Overrides:     0 (Sacrificed V4 True Positives)
  Neutral Overrides:      0 (Neutral Error Shift)
  Net Transition Gain:    +0 net correct predictions
================================================================================================
```

---

## 4. Progressive Milestones

| Milestone ($N$) | Required Sample Size | Progress ($N=0$) | Status | Operational Action |
|---|---:|---:|:---:|---|
| **Milestone 1** | $N = 100$ | $0 / 100$ ($0.0\%$) | PENDING | First live operational checkpoint |
| **Milestone 2** | $N = 300$ | $0 / 300$ ($0.0\%$) | PENDING | Preliminary Draw precision check |
| **Milestone 3** | $N = 450$ | $0 / 450$ ($0.0\%$) | PENDING | Early stability checkpoint |
| **Milestone 4** | $N = 600$ | $0 / 600$ ($0.0\%$) | PENDING | Mid-point error economics check |
| **Milestone 5** | $N = 750$ | $0 / 750$ ($0.0\%$) | PENDING | League generalization audit |
| **Milestone 6** | $N = 900$ | $0 / 900$ ($0.0\%$) | PENDING | Pre-power gate stability audit |
| **Milestone 7 (Gate)**| **$N = 1,050$** | **$0 / 1,050$ ($0.0\%$)** | **MANDATORY POWER GATE** | **Freeze cohort & run formal promotion analysis** |
| **Milestone 8** | $N = 1,301$ | $0 / 1,301$ ($0.0\%$) | PREFERRED FULL | Full cohort confirmation |

---

## 5. Security & Causality Verification

- **Pre-Kickoff Buffer Enforcement:** Active (rejects prediction if $t_{\text{pred}} > t_{\text{kickoff}} - 15\text{m}$).
- **SHA-256 Digest Calculation:** Deterministic canonical JSON hashing over all fixture features.
- **Two-Stage Outcome Separation:** `live_predictions` table is write-once and completely immutable; `match_outcomes` is populated post-match.
- **Fail-Closed Tamper Detection:** Hash mismatch raises `HashMismatchTamperError` and aborts evaluation.
- **Historical Fixture Rejection:** All 1,751 matches in 2025/2026 DB and previous validation cohorts are strictly excluded.
- **Zero Market Contamination:** Model inputs remain 100% market-odds free.

---

## 6. Test Suite & Integrity Status

- `tests/test_v46_operational_collection.py`: **4/4 PASS**
- `tests/test_v46_live_prospective_collection.py`: **6/6 PASS**
- `tests/test_v46_outcome_reconciliation.py`: **3/3 PASS**
- `tests/test_v4_6_physical_draw_gate.py`: **5/5 PASS**
- `tests/test_v4_5_causal_draw_meta.py`: **4/4 PASS**
- `tests/test_v4_4_robust_draw_override.py`: **5/5 PASS**
- `tests/test_v4_3_accuracy_preserving_draw_override.py`: **5/5 PASS**
- `tests/test_v4_2_prospective_integration.py`: **4/4 PASS**
- `tests/test_draw_resolution_candidate.py`: **6/6 PASS**
- `tests/test_prospective_validation_pipeline.py`: **7/7 PASS**
- `tests/test_prospective_operational_collection.py`: **1/1 PASS**
- `tests/test_fresh_100_prospective_pilot.py`: **1/1 PASS**
- **Total Modern Test Suite:** **60/60 PASS / 0 FAIL / 0 SKIP** (in 2.11s).
- **All 20 protected repository baseline assets:** **100% bit-identical**.
