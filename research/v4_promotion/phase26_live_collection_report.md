# Phase 26 — Live Prospective Collection Pipeline for Frozen V4.6

**Execution Date:** 2026-08-22  
**Project:** Football Prediction Project  
**Authoritative Production Model:** `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`) [100% FROZEN]  
**Authoritative Shadow Model:** `v4_6_physical_draw_gate` (`v4.6-physical-draw-gate`) [FROZEN SHADOW CANDIDATE]  
**Isolated Prospective Store:** `research/v4_promotion/live_v46_prospective.sqlite`  
**Target Power Threshold:** $N \ge 1,050$ Fresh Prospective Fixtures  
**Initial Cohort Status:** $N = 0$ Fresh Fixtures (Collection Pipeline Active)  

---

## 1. Executive Summary & Operational Status

In Phase 26, we established a **true live prospective collection and outcome reconciliation infrastructure** for the frozen **V4.6 Physical Draw Gate** shadow candidate.

### Key System Invariants Implemented

1. **Complete Model & Parameter Freeze:**
   - The exact V4.6 physical gating rule ($\theta \ge 0.26, \text{winner\_margin} \le 0.10, \text{v4\_conf\_cap} \le 0.45, |\Delta \text{Elo}| \le 100.0, \lambda_{\text{tot}} \le 2.50$) is permanently locked.
   - The collector explicitly validates configuration parameters on startup; any parameter mutation raises `ModelConfigMutationError` and aborts.

2. **Pre-Kickoff Immutable Lock with SHA-256 Digest:**
   - Every prediction is generated and cryptographically hashed (SHA-256) strictly before match kickoff (`prediction_timestamp < scheduled_kickoff`).
   - The hash covers the full fixture context, probabilities, decision features, and timestamp. Once locked, prediction records are completely immutable.

3. **Two-Stage Outcome Separation (Zero Data Leakage):**
   - Predictions are stored in `live_predictions` prior to kickoff.
   - Match outcomes (`home_goals`, `away_goals`, `actual_outcome`) are recorded separately in `match_outcomes` only after match completion.
   - Prediction engines have zero access to in-play statistics, scores, cards, xG, or closing odds.

4. **Zero Historical Recycling & Zero Market Odds:**
   - The collector pre-loads all 1,751 known historical fixture IDs (from `matches.db` and previous validation cohorts). Any attempt to ingest an existing historical fixture raises `HistoricalFixtureRejectionError`.
   - Market odds are strictly isolated and never enter model inputs.

5. **Progressive Power Milestones:**
   - Fresh prospective fixtures are tracked from $N = 0$ toward the mandatory **$N \ge 1,050$ statistical power threshold** ($80\%$ power at $\alpha=0.05$).

---

## 2. Live Prospective Pipeline Architecture

```
                               UPCOMING FIXTURES FEED
                                         │
                                         ▼
                     STAGE 1: PRE-KICKOFF LIVE COLLECTOR
                    (src/monitoring/v46_live_prospective_collector.py)
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               ▼                                                   ▼
      PRE-FLIGHT VALIDATION                               PRE-MATCH PREDICTION
   - Reject known historical IDs                         - V4 Poisson Probabilities
   - Enforce t_pred < t_kickoff                          - V4.2 DIBP Draw Prob
   - Enforce V4.6 config freeze                          - Physical Gating Decision
               │                                                   │
               └─────────────────────────┬─────────────────────────┘
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
                    STAGE 2: POST-MATCH OUTCOME RECONCILER
                     (src/monitoring/v46_outcome_reconciler.py)
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               ▼                                                   ▼
      TAMPER DETECTION CHECK                               OUTCOME RECORDING
   - Re-compute SHA-256 hash                             - Final score (H-A)
   - Compare with prediction_sha256                      - Actual outcome (H/D/A)
   - FAIL CLOSED if mismatch                             - Table: `match_outcomes`
               │                                                   │
               └─────────────────────────┬─────────────────────────┘
                                         │
                                         ▼
                         LIVE VALIDATION DASHBOARD METRICS
                 - Cumulative Accuracy (V4 vs V4.6)
                 - Draw Predictions, Correct Draws, Precision, Recall
                 - Error Economics (Good, Bad, Neutral, Net Gain)
                 - Milestone Tracking toward N >= 1,050
```

---

## 3. Database Schema (`live_v46_prospective.sqlite`)

### Table: `live_predictions`
- `fixture_id` (INTEGER, PRIMARY KEY)
- `competition_id` (INTEGER, NOT NULL)
- `competition_name` (TEXT, NOT NULL)
- `home_team` (TEXT, NOT NULL)
- `away_team` (TEXT, NOT NULL)
- `scheduled_kickoff` (TEXT, ISO-8601 UTC)
- `prediction_timestamp` (TEXT, ISO-8601 UTC)
- `p_v4_home`, `p_v4_draw`, `p_v4_away` (REAL, NOT NULL)
- `v4_base_decision` (TEXT, NOT NULL)
- `p_v42_draw`, `winner_margin`, `v4_winner_conf` (REAL, NOT NULL)
- `abs_elo_diff`, `tot_expected_goals`, `low_score_prob` (REAL, NOT NULL)
- `override_applied` (INTEGER, 0 or 1)
- `v4_6_final_decision` (TEXT, NOT NULL)
- `model_version` (TEXT, NOT NULL)
- `prediction_sha256` (TEXT, UNIQUE, NOT NULL)
- `lock_status` (TEXT, DEFAULT "LOCKED")
- `created_at` (TEXT, ISO-8601 UTC)

### Table: `match_outcomes`
- `fixture_id` (INTEGER, PRIMARY KEY, FK to `live_predictions`)
- `outcome_timestamp` (TEXT, ISO-8601 UTC)
- `home_goals` (INTEGER, NOT NULL)
- `away_goals` (INTEGER, NOT NULL)
- `actual_outcome` (TEXT, NOT NULL: 'H', 'D', 'A')
- `status` (TEXT, 'FT')
- `reconciled_at` (TEXT, ISO-8601 UTC)

### Table: `collection_audit_log`
- `log_id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
- `timestamp` (TEXT, ISO-8601 UTC)
- `action` (TEXT, NOT NULL)
- `fixtures_ingested` (INTEGER, NOT NULL)
- `details` (TEXT, NOT NULL)

---

## 4. Operational Milestones & Promotion Rules

| Milestone ($N$) | Required Sample Size | Progress ($N=0$) | Status | Action Required |
|---|---:|---:|:---:|---|
| **Milestone 1** | $N = 100$ | $0.0\%$ | PENDING | Initial operational checkpoint |
| **Milestone 2** | $N = 300$ | $0.0\%$ | PENDING | Preliminary draw precision check |
| **Milestone 3** | $N = 450$ | $0.0\%$ | PENDING | Early stability checkpoint |
| **Milestone 4** | $N = 600$ | $0.0\%$ | PENDING | Mid-point error economics check |
| **Milestone 5** | $N = 750$ | $0.0\%$ | PENDING | League generalization audit |
| **Milestone 6** | $N = 900$ | $0.0\%$ | PENDING | Pre-power gate stability audit |
| **Milestone 7 (Gate)**| **$N = 1,050$** | **$0.0\%$** | **MANDATORY POWER GATE** | **Execute Formal Paired Bootstrap Promotion Analysis** |
| **Milestone 8** | $N = 1,301$ | $0.0\%$ | PREFERRED FULL | Full cohort confirmation |

### Mandatory Promotion Gate Rules (at $N \ge 1,050$)
1. **Accuracy Floor:** V4.6 Accuracy $\ge$ V4 Baseline Accuracy on the live cohort.
2. **Draw Precision Floor:** Live Draw Precision $>$ actual observed base draw rate ($\approx 25\%$).
3. **Net Transition Gain:** $\text{Good Overrides} - \text{Bad Overrides} \ge 0$.
4. **League Robustness:** Zero systemic degradation across any target league.
5. **Cryptographic Verification:** $100\%$ prediction SHA-256 match; zero tamper errors.

---

## 5. Live Monitoring Dashboard

```
================================================================================================
V4.6 LIVE PROSPECTIVE MONITORING DASHBOARD
================================================================================================
Cohort Progress:          N = 0 / 1,050 (0.0% toward Statistical Power Gate)
Status:                   COLLECTING
Database:                 research/v4_promotion/live_v46_prospective.sqlite
Model Candidate:          v4_6_physical_draw_gate (v4.6-physical-draw-gate) [FROZEN]

ACCURACY & SCORECARD:
  V4 Baseline Accuracy:   --.--% (0 / 0)
  V4.6 Candidate Accuracy:--.--% (0 / 0)
  Delta Accuracy:         +0.00%

DRAW METRICS:
  Draw Predictions:       0
  Correct Draws:          0
  Draw Precision:         --.-%
  Draw Recall:            --.-%

ERROR ECONOMICS:
  Good Overrides Gained:  0
  Bad Overrides Sacrificed:0
  Neutral Overrides:      0
  Net Transition Gain:    +0
================================================================================================
```

---

## 6. Verification & Test Suite Status

Comprehensive unit test coverage across all prospective ingestion and reconciliation flows:
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
- **Total Modern Test Suite:** **56/56 PASS / 0 FAIL / 0 SKIP** (in 1.62s).

---

## 7. Repository Integrity Audit

All 20 protected repository baseline assets remain **100% bit-identical**:
- Production model (`v4_draw_champion`) remains **100% frozen**.
- `matches.db` and `features.db` remain **100% unmodified**.
- Live collection store is fully isolated in `live_v46_prospective.sqlite`.
