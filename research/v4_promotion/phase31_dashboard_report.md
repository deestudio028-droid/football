# Phase 31 — Model Evaluation & 2026/27 Fresh Match Validation Dashboard Report

**Execution Date:** 2026-08-22  
**Project:** Football Prediction Project  
**Authoritative Production Model:** `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`) [100% FROZEN]  
**Primary Shadow Candidate:** `v4_6_physical_draw_gate` (`v4.6-physical-draw-gate`) [100% FROZEN]  
**Research Retrained Candidate:** `historical_candidate_h` (`v4.6-historical-only-retrained`) [100% FROZEN]  
**Dashboard Technology:** Local Streamlit Application (`Streamlit 1.62.0`)  
**Status:** **OPERATIONAL & VERIFIED**  

---

## 1. Executive Summary & Dashboard Architecture

Phase 31 successfully establishes a local **Football Prediction Model Lab** research dashboard for evaluating production models and shadow candidates against new 2026/27 matches, exploring pre-match inferences, and monitoring draw resolution performance under strict pre-kickoff causality.

```
src/dashboard/
├── __init__.py               # Package exports
├── app.py                    # Streamlit interface with 9 interactive research tabs
├── model_registry.py         # Read-only registry with SHA-256 and MD5 integrity verification
├── fixture_service.py        # Database fixture discovery across the 5 target leagues
├── prediction_service.py     # Unified multi-model inference & 5-gate check engine
├── evaluation_service.py     # Two-stage blind out-of-sample replay & promotion report generator
└── metrics_service.py        # Scorecards, Error Economics, Model Agreement, & Bootstrap CIs
```

---

## 2. Model Registry & Immutability Verification

All models are loaded in strictly **read-only mode**, preventing any in-memory or on-disk mutation:

| Registered Model | Version | Role | State | MD5 Hash | Source Path |
|---|---|---|:---:|:---:|---|
| **`v4_baseline`** | `v4.0-baseline` | Production Core | **FROZEN** | `06841f0c03c8597b2b8cd8f8ab064864` | `data/models/v4_poisson_venue_elo_online_ad.pkl` |
| **`v4_0_draw_champion`** | `v4.0-champion-dc-elo-stacking` | Production Authoritative | **FROZEN** | `9c396e7e5364f93f079313726c1ba499` | `research/v4_promotion/draw_champion_method_frozen.json` |
| **`v4_2_candidate_d`** | `v4.2-dibp-calibrated-stacking` | Research | Experimental | `c7f76ca09633e9ec2d26d70ebf30e7ca` | `src/models/v4_2_draw_resolution_candidate.py` |
| **`v4_6_physical_draw_gate`** | `v4.6-physical-draw-gate` | Primary Shadow | **FROZEN** | `a181165cbdfd908eb8ff7bfa7c5ea2d8` | `src/models/v4_6_physical_draw_gate.py` |
| **`historical_candidate_h`** | `v4.6-historical-only-retrained` | Historical Retrained | **FROZEN** | `19aa786c57f2081f2fcb6c006e87a205` | `src/models/historical_draw_research_candidate.py` |

---

## 3. Dashboard Functional Capabilities (9 Dedicated Tabs)

1. **🏠 Overview:**
   - Real-time display of authoritative production model, shadow candidate, and performance summaries.
   - High-level architecture pipeline showing Poisson expected goals $\to$ DIBP draw density $\to$ Physical Draw Gate.
2. **🔮 Single Match Prediction:**
   - Interactive dropdown selection across the 5 target leagues: Premier League (423), La Liga (419), Bundesliga (477), Serie A (499), Ligue 1 (200).
   - Dynamic team roster population preventing Home Team == Away Team.
   - Generates simultaneous inferences: V4 Baseline, V4.0 Champion, V4.2, V4.6, and Historical Candidate H.
   - Detailed **5-Gate Condition Breakdown**:
     - Gate 1: $P(\text{Draw}) \ge 0.2600$
     - Gate 2: $\text{Winner Margin} \le 0.1000$
     - Gate 3: $\text{V4 Max Confidence} \le 0.4500$
     - Gate 4: $|\Delta \text{Elo}| \le 100.0$
     - Gate 5: $\lambda_{\text{total}} \le 2.5000$
3. **📅 Today's Matches:**
   - Live calendar date query (2026-08-22) reporting scheduled fixtures, pre-kickoff predictions, and status.
4. **📊 2026/27 Evaluation:**
   - Automated discovery of 2026/27 matches in the database.
   - Sample power status tracking (`EARLY / INSUFFICIENT` for $N < 100 \dots$ `POWER GATE` for $N \ge 1,050$).
   - Two-stage evaluation table joining final outcomes only after predictions are persisted.
5. **⚽ Draw Analysis:**
   - Comparative draw metrics (Precision, Recall, F1, Base Draw Rate vs Model Prediction Rate).
   - Full **Draw Error Economics** scorecard (Good Wins Gained, Bad Wins Sacrificed, Net Transition Gain).
6. **🏆 League Breakdown:**
   - Granular 5-league performance tables with automated **GREEN / YELLOW / RED** status flags.
7. **📈 Temporal Analysis:**
   - Early, Middle, and Late chronological robustness slices.
8. **🧠 Model Agreement:**
   - Pairwise agreement and divergence analysis between V4, V4.6, and Historical Candidate H (showing 100% policy equivalence between V4.6 and Candidate H).
9. **🔐 Production Safety & Promotion Review:**
   - Protected baseline hash integrity monitoring.
   - Interactive `[GENERATE PROMOTION REVIEW]` button outputting a formal audit and review markdown artifact.

---

## 4. Launching the Local Dashboard

To launch the local dashboard on Windows:

```cmd
scripts\run_dashboard.bat
```

Or directly via CLI:
```bash
streamlit run src/dashboard/app.py
```

---

## 5. Verification & Test Suite

The test suite across all 19 test modules was executed:
- `tests/test_phase31_dashboard.py`: **6/6 PASS**
- **Complete Test Suite Total:** **76/76 PASS / 0 FAIL / 0 SKIP** (in 15.68s).
- **All 20 protected repository baseline assets:** **100% bit-identical**.
