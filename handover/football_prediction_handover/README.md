# Football Prediction System — Independent Audit & Candidate Handover Package

## Project Overview
This self-contained package contains the complete prediction pipeline, feature generation, model artifacts, and evaluation suites for the **V4 Poisson Baseline** and the **Production Draw Champion Layer** across Europe's top 5 football leagues:
1. **Premier League** (England)
2. **La Liga** (Spain)
3. **Bundesliga** (Germany)
4. **Serie A** (Italy)
5. **Ligue 1** (France)

---

## Model Architecture & Provenance

```
                 V4 Poisson Model Baseline (Frozen Production Artifact)
                                           ↓
                 V4 Poisson Rates (lambda_home, lambda_away) & Baseline P(H), P(D), P(A)
                                           ↓
       +-----------------------------------+-----------------------------------+
       |        Dixon-Coles Layer          |        Elo Draw Curve Layer       |
       |  - Per-league rho correlation     |  - Pre-match |dElo| / 100         |
       |  - Low-score tau matrix correction|  - logit(P(D)_V4)                 |
       |  -> P(D)_DC                       |  -> P(D)_Elo                      |
       +-----------------------------------+-----------------------------------+
                                           ↓
       +-----------------------------------------------------------------------+
       |                      Logistic Stacking Integration                    |
       |  logit(P(D)_new) = 0.1130 + 0.6037 * logit(P_DC) + 0.4812 * logit(P_Elo)|
       +-----------------------------------------------------------------------+
                                           ↓
       +-----------------------------------------------------------------------+
       |                  Proportional Odds Simplex Redistribution             |
       |  P(H)_new = P(H)_V4 * (1 - P(D)_new) / (1 - P(D)_V4)                  |
       |  P(A)_new = P(A)_V4 * (1 - P(D)_new) / (1 - P(D)_V4)                  |
       +-----------------------------------------------------------------------+
                                           ↓
       Champion Probabilities [P(H), P(D), P(A)] + Original V4 Shadow Retained
```

---

## Directory Structure

```
football_prediction_handover/
├── README.md                                    # This guide
├── HANDOVER_MANIFEST.md                         # Detailed file manifest & inventory
├── SECURITY_AUDIT.md                            # Security and secret sanitization audit
├── requirements.txt                             # Python dependencies
├── config/
│   ├── competitions.json                        # League metadata & season ID mappings
│   └── .env.example                             # Sanitized config template
├── data/
│   ├── models/
│   │   └── v4_poisson_venue_elo_online_ad.pkl   # Frozen V4 model bundle
│   └── processed/
│       ├── matches.db                           # Canonical fixtures & match results database
│       └── features.db                          # Pre-match causal feature store
├── src/
│   ├── models/                                  # Model definition & prediction pipelines
│   │   ├── draw_champion.py                     # Production Draw Champion layer (v4.0)
│   │   ├── draw_champion_v41.py                 # Research candidate layer (v4.1)
│   │   ├── v4_artifact.py                       # V4 bundle deserialization & inference
│   │   └── poisson.py                           # Poisson probability calculation & grids
│   ├── features/                                # Online causal feature engineering
│   │   ├── elo.py                               # Chronological pre-match Elo rating calculation
│   │   └── online_attack_defense.py             # Causal online attack/defense state updates
│   └── monitoring/                              # Prospective pipeline & contracts
├── research/
│   └── v4_promotion/                            # Research specs, diagnostic & evaluation scripts
│       ├── draw_champion_method_frozen.json     # Frozen Champion methodology definition
│       ├── prospective_validation_protocol.json # Frozen validation protocol definition
│       ├── draw_champion_1301_evaluation.py     # 1,301-match retrospective diagnostic script
│       └── draw_champion_v41_candidate_analysis.py # Candidate sensitivity & calibration analysis
└── tests/                                       # Comprehensive test & regression suite
    ├── test_draw_champion_production.py         # Production Champion invariants (58 tests)
    ├── test_draw_champion_v41.py                # Candidate layer tests (13 tests)
    ├── test_draw_champion_1301_evaluation.py    # 1,301 evaluation test suite (20 tests)
    └── test_2025_26_dataset_inventory.py        # Dataset audit test suite (15 tests)
```

---

## Setup & Environment

### 1. Requirements
- Python `>= 3.10` (tested on Python 3.10 and 3.12)
- Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `requests`, `pytest`

```bash
pip install -r requirements.txt
```

### 2. Setting PYTHONPATH
To ensure modules resolve cleanly from the root of the handover package:

**PowerShell (Windows):**
```powershell
$env:PYTHONPATH="$PWD\src;$PWDesearch\dixon_coles"
```

**Bash / Linux / macOS:**
```bash
export PYTHONPATH="$PWD/src:$PWD/research/dixon_coles"
```

---

## How to Run Baseline & Draw Champion Inference

### 1. Run Baseline Unit & Champion Invariant Tests
```bash
python tests/test_draw_champion_production.py
```
*Expected Result: 58 / 58 PASS.*

### 2. Run Candidate Layer Invariant Tests
```bash
python tests/test_draw_champion_v41.py
```
*Expected Result: 13 / 13 PASS.*

### 3. Run Full 1,301-Match Retrospective Evaluation
```bash
python research/v4_promotion/draw_champion_1301_evaluation.py
```
*Evaluates V4 baseline vs Production Draw Champion across all 1,301 completed 2025/26 fixtures.*

### 4. Run Candidate Sensitivity Analysis (v4.1 Candidate)
```bash
python research/v4_promotion/draw_champion_v41_candidate_analysis.py
```
*Evaluates candidate stacking intercepts across 11 grid points with calibration curves and bootstrap validation.*

---

## Critical Governance & Safety Notice

> [!IMPORTANT]
> **Retrospective Diagnostic vs Live Prospective Validation:**  
> The 1,301 completed 2025/26 fixtures in this dataset are provided strictly for **retrospective diagnostic research and model auditing**. They do **NOT** count toward the live cryptographic pre-kickoff prospective confirmation requirement ($N \ge 1,050$), because predictions on these matches were evaluated post-hoc.

> [!WARNING]
> **Frozen Production Files:**  
> The production model `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`) and its configuration in `draw_champion_method_frozen.json` are **frozen reference standards**. Do not modify them directly. All candidate work should be developed in separate modules (such as `draw_champion_v41.py`).
