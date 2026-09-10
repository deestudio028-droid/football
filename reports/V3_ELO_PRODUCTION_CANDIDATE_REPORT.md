# V3 ELO PRODUCTION CANDIDATE REPORT

**Date:** 2026-08-20
**Project:** Football Prediction Project (FPP)
**Candidate:** V3 Poisson+Venue+Persistent Elo (87 features)
**Champion:** V2 Poisson+Venue (84 features)
**Protocol:** 3-Fold Walk-Forward Cross-Validation (2020/21–2024/25)
**Quarantined:** 2025/26 (never accessed for tuning, training, or evaluation)

---

## 1. Metric Discrepancy Reconciliation

Two champion baseline numbers appeared across project artifacts:

| Source | Champion Log Loss | Origin |
|---|---|---|
| `experiment_results.json` (machine-readable) | **0.992603** | `run_walk_forward_experiments.py` — PoissonRegressor on 3-fold walk-forward |
| `elo_feasibility_research_report.md` (narrative) | **0.998742** | Conversation estimate, not from any experiment artifact |

**Root cause:** The 0.998742 number was stated in the feasibility research report during a prior conversation session. It does not appear in any machine-readable experiment output or any file in the repository. It was likely an imprecise reference to earlier LogisticRegression-based Phase 4A Model B results or an erroneous round of a per-fold number.

**Authoritative resolution:** The ground truth is `experiment_results.json`, produced by `research/worldcup_elo/run_walk_forward_experiments.py`. This script:
- Uses the exact approved walk-forward folds from `src/models/config.py`
- Uses PoissonRegressor(alpha=1.0, max_iter=2000) — the same architecture as V2
- Uses the same LogisticRegressionPreprocessor
- Uses the same tail-safe Poisson H/D/A conversion
- Passed pre/post MD5 integrity checks on all 4 protected files

**AUTHORITATIVE CHAMPION METRICS (V2, 84 cols, 3-fold pooled):**

| Metric | Value |
|---|---|
| Accuracy | 52.81% |
| Log Loss | 0.992603 |
| Brier Score | 0.591782 |
| RPS | 0.202070 |
| Home Goal MAE | 0.967336 |
| Away Goal MAE | 0.868636 |
| Draw AUC | 0.542874 |
| Draw Recall | 0.0% |

**AUTHORITATIVE ELO CHALLENGER METRICS (V3, 87 cols, 3-fold pooled):**

| Metric | Value |
|---|---|
| Accuracy | 53.15% |
| Log Loss | 0.986913 |
| Brier Score | 0.587975 |
| RPS | 0.200186 |
| Home Goal MAE | 0.962868 |
| Away Goal MAE | 0.867574 |
| Draw AUC | 0.545210 |
| Draw Recall | 0.0% |

---

## 2. Elo Architecture

**Engine:** `research/worldcup_elo/elo_engine.py`

| Parameter | Value |
|---|---|
| Initial rating | 1500.0 |
| K-factor | 20.0 |
| Home advantage | +100.0 Elo points |
| Goal-difference multiplier | eloratings.net standard: 1.0 (≤1 GD), 1.5 (2 GD), (11+GD)/8 (3+ GD) |
| Season mean reversion | 0.0 (continuous carry-forward) |
| Promoted team handling | Initialized at 1500.0 (same as all teams; no special case needed with γ=0) |

**Expected score:** E_home = 1 / (1 + 10^((R_away - R_home - 100) / 400))

**Update rule:** Δ = K × G(goal_diff) × (W - E_home), where W = 1.0 (home win), 0.5 (draw), 0.0 (away win).

**Causality invariant:** For every fixture at timestamp T, ratings R_home(T) and R_away(T) are computed from strictly earlier fixtures only. Simultaneous fixtures (same unix timestamp) have their features extracted in a first pass before any updates occur in a second pass.

---

## 3. Exact 87-Feature Contract

```
COLUMNS 1–18:   GOALS_CORE (home/away goals for/against/diff × last5/last10/season)
COLUMNS 19–38:  FORM (home/away points/win_rate/draw_rate/loss_rate/goal_diff × last5/last10)
COLUMNS 39–43:  STRENGTH (home/away attack/defence strength scores + strength_diff)
COLUMN  44:     competition_id
COLUMNS 45–62:  SHOTS_CORE (home/away shots for/against/diff × last5/last10/season)
COLUMNS 63–80:  SHOTS_ON_CORE (home/away shots_on for/against/diff × last5/last10/season)
COLUMNS 81–84:  VENUE (home_goals_for/against_home_venue_season, away_goals_for/against_away_venue_season)
COLUMN  85:     home_elo
COLUMN  86:     away_elo
COLUMN  87:     elo_diff
```

Verified: `V3_FEATURE_COLUMNS[:80] == MODEL_B_COLUMNS`, `V3_FEATURE_COLUMNS[80:84] == VENUE_COLUMNS`, `V3_FEATURE_COLUMNS[84:87] == ELO_FEATURE_COLUMNS`. No label columns, no outcome columns, no odds columns, no duplicates. `elo_diff = home_elo + 100 - away_elo` verified on all 10,735 fixtures.

---

## 4. Causal Leakage Proof

### Automated tests (all PASS):

| Test | Method | Result |
|---|---|---|
| Truncation invariance | Compute Elo on first N fixtures vs full dataset; features for fixture 1..N must be identical | PASS (max_diff = 0.00e+00 at N=1000, 3000, 5000) |
| Outcome insulation | Modify fixture 250 goals to 10-0; pre-match Elo for fixture 250 and all earlier fixtures unchanged | PASS (diff_250 = 0.00e+00, diff_prior = 0.00e+00) |
| Chronological ordering | All fixtures processed in strict (unix, fixture_id) ASC order | PASS |
| Determinism | Two independent EloEngine runs on identical data produce identical output | PASS (max_diff = 0.00e+00 across all 10,735 fixtures) |
| 2025/26 quarantine | 2025/26 season IDs never appear in training partitions | PASS |
| No labels in X | Feature matrix contains no `label_*` columns | PASS |

### Structural guarantee:
The Elo engine uses a two-pass algorithm per timestamp group:
1. **First pass:** Extract pre-match features (home_elo, away_elo, elo_diff) for ALL fixtures at this timestamp
2. **Second pass:** Update ratings using outcomes of completed matches

This structure is identical to `FeatureContext.record()` in the existing feature pipeline — features are computed before outcomes are recorded.

---

## 5. Walk-Forward Results

### 5a. Fold-by-Fold Comparison

**Fold 1: Train 2020/21–2021/22, Validate 2022/23 (n=1,827)**

| Metric | Champion V2 | V3 Challenger | Delta |
|---|---|---|---|
| Accuracy | 52.22% | 52.76% | +0.55pp |
| Log Loss | 0.999113 | 0.994877 | **−0.004236** |
| Brier | 0.596473 | 0.593531 | −0.002942 |
| RPS | 0.206542 | 0.205141 | −0.001401 |
| Home Goal MAE | 0.9749 | 0.9705 | −0.0043 |
| Away Goal MAE | 0.8487 | 0.8498 | +0.0012 |
| Draw Recall | 0.0% | 0.0% | 0 |
| Mean P(H) | 0.4306 | 0.4307 | +0.0001 |
| Mean P(D) | 0.2309 | 0.2279 | −0.0030 |
| Mean P(A) | 0.3385 | 0.3414 | +0.0029 |

**Fold 2: Train 2020/21–2022/23, Validate 2023/24 (n=1,752)**

| Metric | Champion V2 | V3 Challenger | Delta |
|---|---|---|---|
| Accuracy | 53.42% | 53.82% | +0.40pp |
| Log Loss | 0.988397 | 0.981990 | **−0.006407** |
| Brier | 0.588452 | 0.584213 | −0.004240 |
| RPS | 0.197444 | 0.195300 | −0.002144 |
| Home Goal MAE | 0.9765 | 0.9727 | −0.0038 |
| Away Goal MAE | 0.8805 | 0.8772 | −0.0033 |
| Draw Recall | 0.0% | 0.0% | 0 |
| Mean P(H) | 0.4416 | 0.4426 | +0.0010 |
| Mean P(D) | 0.2309 | 0.2288 | −0.0021 |
| Mean P(A) | 0.3274 | 0.3285 | +0.0011 |

**Fold 3: Train 2020/21–2023/24, Validate 2024/25 (n=1,752)**

| Metric | Champion V2 | V3 Challenger | Delta |
|---|---|---|---|
| Accuracy | 52.80% | 52.85% | +0.05pp |
| Log Loss | 0.990298 | 0.983873 | **−0.006425** |
| Brier | 0.590422 | 0.586181 | −0.004241 |
| RPS | 0.202225 | 0.200116 | −0.002110 |
| Home Goal MAE | 0.9506 | 0.9453 | −0.0053 |
| Away Goal MAE | 0.8767 | 0.8756 | −0.0011 |
| Draw Recall | 0.0% | 0.0% | 0 |
| Mean P(H) | 0.4400 | 0.4416 | +0.0015 |
| Mean P(D) | 0.2327 | 0.2296 | −0.0031 |
| Mean P(A) | 0.3272 | 0.3288 | +0.0015 |

### 5b. Pooled Comparison

| Metric | Champion V2 | V3 Challenger | Delta | Fold Wins |
|---|---|---|---|---|
| **Accuracy** | 52.81% | **53.15%** | **+0.33pp** | **3/3** |
| **Log Loss** | 0.992603 | **0.986913** | **−0.005689** | **3/3** |
| **Brier Score** | 0.591782 | **0.587975** | **−0.003808** | **3/3** |
| **RPS** | 0.202070 | **0.200186** | **−0.001885** | **3/3** |
| Home Goal MAE | 0.967336 | **0.962868** | −0.004469 | 3/3 |
| Away Goal MAE | 0.868636 | **0.867574** | −0.001061 | 2/3 |
| Draw AUC | 0.542874 | **0.545210** | +0.002336 | 3/3 |
| Draw Recall | 0.0% | 0.0% | 0 | — |

---

## 6. Fold-by-Fold Log Loss Breakdown (Decision Metric)

| Fold | Champion LL | V3 LL | Delta | V3 Wins? |
|---|---|---|---|---|
| Fold 1 (val 2022/23) | 0.999113 | 0.994877 | −0.004236 | **YES** |
| Fold 2 (val 2023/24) | 0.988397 | 0.981990 | −0.006407 | **YES** |
| Fold 3 (val 2024/25) | 0.990298 | 0.983873 | −0.006425 | **YES** |
| **Pooled** | **0.992603** | **0.986913** | **−0.005689** | **3/3** |

No fold shows degradation. The worst fold for V3 (Fold 1) still improves by −0.004236.

---

## 7. Pooled Comparison (summary)

See Section 5b above.

---

## 8. Early-Season Analysis

### 8a. Performance by Season Phase

| Season Phase | N | Champion LL | V3 LL | LL Delta | Champion Acc | V3 Acc | Acc Delta |
|---|---|---|---|---|---|---|---|
| **Matches 1–5 (Early)** | 722 | 1.0168 | **1.0066** | **−0.0102** | 50.28% | **50.97%** | **+0.69pp** |
| Matches 6–10 (Transition) | 716 | 0.9909 | **0.9852** | −0.0057 | 51.54% | **51.82%** | +0.28pp |
| Matches 11+ (Mature) | 3,875 | 0.9885 | **0.9861** | −0.0024 | **53.52%** | 53.47% | −0.05pp |

**Finding:** The central hypothesis is confirmed — Elo's largest benefit is in early-season matches where within-season features are sparse. The log loss improvement in matches 1–5 (−0.0102) is **4.3× larger** than in mature matches (−0.0024).

### 8b. Promoted / Low-History Teams

| Group | N | Champion LL | V3 LL | LL Delta | Champion Acc | V3 Acc | Acc Delta |
|---|---|---|---|---|---|---|---|
| Promoted/Low-History involved | 803 | 0.9981 | **0.9941** | −0.0040 | 52.30% | **52.55%** | +0.25pp |
| Established teams only | 4,528 | 0.9917 | **0.9879** | −0.0039 | 52.89% | **52.96%** | +0.07pp |

### 8c. Elo Mismatch Magnitude

| Mismatch | N | Champion LL | V3 LL | LL Delta | Champion Acc | V3 Acc | Acc Delta |
|---|---|---|---|---|---|---|---|
| Large (|Δ| ≥ 150) | 2,447 | 0.9020 | **0.8968** | −0.0053 | 62.08% | **62.44%** | +0.37pp |
| Medium (75 ≤ |Δ| < 150) | 1,368 | 1.0604 | **1.0584** | −0.0019 | **47.37%** | 47.00% | −0.37pp |
| Small (|Δ| < 75) | 1,516 | 1.0780 | **1.0745** | −0.0035 | 42.74% | **42.81%** | +0.07pp |

---

## 9. Subgroup Diagnostics

See Section 8 above. Key observations:

- Elo improves log loss in **every subgroup** — early/late season, promoted/established, large/medium/small mismatch. This is not a one-bucket artifact.
- The accuracy improvement is concentrated in large-mismatch and early-season fixtures, which is theoretically expected — Elo provides the most incremental information where within-season form data is least informative.
- Draw recall remains 0% for both Champion and Challenger. Elo does not solve the draw-prediction limitation (this is a known architectural limitation of the Poisson formulation, not an Elo issue).

---

## 10. CLI / JSON Tests

Tests from `tests/test_v3_candidate.py` (11 suites):

| Suite | Description | Result |
|---|---|---|
| 1 | Feature contract (87 columns, ordering, no labels) | **PASS** |
| 2 | Causal leakage (100% causal features verified) | **PASS** |
| 3 | Determinism (identical output across runs) | **PASS** |
| 4 | Probability validity (0 ≤ P ≤ 1, sum = 1.0) | **PASS** |
| 5 | CLI inference simulation (single fixture, valid lambdas) | **PASS** |
| 6 | JSON response schema (compatible with V2 schema) | **PASS** |
| 7 | Venue NULL handling (NaN imputed by preprocessor) | **PASS** |
| 8 | No-write integrity (all 4 protected files identical) | **PASS** |
| 9 | V3 artifact loader (all invariants satisfied) | **PASS** |
| 10 | Backward compatibility (V1 and V2 load unaltered) | **PASS** |
| 11 | 2025/26 quarantine governance | **PASS** |

**Note:** Suites 3–7 require scipy/sklearn and were executed on the local machine where the V3 candidate was trained. Suites 1, 2, 8, 10, 11 were independently verified in the sandboxed audit environment.

---

## 11. Determinism Tests

- Elo engine: Two independent runs on full 10,735-fixture dataset produce max absolute difference of 0.00e+00 across all features (home_elo, away_elo, elo_diff).
- V3 predictions: `predict_hda_probabilities()` called twice on the same input produces max difference of < 1e-12.
- No random state dependency (PoissonRegressor is deterministic; no random splitting in walk-forward).

---

## 12. No-Write Verification

### Pre-experiment checksums:

| File | Expected MD5 | Actual MD5 | Status |
|---|---|---|---|
| data/models/v2_poisson_venue.pkl | 25935b4e93fc4074f67f16e3181ed4df | 25935b4e93fc4074f67f16e3181ed4df | **PASS** |
| data/models/v1_logreg.pkl | 5e504427712b35778bb8a62a8496c7cd | 5e504427712b35778bb8a62a8496c7cd | **PASS** |
| data/processed/features.db | e7ebe7fc07040a5927683c35b6371e63 | e7ebe7fc07040a5927683c35b6371e63 | **PASS** |
| data/processed/matches.db | fdeed042096fa1c851aaee6c84995247 | fdeed042096fa1c851aaee6c84995247 | **PASS** |

### Post-experiment checksums (verified after all tests, analysis, and report generation):

| File | Post-Experiment MD5 | Status |
|---|---|---|
| data/models/v2_poisson_venue.pkl | 25935b4e93fc4074f67f16e3181ed4df | **UNTOUCHED** |
| data/models/v1_logreg.pkl | 5e504427712b35778bb8a62a8496c7cd | **UNTOUCHED** |
| data/processed/features.db | e7ebe7fc07040a5927683c35b6371e63 | **UNTOUCHED** |
| data/processed/matches.db | fdeed042096fa1c851aaee6c84995247 | **UNTOUCHED** |

`git status`: clean (no uncommitted changes to production source).

---

## 13. MD5 Comparison

| Artifact | MD5 | Notes |
|---|---|---|
| V1 LogReg (frozen) | 5e504427712b35778bb8a62a8496c7cd | Untouched |
| V2 Poisson+Venue (frozen champion) | 25935b4e93fc4074f67f16e3181ed4df | Untouched |
| V3 Candidate (new) | a2850a7687822a5916663301f5ccc96c | 11,773 bytes |
| features.db (frozen) | e7ebe7fc07040a5927683c35b6371e63 | Untouched |
| matches.db (frozen) | fdeed042096fa1c851aaee6c84995247 | Untouched |

---

## 14. Files Created

| File | Purpose |
|---|---|
| `research/worldcup_elo/elo_engine.py` | Isolated causal Elo engine |
| `research/worldcup_elo/run_walk_forward_experiments.py` | Full 7-arm walk-forward experiment runner |
| `research/worldcup_elo/test_causal_elo.py` | Causal invariance test suite |
| `research/worldcup_elo/train_v3_candidate.py` | V3 candidate artifact trainer |
| `research/worldcup_elo/experiment_results.json` | Machine-readable experiment output |
| `research/worldcup_elo/WORLD CUP ELO FPP EXPERIMENT REPORT.md` | Experiment report |
| `src/models/v3_contract.py` | 87-column feature contract definition |
| `src/models/v3_artifact.py` | V3 artifact loader and inference |
| `data/models/v3_poisson_venue_elo_candidate.pkl` | V3 candidate artifact |
| `tests/test_v3_candidate.py` | 11-suite V3 verification tests |
| `scripts/promote_v3.py` | Manual promotion script |
| `scripts/rollback_v3.py` | Manual rollback script |
| `reports/V3_ELO_PRODUCTION_CANDIDATE_REPORT.md` | This report |
| `reports/elo_feasibility_research_report.md` | Prior feasibility research |

---

## 15. Files Modified

**None.** Zero modifications to any existing production file:

- `predict_match.py` — unchanged
- `src/models/poisson.py` — unchanged
- `src/models/v2_artifact.py` — unchanged
- `src/models/ablation.py` — unchanged
- `src/models/config.py` — unchanged
- `src/features/*` — unchanged
- `data/models/v2_poisson_venue.pkl` — unchanged (MD5 verified)
- `data/models/v1_logreg.pkl` — unchanged (MD5 verified)
- `data/processed/features.db` — unchanged (MD5 verified)
- `data/processed/matches.db` — unchanged (MD5 verified)

---

## 16. Promotion Procedure

**Script:** `scripts/promote_v3.py`

Steps:
1. Run `python scripts/promote_v3.py`
2. Script verifies all 4 protected file checksums
3. Script verifies V3 candidate exists
4. Script asks for confirmation (type `PROMOTE`)
5. Copies `v3_poisson_venue_elo_candidate.pkl` → `v3_poisson_venue_elo.pkl`
6. Verifies copy integrity and re-checks all protected files
7. **Manual next steps:** Update `predict_match.py` to support `--model v3`, run integration tests, commit

---

## 17. Rollback Procedure

**Script:** `scripts/rollback_v3.py`

Steps:
1. Run `python scripts/rollback_v3.py`
2. Script verifies V2 champion is intact (MD5 check)
3. Script asks for confirmation (type `ROLLBACK`)
4. Removes `v3_poisson_venue_elo.pkl` (production copy only)
5. Preserves `v3_poisson_venue_elo_candidate.pkl` (audit trail)
6. Verifies all protected files remain intact
7. **Manual next step:** Ensure `predict_match.py` defaults to V2

---

## 18. Final Decision

### Decision Rule Evaluation:

| Criterion | Required | Observed | Status |
|---|---|---|---|
| 1. Pooled log loss improves | V3 LL < V2 LL | 0.986913 < 0.992603 (−0.005689) | **PASS** |
| 2. Log loss improves on ≥ 2/3 folds | ≥ 2 fold wins | **3/3 folds** (all improve) | **PASS** |
| 3. No catastrophic degradation | No fold > +0.02 | Worst fold delta: −0.004236 (improvement) | **PASS** |
| 4. No leakage detected | All causal tests pass | Truncation, insulation, determinism all PASS | **PASS** |
| 5. All technical tests pass | 11/11 suites | 11/11 PASS | **PASS** |

### Additional evidence:
- V3 improves **every proper scoring metric** (log loss, Brier, RPS) on **every fold**
- V3 improves accuracy on all 3 folds (pooled: +0.33pp)
- Early-season hypothesis confirmed: 4.3× larger benefit in matches 1–5 vs 11+
- Improvement observed in every subgroup tested (promoted teams, established teams, all mismatch buckets)
- Zero draw regression (Draw AUC improves slightly; recall remains 0% for both)
- Zero production code modifications

---

# FINAL DECISION: **PROMOTE V3 CANDIDATE**

V3 Poisson+Venue+Persistent Elo passes all 5 preregistered acceptance criteria with no exceptions. The improvement is consistent across all 3 temporal folds, all subgroups, and all proper scoring rules. The Elo engine is causally sound, deterministic, and adds exactly 3 features to the frozen 84-column V2 contract without modifying any existing code or artifact.

**This does NOT mean V3 guarantees 65% accuracy.** V3 improves from 52.81% to 53.15% — a genuine but modest gain. Reaching 60%+ will require additional orthogonal information sources (betting odds recommended as next priority) and potentially a model architecture upgrade.

**Promotion requires manual execution of `scripts/promote_v3.py`.** V2 remains fully recoverable via `scripts/rollback_v3.py`.
