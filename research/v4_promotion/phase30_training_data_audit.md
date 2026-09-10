# Phase 30 — Training Data & Leakage Prevention Audit Report

**Execution Date:** 2026-08-22  
**Project:** Football Prediction Project  
**Authoritative Production Model:** `v4_draw_champion` (`v4.0-champion-dc-elo-stacking`) [100% FROZEN]  
**Research Experiment:** Historical-Only Draw Model Retraining & 2025/26 Blind Replay  
**Training Cutoff Boundary:** End of 2024/2025 Season (June 2025)  
**Audit Status:** **100% PASSED — ZERO LEAKAGE CONFIRMED**  

---

## 1. Dataset Split & Volume Verification

| Dataset Split | Season Window | Included Competitions | Matches ($N$) | Status |
|---|---|---|---:|:---:|
| **Historical Training Cohort** | 2020/21 – 2024/25 | Premier League, La Liga, Bundesliga, Serie A, Ligue 1 | **8,983** | **AUTHORIZED TRAINING SET** |
| **2025/2026 Test Window** | 2025/2026 Season | Premier League, La Liga, Bundesliga, Serie A, Ligue 1 | **1,751** | **STRICTLY EXCLUDED FROM TRAINING** |
| — *Diagnostic Cohort* | 2025/2026 Season | 5 Target Leagues | 1,301 | Blind Replay Evaluation Set |
| — *Independent Validation Cohort* | 2025/2026 Season | 5 Target Leagues | 450 | Independent Prospective Cohort |
| — *Live Prospective Store* | Post-Phase 25 | 5 Target Leagues | 0 | Live Shadow Store |
| **Total Ingested Database** | 2020/21 – 2025/26 | 5 Target Leagues | **10,734** | Matches Database Total |

---

## 2. Cryptographic Set Disjointness Proofs

```
1. Historical Training IDs ∩ 2025/2026 Test IDs:
   | {train_fids} ∩ {test_fids} | = 0 (EMPTY SET) -> PASS

2. Historical Training IDs ∩ Phase 25 Independent Validation Cohort (N=450):
   | {train_fids} ∩ {union_450_ids} | = 0 (EMPTY SET) -> PASS

3. Historical Training IDs ∩ Live Prospective Store (live_v46_prospective.sqlite):
   | {train_fids} ∩ {live_prospective_ids} | = 0 (EMPTY SET) -> PASS
```

- **Set Disjointness Result:** Zero overlap across all partitions. No fixture from the 2025/2026 season or prospective cohorts was accessible to the historical training process.

---

## 3. Causal Feature Engineering & Zero-Lookahead Audit

1. **Pre-Match Poisson Goal Models:**
   - Baseline home/away goal rates ($\mu_{\text{home}}, \mu_{\text{away}}$) were fitted strictly on pre-2025/26 completed training fixtures (`FINAL_TRAIN_SEASONS`).
2. **Online Attack/Defense (E6) States:**
   - Team attack and defense parameters were updated strictly sequentially along chronological kickoff timestamps.
   - For every match at timestamp $T$, features were extracted in Pass 1 *before* outcome updates were applied in Pass 2.
3. **Elo Ratings (E1):**
   - Elo ratings were calculated causally using pre-kickoff states only.
4. **Market Odds Isolation:**
   - Zero betting market odds, opening odds, closing odds, or market probabilities were included in the training feature matrix.

---

## 4. Chronological Walk-Forward Folds for Model Selection

Model selection was conducted purely within the 8,983 pre-2025/26 historical fixtures across three walk-forward folds:

| Fold Name | Training Window ($N_{\text{train}}$) | Validation Window ($N_{\text{val}}$) | Candidate Selection Role |
|---|---|---|---|
| **Fold 1** | 2020/21 + 2021/22 ($N = 3,594$) | 2022/2023 ($N = 1,797$) | Parameter discovery & initial grid search |
| **Fold 2** | 2020/21 – 2022/23 ($N = 5,391$) | 2023/2024 ($N = 1,796$) | Generalization & stability audit |
| **Fold 3** | 2020/21 – 2023/24 ($N = 7,187$) | 2024/2025 ($N = 1,796$) | Final candidate selection & parameter freezing |

**Formal Conclusion:** The model selection decision was completed and frozen before reading any 2025/2026 fixture data.
