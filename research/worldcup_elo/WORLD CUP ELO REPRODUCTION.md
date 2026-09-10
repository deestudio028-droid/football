# WORLD CUP ELO REPRODUCTION REPORT

**Date:** 2026-08-20  
**Repository:** [hjjbh1314/worldcup-predictor](https://github.com/hjjbh1314/worldcup-predictor)  
**Environment:** Python 3.13 on Windows (UTF-8 console)  
**Dataset:** `martj42/international_results` (49,520 international fixtures, 1872-11-30 to 2026-07-19)

---

## 1. Summary of Reproduction

All native scripts provided by the repository were executed directly on the downloaded dataset without modifying the repository source code.

| Script / Experiment | README Claimed | Actual Reproduced | Status |
|---|---|---|---|
| **test_sanity.py** | All tests pass | 4/4 tests PASS (causality, monotonicity, probability validity, metrics) | **PASS** |
| **run_backtest.py (Naive)** | Acc: 47.7%, LL: 1.0512, Brier: 0.6340, RPS: 0.2283 | Acc: 47.7%, LL: 1.0511, Brier: 0.6340, RPS: 0.2283 | **PASS** |
| **run_backtest.py (Elo)** | Acc: 60.0%, LL: 0.8735, Brier: 0.5137, RPS: 0.1707 | Acc: 60.1%, LL: 0.8730, Brier: 0.5134, RPS: 0.1705 | **PASS** |
| **run_ml_backtest.py (GB)** | Acc: 60.0%, LL: 0.8732, Brier: 0.5132, RPS: 0.1705 | Acc: 60.0%, LL: 0.8733, Brier: 0.5131, RPS: 0.1705 | **PASS** |
| **run_ml_backtest.py (Permutation Importance)** | elo_diff (+0.3418), ga_diff (+0.0097), others ≈ 0 | elo_diff (+0.3368), ga_diff (+0.0088), form_diff (+0.0056) | **PASS** |
| **run_confed_backtest.py (All)** | Acc: 60.2%, RPS: 0.1703 | Acc: 60.2%, LogLoss: 0.8721, RPS: 0.1701 | **PASS** |
| **run_confed_backtest.py (Cross-Confed)** | Acc: 58.3%, RPS: 0.1837 | Acc: 58.9%, LogLoss: 0.9142, RPS: 0.1803 | **PASS** |
| **run_calibration.py (Calibrated Elo)** | Acc: 60.40%, LL: 0.8701, Brier: 0.5114, RPS: 0.1696 | Acc: 60.40%, LL: 0.8701, Brier: 0.5114, RPS: 0.1696 | **PASS** |
| **run_dc_backtest.py (Calibrated Elo)** | Acc: 60.5%, RPS: 0.169 | Acc: 60.53%, LL: 0.8684, Brier: 0.5106, RPS: 0.1692 | **PASS** |
| **run_dc_backtest.py (Dixon-Coles)** | Acc: 58.4%, RPS: 0.177 | Acc: 58.45%, LL: 0.8949, Brier: 0.5270, RPS: 0.1774 | **PASS** |
| **tune_elo.py (Grid Search)** | Defaults optimal (HA=100, K=1.0, Regress=0.0) | Defaults optimal (HA=100, K=1.0, Regress=0.0, RPS 0.1783) | **PASS** |

---

## 2. Detailed Execution Log Excerpts

### 2.1 Sanity & Causality Tests (`tests/test_sanity.py`)
```
  PASS  test_elo_monotonic
  PASS  test_metrics_perfect_and_uniform
  PASS  test_no_leakage_causality
  PASS  test_proba_valid
全部通过
```

### 2.2 Elo vs Naive Baseline (`scripts/run_backtest.py`)
- Split: `2018-01-01` (Train: 41,300 matches / Test: 8,220 matches)
- Actual outcome distribution in test set: `{'H': 0.477, 'A': 0.292, 'D': 0.230}`
- Naive Baseline: Accuracy 47.7%, LogLoss 1.0511, Brier 0.6340, RPS 0.2283
- Elo Baseline: Accuracy 60.1%, LogLoss 0.8730, Brier 0.5134, RPS 0.1705 (Δ Accuracy: +12.4pp, Δ RPS: -0.0578)

### 2.3 Machine Learning Permutation Importance (`scripts/run_ml_backtest.py`)
- HistGradientBoosting multi-feature model: Accuracy 60.0%, LogLoss 0.8733, Brier 0.5131, RPS 0.1705
- Permutation importance (neg_log_loss):
  - `elo_diff`: `+0.3368 ± 0.0043`
  - `ga_diff`: `+0.0088 ± 0.0015`
  - `form_diff`: `+0.0056 ± 0.0011`
  - `elo_away`: `+0.0031 ± 0.0009`
  - `gf_diff`: `+0.0020 ± 0.0004`
  - `elo_home`: `+0.0017 ± 0.0005`
  - `tournament_k`: `+0.0015 ± 0.0005`
  - `rest_away`: `+0.0011 ± 0.0005`
  - `rest_home`: `+0.0003 ± 0.0005`
  - `rest_diff`: `+0.0003 ± 0.0004`
  - `neutral`: `+0.0002 ± 0.0002`
  - `congestion_diff`: `-0.0000 ± 0.0002`

### 2.4 Dixon-Coles vs Calibrated Elo (`scripts/run_dc_backtest.py`)
- Test subset: 7,741 matches (2018–2025)
- Calibrated Elo: Accuracy 60.53%, LogLoss 0.8684, Brier 0.5106, RPS 0.1692
- Dixon-Coles (rolling refit): Accuracy 58.45%, LogLoss 0.8949, Brier 0.5270, RPS 0.1774
- 50/50 Ensemble: Accuracy 60.12%, LogLoss 0.8689, Brier 0.5113, RPS 0.1701

### 2.5 Probability Calibration (`scripts/run_calibration.py`)
- Uncalibrated: Accuracy 60.12%, LogLoss 0.8724, Brier 0.5128, RPS 0.1702 (Mean P(H): 0.510 vs Actual 0.477)
- Calibrated (Platt): Accuracy 60.40%, LogLoss 0.8701, Brier 0.5114, RPS 0.1696 (Mean P(H): 0.477 vs Actual 0.477)