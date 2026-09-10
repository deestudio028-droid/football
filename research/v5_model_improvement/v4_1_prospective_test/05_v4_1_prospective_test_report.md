# 05 — V4.1 Prospective Test & Live Forecast Lock Report

## 1. Executive Summary

**Model Identity:** `v4_1_prospective_candidate`  
**Model Version:** `v4.1-champion-dc-elo-stacking-2025-26-trained`  
**Candidate Artifact Path:** `data/models/v4_1_prospective_candidate_2025_26.pkl`  
**Candidate MD5:** `145f918d933eb343c0f63ca342b10289`  
**Candidate SHA256:** `cbb00b32c3ce4dbf600564a95d11b67a0c236f6b34cd184a9ca8abcc8ab94ac8`  
**Frozen Production V4.0 Baseline:** `data/models/v4_poisson_venue_elo_online_ad.pkl` (`06841f0c03c8597b2b8cd8f8ab064864` — **100% BIT-IDENTICAL**)  
**Locked Prediction Count:** **99 Matches**  
**Prediction Ledger:** [`02_live_forecast_ledger.csv`](file:///e:/Football%20Prediction%20Project/research/v5_model_improvement/v4_1_prospective_test/02_live_forecast_ledger.csv)  

---

## 2. Prospective League Breakdown

| Competition / League | Locked Pre-Match Predictions |
|:---|:---:|
| **La Liga** | 24 |
| **Premier League** | 21 |
| **Serie A** | 19 |
| **Bundesliga** | 18 |
| **Ligue 1** | 17 |

| **Total Locked Predictions** | **99** |

---

## 3. Excluded Fixtures Summary

| Fixture ID | League | Match | Kickoff | Reason for Exclusion |
|:---:|:---|:---|:---:|:---|
| 420583814 | Ligue 1 | Rennes vs Paris Saint Germain | 2026-08-23T18:45:00.000000Z | Match already finished with status 'FT' |
| 420583812 | Serie A | Torino vs AC Milan | 2026-08-23T18:45:00.000000Z | Match already finished with status 'FT' |
| 420583813 | Serie A | Atalanta vs Sassuolo | 2026-08-23T18:45:00.000000Z | Match already finished with status 'FT' |
| 420583819 | La Liga | Elche vs FC Barcelona | 2026-08-23T19:30:00.000000Z | Match already finished with status 'FT' |


---

## 4. Information Clock & Pre-Match Integrity Certification

1. **Pre-Kickoff Timing**: All 99 predictions locked strictly prior to kickoff ($t_\text{pred} < t_\text{kickoff}$).
2. **Simplex Verification**: 100% of predictions satisfy $P(H) + P(D) + P(A) = 1.000000$.
3. **Zero Outcome Leakage**: Zero 2026 match outcomes were accessed.
4. **Duplicate Protection**: Exactly 1 official prediction per fixture.
5. **V4.0 and V4.1 Integrity**: Verified 100% bit-identical.
