# 03 — Data Inventory & Pre-Match Provenance Matrix

## 1. Executive Summary

Experiment $E_{13}$ uses the unified 10,735-match dataset from `data/processed/matches.db` and `data/processed/features.db` across the 5 target European competitions from 2020/21 through 2025/26.

---

## 2. Ingested Data Assets

| Asset Name | Format | Matches | Features | Role in Experiment |
|:---|:---:|:---:|:---:|:---|
| `matches.db` (fixtures table) | SQLite | 10,735 | 84 columns | Ground-truth fixtures, kickoff timestamps, goals, status |
| `features.db` | SQLite | 10,735 | 91 columns | $V_4$ production supervised baseline features |
| `v4_poisson_venue_elo_online_ad.pkl` | Pickle | 1 | 91 columns | Frozen baseline intensity predictor $(\lambda_H^{V_4}, \lambda_A^{V_4})$ |
| `dynamic_elo` (in-memory) | DataFrame | 10,735 | 11 columns | Causal adaptive-K ratings & attack/defense Elo |
| `dynamic_attack_defense` (in-memory) | DataFrame | 10,735 | 18 columns | Causal state-space latent states & posterior variances |

---

## 3. Data Integrity & Verification Checklist

- [x] Zero external data scraping required.
- [x] Zero duplicate data fetches.
- [x] All 20 baseline protected assets remain 100% bit-identical.
