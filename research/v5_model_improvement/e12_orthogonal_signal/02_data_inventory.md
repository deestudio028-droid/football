# 02 — Data Inventory & Pre-Match Feature Ingestion Matrix

## 1. Executive Summary

Experiment $E_{12}$ utilizes the versioned, immutable research dataset `data/research/e11_unified_xg_dataset.parquet` (10,735 official fixtures across 5 European competitions from 2020/21 through 2025/26).

---

## 2. Ingested Dataset Inventory

| Dataset Asset | Format | Row Count | Primary Features / Fields | Provenance & Source |
|:---|:---:|:---:|:---|:---|
| `e11_unified_xg_dataset.parquet` | Parquet | 10,735 | `xg_home`, `xg_away`, `npxg_home`, `npxg_away`, `stat_home_shots`, `stat_away_shots`, `home_goals`, `away_goals`, `unix`, `date` | `data/processed/matches.db` + OddAlerts Vendor ground truth + Calibrated Shot Model |
| `features.db` (Production) | SQLite | 10,735 | 91 Production Supervised Features | Frozen Production Asset |
| `matches.db` (Production) | SQLite | 10,735 | Fixture Schedules, Kickoff Times, Team IDs | Frozen Production Asset |
| `e12_orthogonal_features.parquet` | Parquet | 10,735 | 40 Orthogonal Feature Columns (Residuals, Trends, Venue, Matchup, Fatigue) | Generated strictly causally by `orthogonal_features.py` |

---

## 3. Data Integrity & Verification Checklist

- [x] Zero external commercial APIs introduced.
- [x] Zero duplicate data fetches.
- [x] 100% of rows contain valid pre-match timestamps (`unix`).
- [x] All 20 baseline protected assets remain 100% bit-identical.
