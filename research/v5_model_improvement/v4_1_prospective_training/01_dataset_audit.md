# 01 — 2025/26 Dataset Audit & Historical Inventory

## 1. Overview & Data Scope

This audit evaluates the complete historical dataset through the end of the **2025/2026 season** utilized to train **v4_1_prospective_candidate**.

- **Source Databases**:
  - `data/processed/matches.db` (Table: `fixtures`)
  - `data/processed/features.db` (Table: `feature_rows`)
- **Total Historical Seasons**: 6 complete seasons (`2020/2021` through `2025/2026`)
- **Total Historical Fixtures**: 10,735 fixtures
- **Total Labelled Training Matches**: **10,734 matches**
- **Unlabelled / Quarantined**: Exactly 1 fixture (`fixture_id = 420450481`, Nantes vs Toulouse, ABANDONED with no final result)

---

## 2. Season-by-Season Breakdown

| Season | Total Matches | Labelled Matches | Start Date | End Date | Status Summary |
|:---|:---:|:---:|:---:|:---:|:---|
| **2020/2021** | 1,826 | 1,826 | 2020-08-21 | 2021-05-23 | 1,826 FT |
| **2021/2022** | 1,826 | 1,826 | 2021-08-06 | 2022-05-22 | 1,826 FT |
| **2022/2023** | 1,827 | 1,827 | 2022-08-05 | 2023-06-04 | 1,827 FT |
| **2023/2024** | 1,752 | 1,752 | 2023-08-11 | 2024-05-26 | 1,752 FT |
| **2024/2025** | 1,752 | 1,752 | 2024-08-16 | 2025-05-25 | 1,751 FT, 1 AWARDED |
| **2025/2026** | 1,752 | **1,751** | 2025-08-15 | 2026-05-24 | 1,751 FT, 1 ABANDONED |
| **Total (All 6 Seasons)** | **10,735** | **10,734** | **2020-08-21** | **2026-05-24** | **10,734 Valid Labelled Fixtures** |

---

## 3. 2025/2026 Competition Inventory

| Competition | Competition ID | Labelled Matches | Date Range | Goal Label Availability |
|:---|:---:|:---:|:---:|:---:|
| **Ligue 1** | 200 | 306 | 2025-08-15 to 2026-05-17 | 100% Complete (306/306) |
| **La Liga** | 419 | 380 | 2025-08-15 to 2026-05-24 | 100% Complete (380/380) |
| **Premier League** | 423 | 380 | 2025-08-15 to 2026-05-24 | 100% Complete (380/380) |
| **Bundesliga** | 477 | 306 | 2025-08-22 to 2026-05-16 | 100% Complete (306/306) |
| **Serie A** | 499 | 380 | 2025-08-23 to 2026-05-24 | 100% Complete (380/380) |
| **Total 2025/2026** | — | **1,751** | **2025-08-15 to 2026-05-24** | **1,751 / 1,752 (99.94%)** |

---

## 4. Data Quality & Integrity Verifications

1. **Target Completeness**:
   - `label_home_goals` and `label_away_goals` are strictly non-negative and finite across all 10,734 training matches.
   - Mean historical home goals: $1.5348$
   - Mean historical away goals: $1.2740$
2. **Missing Feature Values**:
   - Features missingness is strictly caused by early-season warmups (e.g. `last5` rolling features for newly promoted teams), which is deterministically median-imputed by the fitted `LogisticRegressionPreprocessor`.
3. **No Duplicate Matches**:
   - Every `fixture_id` is unique in both `fixtures` and `feature_rows`.
4. **Chronological Sorting**:
   - Matches are ordered strictly chronologically by `(unix, date, fixture_id)`.
