# Phase 2: Full Post-Ingestion Data Integrity & Quality Audit

Read-only audit of the completed full historical ingestion (5 leagues × 6 seasons, 30 windows). No API calls were made. No data was modified or repaired. Every finding is classified GREEN (verified/safe), YELLOW (anomaly/limitation to be aware of), or RED (integrity problem/blocker). Machine-readable evidence backing every section is in `data/audit/phase2_*.json`.

---

## A. Dataset summary

| | |
|---|---|
| Total fixture rows in SQLite | **10,735** |
| Total raw fixture records (across 60 pages) | **10,735** |
| Windows (competition × season) | **30 / 30 complete** |
| Leagues | Premier League, La Liga, Bundesliga, Serie A, Ligue 1 |
| Seasons | 2020/21 – 2025/26 |
| Test suite | **45/45 passing** |

Note on the console figure from the ingestion run ("Fixtures stored: 10355"): that number reflects only fixtures **newly written during that specific run**. The Premier League 2024/25 window (380 fixtures) was correctly skipped because it was already `complete` from the earlier smoke test. 10,355 + 380 = 10,735, which is exactly what's in the database. This is correct resumable-pipeline behavior, not a discrepancy — classified **YELLOW** only because it's easy to misread the console total as "the whole dataset" if you don't know a window was pre-seeded.

---

## B. Integrity results

| Check | Result | Class |
|---|---|---|
| SQLite row count | 10,735 (not 10,355 — see note above; reconciled and correct) | 🟡 |
| `fixture_id` uniqueness | 10,735 distinct IDs = 10,735 rows. **0 duplicates** | 🟢 |
| Raw fixture ID set vs DB fixture ID set | Raw: 10,735. DB: 10,735. **Raw-only IDs: 0. DB-only IDs: 0.** Exact match. | 🟢 |
| Checkpoint completeness | All 30 expected windows present, all `status: complete`. 0 missing, 0 unexpected. | 🟢 |
| Raw page validity | 60/60 files parse as valid JSON. 0 invalid. | 🟢 |
| Page/fixture counts vs API metadata | Every window's summed page fixture counts match `info.total` from the API's own response exactly. 0 mismatches. | 🟢 |
| Cross-window duplicate fixtures | 0 fixture IDs appear under more than one (competition, season) raw directory. | 🟢 |
| SQLite numeric column types | All checked numeric columns (goals, xG/xGOT, shots, shots_on, possession, corners, fouls, cards, attacks, pressure, tackles, offsides, goal_kicks, throw_ins, unix, competition_id, season_id) report `typeof()` as `integer`/`real`. **0 columns holding numeric data as TEXT.** | 🟢 |
| Reopen check | DB copied, closed, reopened fresh: row count identical (10,735) both times. | 🟢 |
| Raw → normalize → SQLite trace | 5 representative fixtures (one per league, spanning 2020/21, 2022/23, 2023/24, 2024/25, 2025/26) traced field-by-field from raw JSON through `normalize_fixture()` to the actual DB row. **All 5: exact match, 0 mismatches.** | 🟢 |
| Test suite | `python -m unittest discover -s tests` → **45/45 pass**, unchanged from Phase 1.1. | 🟢 |

---

## C. Coverage matrix (aggregate, all 10,735 fixtures)

| Field | Null % | Class | Note |
|---|---|---|---|
| home_name / away_name / home_id / away_id | 0.0% | 🟢 | |
| home_goals / away_goals | 0.01% (1 row) | 🟢 | The one null is a genuinely `ABANDONED` match (see §F/G) — not a data gap. |
| shots / shots_on / possession / attacks / dang_attacks | ~0.4% | 🟢 | Consistently near-complete across the whole 6-season window. |
| corners | 0.0% | 🟢 | |
| fouls | 0.39% | 🟢 | |
| yellow_cards | 0.95% | 🟢 | |
| pressure | 0.72% | 🟢 | |
| has_odds | 0.0% (true on 10,730/10,735) | 🟢 | 3 windows have 1-4 fixtures with `has_odds: false`; immaterial. |
| red_cards | 27.1% | 🟡 | Consistent with the original audit finding: null appears to mean "zero red cards," not missing data. Not re-verified statistically this pass — flagged, not fixed. |
| offsides | 19.3% | 🟡 | Elevated but fairly consistent across seasons (not a cutover pattern) — likely the same zero-vs-null ambiguity as red cards. |
| tackles | 15.4% overall, but **0% from 2022/23 onward** | 🟡 | See §C.1 below — this is a coverage-depth pattern, not random gaps. |
| goal_kicks / throw_ins | ~41% overall, but **~0-1% from 2023/24 onward** | 🟡 | See §C.1 below. |
| xG / xGOT | 65.1% overall | 🟡 | See §E — expected, driven by the pre-2024 blackout. |

### C.1 New finding: granular-stat rollout pattern (tackles, goal_kicks, throw_ins)

Not previously documented in the Phase 0 sufficiency study, because it wasn't tested at this depth. These three fields follow an xG-like "rolled out gradually" pattern rather than being uniformly available or unavailable:

| Season | tackles null% | goal_kicks null% | throw_ins null% |
|---|---|---|---|
| 2020/21 | 43.4% | 93.2% | 92.1% |
| 2021/22 | 47.1% | 92.2% | 91.5% |
| 2022/23 | **0.0%** | 54.7% | 51.9% |
| 2023/24 | 0.0% | **0.8%** | **0.8%** |
| 2024/25 | 0.1% | 0.0% | 0.1% |
| 2025/26 | 0.0% | 0.0% | 0.0% |

**Practical read:** `tackles` is reliable from 2022/23 onward; `goal_kicks`/`throw_ins` are reliable from 2023/24 onward. Using any of these three fields for a feature that spans the full 2020/21–2025/26 window would silently bias toward recent seasons unless this is explicitly handled (e.g., excluded from historical-depth features, or used only for the seasons where they're populated). **Classified 🟡 — real, previously-undocumented limitation, not a defect in this ingestion.**

Full per-league-per-season breakdown for all 22 fields: `data/audit/phase2_coverage_matrix_by_league_season_field.json`.

---

## D. Missing-data matrix

Full league × season × field null-count/percentage breakdown (30 windows × 22 fields = 660 data points) is in `data/audit/phase2_coverage_matrix_by_league_season_field.json`. Nothing in it contradicts the aggregate patterns in §C — no league behaves as an outlier relative to the others for any field.

---

## E. xG / xGOT coverage and cutover

**Refines, and is consistent with, the Feb–May 2024 bracket found in the original OddAlerts sufficiency study — now pinned down precisely using the full ingested dataset instead of spot-checks.**

| League | Earliest FT match with xG populated |
|---|---|
| Serie A | 2024-02-26 |
| Bundesliga | 2024-03-15 |
| La Liga | 2024-03-15 |
| Ligue 1 | 2024-03-15 |
| Premier League | 2024-03-16 |

- **Before 2024-02-26 (any league): 0 fixtures have xG populated.** Hard blackout, 100% consistent — 3,748 pre-cutover FT fixtures checked, all null.
- **The rollout was gradual within each league's 2023/24 season**, not an instant switch — using 2024-03-01 as a reference point, post-that-date FT matches in the 2023/24 season still show 47.5–57.9% null xG per league (this is the tail of the gradual rollout landing inside that season, not a new gap).
- **From the 2024/25 season onward, residual null-xG is small and consistent**: 0.3%–1.3% per league per season (2024/25 and 2025/26 across all 5 leagues) — essentially clean coverage, not a lingering structural gap. Full breakdown: `data/audit/phase2_xg_post_cutover_residual_gaps.json`.
- The latest FT match anywhere in the dataset still missing xG is 2026-01-10 (La Liga) — an isolated per-match gap, not evidence of a second cutover or a recurring pattern.
- xGOT tracks xG's null pattern almost exactly (6,984 vs 6,983 null) — no separate cutover.

**Class: 🟡** — real, quantified, and bounded. Any xG-dependent feature must be restricted to matches from late Feb/mid-March 2024 onward (season-safe cutoff: use 2024/25 season onward for a clean >98.7%-populated field; the tail end of 2023/24 is usable but noisier).

---

## F. 2024/25 fixture-count re-investigation

**The `season.progress` field from the OddAlerts API (81%/85% for several leagues in the original DATA_AUDIT.md) was misleading. Actual ingested fixture counts are complete and correct for every league's 2024/25 season:**

| League | Expected | Actual (ingested) | Match |
|---|---|---|---|
| Premier League 2024/25 | 380 | 380 | 🟢 |
| La Liga 2024/25 | 380 | 380 | 🟢 |
| Bundesliga 2024/25 | 306 | 306 | 🟢 |
| Serie A 2024/25 | 380 | 380 | 🟢 |
| Ligue 1 2024/25 | 306 | 306 | 🟢 |

Every one of the 30 windows matches its expected count exactly, including Serie A 2022/23 (381 — the season with one extra fixture, previously noted in `target_league_seasons.json` and confirmed here at the actual-data level, not just the API's season-metadata level). **The API's `progress` field does not reflect actual completed-fixture coverage and should not be used as a data-completeness signal** — this closes out the open question flagged in the original OddAlerts sufficiency study. **Class: 🟢** (resolved; the underlying data was always complete, only the metadata field was misleading).

---

## G. Impossible-value results

All checks returned **zero violations** across all 10,735 fixtures:

| Check | Violations |
|---|---|
| Negative home/away goals | 0 |
| Negative shots / shots on target | 0 |
| Negative cards (yellow or red, either side) | 0 |
| Negative corners | 0 |
| Negative fouls | 0 |
| Possession outside 0-100 | 0 |
| Home+away possession outside 98-102 (sanity band) | 0 |
| Negative xG / xGOT | 0 |
| Invalid unix timestamps (outside year 2000-2040) | 0 |
| Fixture date outside its configured season's date window | 0 |
| Future fixtures (`unix` beyond 2026-08-16) marked as `FT` | 0 |
| `competition_id`/`season_id` on a fixture not matching the raw-file directory it was stored under | 0 (checked across all 10,735 fixtures in all 60 raw files) |
| Rows missing `home_name`/`away_name`/`home_id`/`away_id` | 0 |

**One legitimate null found and explained, not hidden:** fixture `420450481` (Nantes vs Toulouse, Ligue 1 2025/26, dated 2026-05-17) has `status: ABANDONED` and both `home_goals`/`away_goals` are `NULL` — correct behavior for a match with no final score, not a data defect. This is the sole source of the "0.01%" goals-null figure in §C.

**Class: 🟢 across the board.**

---

## H. Token-security result

Full-project grep for the literal API token value, excluding `.env`: **zero occurrences** anywhere in `data/raw/` (all 60 files, including the previously-affected `next_page_url` fields — confirmed redacted via the Phase 1.1 fix, holding across all 30 windows this time, not just the one smoke-test window), `data/processed/matches.db`, `data/checkpoints/ingestion_state.json`, `logs/ingestion.log`, `data/samples/`, `data/audit/`, or any other generated file. **Class: 🟢.**

---

## I. Test result

`python -m unittest discover -s tests` → **45/45 pass**, run fresh against the current codebase (unchanged since Phase 1.1 — no code was modified during this audit, per instructions).

---

## J. All YELLOW / RED findings (consolidated)

No RED findings. Five YELLOW items, all informational/limitations rather than defects:

1. **Console "10355" vs actual DB total 10735** — fully reconciled; caused by one already-complete window being correctly skipped, not data loss.
2. **xG/xGOT: 65% null overall**, concentrated entirely in pre-2024 seasons by design of the data source, with a gradual rollout tail through the 2023/24 season. Clean (>98.7% populated) from the 2024/25 season onward.
3. **`tackles`/`goal_kicks`/`throw_ins`: new finding**, a coverage-depth rollout parallel to xG's — reliable only from 2022/23 (`tackles`) or 2023/24 (`goal_kicks`/`throw_ins`) onward. Must be excluded from, or handled explicitly in, any feature spanning the full historical window.
4. **`red_cards`/`offsides`: 27% / 19% null**, most likely a "null means zero events" convention (consistent with earlier project findings) rather than missing data — not independently re-verified this pass, so flagged rather than assumed.
5. **`has_odds` false on a handful of fixtures** in 3 windows (La Liga 2023/24, Serie A 2020/21, Serie A 2023/24) — 4-5 fixtures total, immaterial to any planned feature.

---

## K. Final verdict

**READY FOR FEATURE ENGINEERING**

Every structural integrity check (row counts, uniqueness, raw/DB reconciliation, checkpoint completeness, page validity, API-metadata cross-check, numeric typing, reopen, end-to-end trace, token security, impossible values, chronological consistency, future-fixture check, ID-to-directory consistency) came back clean with zero RED findings across all 10,735 fixtures and 30 windows. The five YELLOW items are real, quantified data-source characteristics (mostly rollout/coverage-depth patterns in specific fields) that the next phase needs to design around — not defects that need fixing before proceeding. In particular: xG-dependent and tackles/goal_kicks/throw_ins-dependent features must respect the season cutoffs documented in §E and §C.1 rather than assuming full 2020/21–2025/26 depth.
