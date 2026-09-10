# V4 Promotion — Phase 0 Audit Report

**Date:** 2026-08-20
**Status:** Audit only. **No implementation code written. No production file modified.**
**Per the Phase 0 rule:** *"STOP if any of these cannot be established. Do not guess."*

---

## HEADLINE — ONE BLOCKING FINDING

> **E2 market odds do not exist for the 20 validation matches.**
>
> The 20 matches are 2025/26 fixtures. The E2 market dataset was deliberately built with 2025/26 quarantined, so it stops at 2024/25. **0 of 20** have market probabilities.

E1 causal Elo and E6 online A/D both work correctly at the required cutoff — verified, not assumed. The blocker is isolated to the market component, and it is a **data-coverage gap, not a design flaw**. It is fixable with an existing script and your approval.

Because Phase 3 requires E2 to be part of V4, and Phase 8 requires a market column for all 20 matches, I am stopping here rather than silently producing a two-component V4 or a comparison table with 20 blank market cells.

---

## Phase 1 — Protected File Integrity (PRE)

| File | MD5 | SHA256 (first 32) | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `9e662d0dfb0a69e4fd6119dad7d089179...` | **PASS** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `20eca80229c39222541fe56f61800a75f...` | **PASS** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `97cd5c4bdb63ba31424d8fa74ebe14747...` | **PASS** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `04b71da9b03efb7a5816ee68fc29ccada...` | **PASS** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `10d86d374bfd5c41afbe236f34255a5e0...` | **PASS** |

All 5 match expected values. Re-verified after the audit — **unchanged**.

---

## Phase 0 — The 15 Audit Questions

### Q1. Current production inference entry point

`predict_match.py`. Flags: default → V2, `--v1` → V1, `--v3` → V3 (mutually exclusive group). `--json` for machine output.

### Q2. Model artifact currently used

**Default production model is V2**: `data/models/v2_poisson_venue.pkl` via `v2_artifact.DEFAULT_V2_ARTIFACT_PATH`. V3 is wired but **not** the default — it resolves `v3_poisson_venue_elo.pkl` (promoted, absent) then falls back to `v3_poisson_venue_elo_candidate.pkl`.

**This matters for Phase 7.** "The existing production model" is **V2**, not V3.

### Q3. Where the V3 87 features are constructed

`data/processed/features.db` (`feature_rows`) for 84, plus 3 causal Elo computed at inference by `src/features/elo.py`. Contract: `src/models/v3_contract.py::V3_FEATURE_COLUMNS`.

**Verified: all 20 validation fixtures have feature rows (20/20).**

### Q4. Where causal Elo is implemented

`src/features/elo.py` — production module, two-pass per timestamp, `INIT=1500.0`, `K=20.0`, `HOME_ADVANTAGE=100.0`, `MEAN_REVERSION=0.0`. Validated bit-identical to the research engine in E1 across all 10,735 fixtures.

### Q5. Where Online A/D is implemented

`research/online_attack_defense/online_attack_defense.py` — E6's validated module. **Currently research-only**; promotion to V4 requires either importing from `research/` or copying into `src/features/`. That is a Phase 2 decision, flagged below.

### Q6. Where market odds are available

`research/market_odds/research_dataset.sqlite`, table `research_odds`: 4,001 fixtures, Pinnacle only, power de-vigged closing 1X2.

**Seasons present: 2022/23, 2023/24, 2024/25 only.**

### Q7. Can E1 causal Elo be computed strictly before kickoff? — **YES, verified**

Computed for **20/20** fixtures. `home_elo` range over the 20: **[1382.4, 1784.5]**. Ratings carry across seasons as designed.

### Q8. Can E6 A/D states be computed strictly before kickoff? — **YES, verified**

Computed for **20/20** fixtures. `A_home` range: **[−0.3267, +0.4796]** — well inside the ±1.5 clip. Baseline rates fitted on **8,983 pre-cutoff fixtures**: `mu_home=1.5345`, `mu_away=1.2826`.

### Q9. Can market odds be a separate signal without becoming a training feature? — **ARCHITECTURALLY YES, BUT NO DATA**

Architecturally straightforward: E2's validated form is a post-hoc probability signal that never enters a design matrix, so keeping it separate is the default rather than a special measure.

**But `research_odds` contains 0 of the 20 fixtures.** See the blocking finding.

### Q10. Files that would need modification for V4

| File | Change |
|---|---|
| `data/models/v4_poisson_venue_elo_online_ad.pkl` | **NEW artifact** |
| `research/v4_promotion/*` | New research code, tests, reports |
| `src/models/v4_contract.py` | **NEW** — 91-column contract (proposed) |
| `src/features/online_attack_defense.py` | **NEW** — promote E6 out of `research/` (proposed) |
| `predict_match.py` | Add `--v4` path, preserving V1/V2/V3 untouched |

Only the last is an edit to an existing production file, and it is additive — the same pattern used for `--v3`.

### Q11. Files that must remain protected

The five in Phase 1, plus `src/models/v3_contract.py`, `src/models/poisson.py`, `src/features/elo.py`, and the existing `predict_v1`/`predict_v2`/`predict_v3` functions.

### Q12. The exact 20 validation matches — **ESTABLISHED**

Selection rule from `predict_blind_2025_26.py`, deterministic and documented:

```sql
SELECT fixture_id FROM fixtures
WHERE season_id IN (2025/26 ids) AND status = 'FT'
ORDER BY unix ASC, fixture_id ASC
LIMIT 20
```

(1,751 FT fixtures exist in 2025/26; these are the earliest 20.)

| # | fixture_id | Date | League | Home | Away | Score |
|---|---|---|---|---|---|---|
| 1 | 343465962 | 2025-08-15 | La Liga | Girona | Rayo Vallecano | 1-3 |
| 2 | 343465735 | 2025-08-15 | Ligue 1 | Rennes | Olympique Marseille | 1-0 |
| 3 | 342254747 | 2025-08-15 | Premier League | Liverpool | AFC Bournemouth | 4-2 |
| 4 | 343465963 | 2025-08-15 | La Liga | Villarreal | Real Oviedo | 2-0 |
| 5 | 342863654 | 2025-08-16 | Premier League | Aston Villa | Newcastle United | 0-0 |
| 6 | 342863842 | 2025-08-16 | Premier League | Brighton & Hove Albion | Fulham | 1-1 |
| 7 | 342863843 | 2025-08-16 | Premier League | Sunderland | West Ham United | 3-0 |
| 8 | 342863844 | 2025-08-16 | Premier League | Tottenham Hotspur | Burnley | 3-0 |
| 9 | 343465733 | 2025-08-16 | Ligue 1 | Lens | Olympique Lyonnais | 0-1 |
| 10 | 342864234 | 2025-08-16 | Premier League | Wolverhampton Wanderers | Manchester City | 0-4 |
| 11 | 343465729 | 2025-08-16 | Ligue 1 | Monaco | Le Havre | 3-1 |
| 12 | 342864255 | 2025-08-16 | La Liga | Mallorca | FC Barcelona | 0-3 |
| 13 | 343465732 | 2025-08-16 | Ligue 1 | Nice | Toulouse | 0-1 |
| 14 | 342863998 | 2025-08-16 | La Liga | Deportivo Alavés | Levante | 2-1 |
| 15 | 342864289 | 2025-08-16 | La Liga | Valencia | Real Sociedad | 1-1 |
| 16 | 343465726 | 2025-08-17 | Premier League | Nottingham Forest | Brentford | 3-1 |
| 17 | 343465727 | 2025-08-17 | Premier League | Chelsea | Crystal Palace | 0-0 |
| 18 | 343465734 | 2025-08-17 | Ligue 1 | Brest | LOSC Lille | 3-3 |
| 19 | 343465842 | 2025-08-17 | La Liga | Celta de Vigo | Getafe | 0-2 |
| 20 | 343465728 | 2025-08-17 | Ligue 1 | Angers SCO | Paris | 1-0 |

**Composition:** 9 Premier League, 6 La Liga… — actually 8 Premier League, 6 La Liga, 6 Ligue 1. **No Serie A, no Bundesliga** (those leagues started later). Actual outcomes: 12 H, 4 D, 4 A.

Two observations worth recording now, before any result exists:

1. **Three of five leagues only.** Any per-league reading of this sample is impossible.
2. **Opening weekend of a new season.** Rolling form features (last5/last10, season-to-date) are at their least informative here — every team has zero 2025/26 history. This is the hardest possible slice for a form-based model and the most favourable for Elo/A-D, which carry state across seasons. It cuts both ways and should temper any conclusion in either direction.

### Q13. Information cutoff — **ESTABLISHED**

Earliest kickoff of the 20: **unix 1755277200 = 2025-08-15 17:00 UTC**. Last: **unix 1755443700 = 2025-08-17 15:15 UTC**.

Per-fixture causal cutoff: each fixture's own `unix`. E1 and E6 both honour this by construction (two-pass per timestamp). Note fixtures 5–15 share 2025-08-16 and several share exact kickoff times, so the **two-pass requirement is live in this sample**, not hypothetical.

### Q14. Is the old-model prediction reproducible exactly? — **YES, conditionally**

`predict_blind_2025_26.py` is deterministic (fixed artifact, fixed selection, no RNG) and hard-fails if the artifact was trained on 2025/26. Reproduction requires **scikit-learn**, which is unavailable in my sandbox but present in your local environment.

**Important caveat on the "9/20" figure:** `run_step10a_previous_20_reconstruction_audit.py` documents that the historical 9/20 was produced by *manual* comparison of two printed tables; **no machine-readable record exists**, and that audit could not fully reconcile 8 vs 9. I will therefore **regenerate the old-model baseline from the artifact** rather than treating 9/20 as ground truth. Phase 7's requirement is reproducibility of the *model*, which holds; the historical count is not a reliable anchor and I will not use it as one.

### Q15. Can V4 use exactly the same cutoff? — **YES for E1/E6, NO for E2 as things stand**

- V3 87 features: available, 20/20
- E1 Elo: computed, 20/20
- E6 A/D: computed, 20/20
- **E2 market: 0/20**

---

## The blocking finding in detail

`matches.db` records `has_odds = 1` for **all 1,752** 2025/26 fixtures — the odds exist at the OddAlerts source. But `research/market_odds/ingest_odds_history.py` line 52:

```python
# Eligible seasons (2025/26 quarantined)
ELIGIBLE_SEASONS = ["2020/2021", "2021/2022", "2022/2023",
                    "2023/2024", "2024/2025"]
```

2025/26 was excluded **deliberately and correctly** — during E2–E9 it was the untouched holdout, and ingesting it would have risked contaminating research.

That reasoning no longer applies here. V4 promotion validation is **not** model development: the 20 matches are already-played fixtures being used to compare two finished models, and 2025/26 was formally spent as the holdout in the earlier production evaluation. Fetching *pre-match closing odds* for fixtures whose outcomes are already known introduces no training leakage — the odds are pre-kickoff information by construction, and nothing is fitted on them.

**But that is a change of scope for the quarantine rule, and it is your call, not mine.**

---

## What I need from you — one decision

**Option A — Ingest 2025/26 market odds, then build the full three-component V4.** *(recommended)*

```powershell
# after changing ELIGIBLE_SEASONS to include "2025/2026"
python research\market_odds\ingest_odds_history.py
```

Fetches ~1,752 fixtures from the existing OddAlerts subscription. Writes only to `research/market_odds/odds_history.sqlite` and `research_dataset.sqlite` — **no protected file is touched**. Then V4 = E1 + E6 + E2 exactly as specified.

**Option B — Build V4 as E1 + E6 only, market reported as "unavailable".** V4 becomes a 91-feature model with no market column, and Phase 8's market row is blank for all 20. Honest, but it is not the V4 you specified and drops a component that PASSed.

**Option C — Choose 20 different matches from a market-covered season.** Rejected in advance: it would abandon the "exact same 20 matches previously used" requirement and amount to selecting a convenient sample. I raise it only to record that I considered and rejected it.

I recommend **A**. The quarantine served its purpose through E9; retaining it now blocks a legitimate promotion check on already-played fixtures, and the ingestion is additive and reversible.

---

## Secondary decision (Phase 2)

E6's module lives in `research/online_attack_defense/`. V4 is a **production** candidate, so it should not import from `research/`. I propose copying it verbatim to `src/features/online_attack_defense.py` — a **new** file, no existing production file modified, algorithm unchanged and re-verified by the V4 test suite.

Confirm or redirect.

---

## Audit summary

| # | Question | Status |
|---|---|---|
| 1 | Production entry point | **Resolved** — `predict_match.py` |
| 2 | Current artifact | **Resolved** — V2 is the default, not V3 |
| 3 | V3 features | **Resolved** — 20/20 available |
| 4 | Causal Elo | **Resolved** — `src/features/elo.py` |
| 5 | Online A/D | **Resolved** — in `research/`, needs promotion |
| 6 | Market odds | **Resolved** — 2022/23–2024/25 only |
| 7 | Elo before kickoff | **VERIFIED** — 20/20 |
| 8 | A/D before kickoff | **VERIFIED** — 20/20 |
| 9 | Market as separate signal | **Architecture yes, DATA MISSING** |
| 10 | Files to modify | **Resolved** — 5, one additive production edit |
| 11 | Files to protect | **Resolved** |
| 12 | The exact 20 matches | **RESOLVED** — listed above |
| 13 | Information cutoff | **RESOLVED** — 2025-08-15 17:00 UTC |
| 14 | Old model reproducible | **YES** (needs sklearn); 9/20 anchor unreliable |
| 15 | V4 same cutoff | **YES for E1/E6, NO for E2** |

**14 of 15 resolved. Q9 blocks.**

---

## STOP

Phase 0 says stop rather than guess, so I am stopping before Phase 2.

Nothing was implemented. No production file was modified. All 5 protected files verified identical before and after this audit.

Roadmap unchanged: **E1 PASS · E2 PASS · E6 PASS · E7 FAIL · E8 FAIL · E9 FAIL.**
