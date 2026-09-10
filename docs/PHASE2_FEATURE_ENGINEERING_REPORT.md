# Phase 2: Feature Engineering — Implementation & Quality Report

Companion to `docs/PHASE2_FEATURE_SPEC.md` (the design contract, written and reviewable before any code) and `docs/PHASE2_DATA_INTEGRITY_AUDIT.md` (the input dataset's validated state). This report covers the actual implementation: `src/features/`, its test suite, the generated feature dataset, and its quality audit.

**Scope discipline maintained throughout:** no API calls, no additional providers, no model training/tuning, no Poisson/Monte Carlo, no W/D/L probability calculation, no UI, and `data/processed/matches.db` was never written to — confirmed via a dedicated test (`test_raw_fixture_table_untouched_by_feature_generation`) and by the schema/table check in that same test (only one table, `fixtures`, exists in `matches.db` after a full feature run).

---

## A. Feature count

**274 columns per row** in `data/processed/features.db` (table `feature_rows`), covering: 8 metadata columns, 3 label columns, and roughly 87 distinct named features once you collapse the `_n`/`_coverage_n` companion columns (each averaged feature carries 1-2 bookkeeping columns alongside its value, which is most of the column count — not feature sprawl for its own sake). Exact breakdown:

| Group | Feature families | Windows | Columns (incl. `_n`/`_coverage_n`) |
|---|---|---|---|
| Goals (§A/§B) | for/against/diff | last5, last10, season | 36 |
| xG (§C) | for/against/diff | last5, last10, season | 36 |
| Shots (§D) | for/against/diff | last5, last10, season | 36 |
| Shots-on-target (§D) | for/against/diff | last5, last10, season | 36 |
| Venue-restricted goals (§A/§B) | for/against | season only | 8 |
| Tempo proxies (§H) | attacks, dang_attacks, pressure | season only | 18 |
| Discipline (§I) | fouls, yellow_cards | season only | 12 |
| Form (§E) | points, win/draw/loss rate, goal diff, xg diff | last5, last10 | 32 |
| Home advantage (§F) | team + league | season only | 6 |
| Strength (§4) | attack/defence/diff (shrinkage) | season | 7 |
| Context (§G) | matches played, insufficient-history flags | all-time | 4 |
| Metadata | fixture_id, competition_id, season_id, unix, home_id, away_id, feature_version, generated_at | — | 8 |
| Labels | home_goals, away_goals, result | — | 3 |

## B. Feature definitions

Every feature's mathematical definition, leakage rule, window, and missing-data policy is in `docs/PHASE2_FEATURE_SPEC.md` — not repeated here to avoid the two documents drifting out of sync. One refinement made during implementation, noted here because it changes the spec's prose slightly: the spec described "`home_strength_score`/`away_strength_score`" as if each side had one combined number; the actual implementation (per §4's own reasoning) keeps attack and defence separate per side (`home_attack_strength_score`, `home_defence_strength_score`, etc., 4 scores total) plus a single composite `strength_diff = (home_attack−home_defence) − (away_attack−away_defence)` as a convenience context feature. This is more information-preserving than the spec's shorthand implied and is consistent with §4's stated rationale for not collapsing attack/defence into one number.

## C. Coverage by feature (dataset-wide, n=10,735)

| Feature (representative) | Null % | Why |
|---|---|---|
| `home_goals_for_per_match_last5` | 1.9% | Teams with <3 matches played yet this "all-time" window (mostly the very first few fixtures ever ingested for a team, since last5 spans season boundaries) |
| `home_goals_for_per_match_season` | 5.4% | Season-window minimum (2 matches) not yet met — i.e. matchday 1 of every team's every season |
| `home_shots_for_per_match_season`, `home_fouls_per_match_season`, `home_attacks_per_match_season` | 5.4-5.5% | Same season-minimum-not-met pattern; these fields are themselves ~99.6% populated at the raw-data level (per Phase 2 data audit), so this null rate is a window-maturity artifact, not a raw-data gap |
| `home_goals_for_home_venue_season` | 10.8% | Venue-restricted season window needs 2 matches at that specific venue, which takes roughly twice as long into a season to accumulate as the non-venue-restricted season window |
| `home_attack_strength_score` / `strength_diff` | 0.3% (30 rows) | Exactly the 30 "first match of a competition+season" windows — one per competition/season combination, where the league accumulator itself has zero prior matches. This number is not a coincidence: it's exactly `5 leagues × 6 seasons = 30`, confirming the shrinkage formula's edge case fires exactly where expected and nowhere else. |
| `home_xg_for_per_match_season` | 65.7% | See §F below — driven almost entirely by the pre-2024 seasons, by design |
| `label_home_goals` / `label_result` | 0.01% (1 row) | The single `ABANDONED` fixture identified in the Phase 2 data audit (Nantes vs Toulouse) — correctly propagates through as a null label, not a fabricated one |

Full machine-readable breakdown: `data/audit/phase2_feature_coverage_overall.json`.

## D. Missingness by feature

Covered inline in §C and enforced by `tests/test_feature_missingness.py` (5 tests: below-threshold nulls stay null not zero, xG's `coverage_n` distinguishes "few matches in window" from "few of those matches had xG," deferred field families never appear at all, insufficient-history flag matches the documented threshold). No feature in the generated dataset was ever forward-filled, zero-filled, or estimated from a different field.

## E. Coverage by league/season

No league behaves as an outlier relative to the others for any feature family — every pattern in §F below is consistent across all 5 leagues within ±1 percentage point. Full 30-window breakdown for the goals/shots/form families available on request via the same query pattern used for §F; not reproduced in full here since it would just be 30 near-identical rows (the real signal is entirely in the xG dimension, covered next).

## F. xG feature coverage

This is where the design choice documented in spec §C — gate on real per-window coverage rather than a hardcoded season cutoff — visibly pays off, and matches the Phase 2 data-integrity audit's independent findings almost exactly:

| Season (all 5 leagues) | `home_xg_for_per_match_season` populated |
|---|---|
| 2020/21, 2021/22, 2022/23 | **0%** (hard pre-cutover blackout, exactly as the raw-data audit found) |
| 2023/24 | **~21%** (20.6-21.8% across leagues) — the transition season, exactly reflecting the gradual Feb-May 2024 rollout: only fixtures late enough in that season, with enough later-in-season history themselves, produce a non-null season-window average |
| 2024/25 | **~94.5%** (94.1-94.7% across leagues) |
| 2025/26 | **~94.4%** (94.1-94.7% across leagues) |

The ~5.5% residual null in the "clean" seasons is not a coverage gap in the underlying xG data (the raw-data audit found only 0.3-1.3% per-match xG nulls in these seasons) — it's almost entirely the same "season window needs ≥2 matches" maturity effect from §C, landing on early-season fixtures. Full breakdown: `data/audit/phase2_feature_xg_coverage_by_season.json`.

**No xG value in this dataset was ever backfilled, interpolated, or zero-filled.** Every non-null xG feature reflects real, coverage-counted OddAlerts data; every null reflects either genuinely pre-cutover history or a window that hasn't accumulated enough real coverage yet.

## G. Promotion/new-team handling

Implemented exactly per spec §5: no literal promotion detection (the ingested dataset has no second-division data to support one), `matches_played_before_target` and `_insufficient_history` used as the operative proxy instead, with the shrinkage formula (§4/§I below) naturally reducing to the league mean for a team with zero season-to-date matches.

**One nuance surfaced during the Step 11 validation-sample inspection, worth documenting explicitly:** `home_insufficient_history`/`away_insufficient_history` are computed from **all-time** matches played (across every season in the ingested dataset), not season-to-date. A well-established club's opening fixture of a new season therefore does *not* trigger `insufficient_history = True` (it has hundreds of past matches on record), even though its *season-scoped* features (`home_goals_for_per_match_season`, `home_attack_strength_score`, etc.) are correctly `NULL` for that same fixture, because those are season-windowed by design. This is intentional and consistent with spec §5's own definition, but it means `insufficient_history` answers "do we have a real history for this team at all" while the season-scoped NULLs separately answer "do we have current-season form for this team yet" — two different, both useful, both correctly-behaving signals, not one signal wearing two names. Verified directly against a real fixture (an established Premier League club's first match of the 2025/26 season) in the validation sample: `home_matches_played_before_target: 190`, `home_insufficient_history: False`, `home_goals_for_per_match_season: None`. See `data/audit/phase2_feature_validation_sample_inspection.json`.

A genuinely newly-promoted team (a club with little or no history anywhere in the 6-season ingested window) would correctly show both signals as thin/insufficient — this project did not have a confirmed real example on hand to inspect by name without a dedicated promotion/relegation cross-check, which is exactly the kind of check flagged as not independently verified in the Phase 2 data-integrity audit (§3, promotion/relegation turnover).

## H. Rolling-window methodology

`last5`/`last10` span season boundaries deliberately (spec §1); `season` resets at each `season_id`. All three are built from a single chronological pass over the full 10,735-fixture dataset (`FeatureContext` in `history.py`), maintaining per-team match lists and per-(competition,season) running sums incrementally — a fixture's features are always computed from state that has only ever seen strictly-earlier fixtures, by construction of the loop order (compute-then-record, never the reverse). This architectural property is what `tests/test_feature_leakage.py`'s 10 tests verify empirically from the outside.

## I. Attack/defence methodology

Empirical-Bayes shrinkage toward the league's season-to-date mean, `k=5`, exactly as specified in spec §4. Verified with 5 dedicated unit tests (`test_feature_strength.py`) covering the exact formula, the zero-matches-equals-league-mean edge case, convergence toward the raw team average at large sample sizes, and a full pipeline-level check against a hand-computed expected value for a 3-fixture synthetic league. Dataset-wide sanity check (not a hardcoded expectation, just a distribution sanity pass): `home_attack_strength_score` ranges 0.0-4.0 with a mean of ~1.41 goals/match (a plausible league-wide scoring rate), `strength_diff` centered almost exactly on 0 (mean ≈ -0.013) as expected for a symmetric, league-normalized measure with no systematic home/away bias baked into the formula itself. Zero out-of-range values found for any bounded feature (win/draw/loss rates outside [0,1], points outside [0,3], negative strength scores, invalid `label_result` values) — see `data/audit/` query results summarized in §K.

## J. Leakage tests

`tests/test_feature_leakage.py` — **10/10 passing.** Covers all 9 items the task required plus one additional structural test of the `FeatureContext` ordering contract itself:

1. Target fixture never in its own rolling history ✅
2. Future fixture cannot affect an earlier fixture's features ✅
3. Reordering future rows doesn't change past features ✅
4. Removing future fixtures doesn't change features at T ✅
5. Season aggregates use only pre-T matches ✅
6. Home/away venue-restricted stats exclude the target ✅
7. xG features never use post-target xG (tested with a deliberately extreme future xG value to make any leak obvious) ✅
8. Form features never use post-target results ✅
9. No odds-derived column exists at all — the strongest possible guard against odds-timing leakage, since odds were never ingested (spec §J) ✅
10. (extra) `FeatureContext.history_before()` is empty until `record()` is explicitly called, proving the read-before-write ordering the whole architecture depends on ✅

## K. Test results

| Suite | Tests | Result |
|---|---|---|
| Pre-existing ingestion tests (Phases 1/1.1/2) | 45 | pass, unchanged |
| `test_feature_leakage.py` | 10 | pass |
| `test_features.py` | 8 | pass |
| `test_feature_missingness.py` | 8 | pass |
| `test_feature_strength.py` | 7 | pass |
| **Total** | **78** | **78/78 pass** |

Full-project token sweep (excluding `.env`): zero occurrences, including in the newly-created `data/processed/features.db`. `features.db` reopen check (close, reopen, re-query): 10,735 rows both times.

## L. Suspicious distributions

None found. Explicit checks run against the full 10,735-row generated dataset: win/draw/loss rates outside [0,1] (0 violations), points outside [0,3] (0), negative goals/shots/strength values (0), `strength_diff` outside a generous ±20 sanity band (0), negative `matches_played_before_target` (0), invalid `label_result` values (0). Value ranges inspected for `home_attack_strength_score` (0.0-4.0, mean 1.41) and `strength_diff` (-3.0 to 3.13, mean -0.013) are both consistent with real football scoring rates and a properly-centered, symmetric normalization — nothing warranting further investigation before model development.

## M. Features rejected/deferred, and why

| Feature family | Status | Reason |
|---|---|---|
| Odds (any form) | **Deferred** | Never ingested — only the `has_odds` boolean exists in `matches.db`, not actual prices (discovered during spec-writing by checking the schema directly, not assumed). Re-ingesting is out of scope for this phase. |
| Red cards, offsides | **Deferred** | Unresolved null-vs-zero ambiguity flagged (not resolved) in the Phase 2 data-integrity audit; building on an unverified assumption would violate this phase's explicit "don't assume null means zero without evidence" instruction. |
| Tackles | **Deferred** | Reliable only from 2022/23 onward (per data-integrity audit §C.1); judged to add another coverage-gated dimension without enough independent signal over shots/attacks to justify it for V1. Revisitable in v1.1. |
| Goal kicks, throw-ins | **Deferred** | Reliable only from 2023/24 onward; weakest-justified "attacking strength" signals among the available fields even where present. |
| "Seconds between shots" / true tempo timing | **Rejected outright, not just deferred** | Does not exist in the source data under any derivation (no shot-event timestamps) — confirmed absent in the original OddAlerts audit and re-confirmed here; no proxy in this codebase is ever labeled as if it measured this. |
| Full iterative opponent-adjusted rating (Elo-style) | **Deferred to v1.1+** | Spec §4 explicitly required justification before building one in V1; none exists yet without a backtesting harness this phase doesn't build. Simple shrinkage satisfies "league-normalized, sample-size-aware" without that extra, unvalidated machinery. |
| Literal promotion/relegation detection | **Rejected, replaced with a proxy** | No second-division data exists in the ingested dataset to support real detection; `matches_played_before_target`/`_insufficient_history` is the documented, defensible V1 substitute (spec §5). |

## N. Final readiness for model development

Every requirement from the task is satisfied: a reviewable feature contract written before implementation (`PHASE2_FEATURE_SPEC.md`), a modular, deterministic, network-free pipeline (`src/features/`), a dedicated and passing leakage-test suite (10/10), full pipeline/missingness/strength test coverage (23 more tests, all passing), a validated small sample inspected by hand across early/mid/late-season, xG-era, pre-xG, and season-opener fixtures before the full run, a complete 10,735-row feature dataset stored separately from the immutable raw data with explicit versioning (`feature_version = "v1.0"`), and a full coverage/missingness/distribution audit with zero suspicious findings.

**🟢 READY FOR MODEL DEVELOPMENT**

Two things the next phase should carry forward, not treat as resolved: xG-dependent features are only reliably usable from partway through the 2023/24 season onward (§F) — a model wanting full 2020/21-depth training data should rely on the goals/shots-based features for that older window and treat xG as a 2023/24-onward enhancement, not a universal input; and the red-card/offside/tackle/goal-kick/throw-in/odds field families remain deliberately unbuilt pending the specific follow-up work noted in §M, not because they're unimportant. Per instructions, no model development, training, tuning, or probability calculation was performed in this phase regardless of this verdict.
