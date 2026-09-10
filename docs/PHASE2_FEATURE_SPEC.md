# Phase 2: Feature Contract — V1 Feature Specification

Defines every V1 feature before any feature-generation code is written, per the project's leakage-prevention and "design before implementing" requirements. Grounded directly in `data/processed/matches.db` as inspected (84 columns, 10,735 rows, single row per fixture with `home_*`/`away_*` columns — not one row per team) and in the findings of `docs/PHASE2_DATA_INTEGRITY_AUDIT.md`.

**FEATURE_VERSION = "v1.0"**. Every row in the feature dataset records this version. Any future change to a feature's definition gets a new version rather than silently overwriting v1.0's methodology.

---

## 0. Global rules (apply to every feature below)

- **Leakage rule (universal):** for a target fixture T with kickoff time `T.unix`, every feature is computed using only fixtures with `unix < T.unix` and `status = 'FT'` (or `'AWARDED'`, which the audit found has real scores — see §0.1), for the relevant team(s)/league. `T` itself is always excluded from its own history, unconditionally, even if some other fixture shares its exact `unix` (defensive tie-break: exclude by `fixture_id != T.fixture_id` in addition to the time filter).
- **Chronological ordering key:** `unix` (fixture kickoff time), not `fixture_id` or ingestion order — `fixture_id` values are not guaranteed chronological (confirmed: OddAlerts' internal IDs mix leagues/years by insertion order, not date). Ties on `unix` are broken by `fixture_id` for determinism only, never to leak the target.
- **Status filter:** only `FT` and `AWARDED` fixtures count as "played" for history purposes. `ABANDONED` (1 fixture in the dataset, null goals) is excluded from every rolling calculation — it has no valid result to contribute. Upcoming/not-yet-played fixtures obviously never appear in this dataset's target set anyway (Phase 2 audit confirmed zero future `FT` fixtures).
- **No fabrication rule:** a feature that cannot be computed from sufficient real data is `NULL`, never `0`, never forward-filled, never estimated from a different field pretending to be the same thing.
- **Coverage transparency rule:** every averaged/rolling feature is stored alongside a `_n` (sample size actually used) and, for coverage-gated fields (xG, tackles, goal_kicks, throw_ins), a `_coverage_n` (how many of the `_n` matches had that specific field populated). A model consuming these features can see not just the value but how much real data backs it.
- **Reproducibility rule:** the pipeline is a pure function of `data/processed/matches.db` — same DB in, same feature table out, every time. No network access, no randomness, no wall-clock dependency (dates come from the data, not `datetime.now()`).

### 0.1 `AWARDED` status handling
The Phase 2 audit found 1 `AWARDED` fixture and did not investigate it further. Before finalizing: `AWARDED` fixtures are matches decided administratively (forfeit) rather than played out — OddAlerts still supplies a `home_goals`/`away_goals` result for these in the data inspected. **V1 policy: treat `AWARDED` identically to `FT`** for history purposes (it has a real, final result), but this is flagged as a decision, not an assumption — if a later audit finds `AWARDED` results are unreliable, this is the one line to change.

---

## 1. Historical windows

Three windows, computed for every feature family that supports rolling calculation:

| Window | Definition | Minimum matches to emit a value |
|---|---|---|
| `last5` | Most recent 5 matches (any competition round, home or away) strictly before T | 3 (else NULL) |
| `last10` | Most recent 10 matches strictly before T | 5 (else NULL) |
| `season` | All matches strictly before T within T's own `season_id` for that team | 2 (else NULL) |

Minimums exist because an average of 1 match is not a "rolling average," it's a single data point dressed up as one — the coverage-count field lets a model tell the difference regardless, but a hard floor avoids emitting a feature that's really just noise dressed as signal. These thresholds are a defensible, simple V1 choice, not empirically tuned — documented as such, and a candidate for revisiting once backtesting exists (Phase 11+ per the project's stated workflow).

`last5`/`last10` deliberately span across seasons (a team's 3rd match of a new season still has a `last5` window filled from the end of last season) — cutting `last5` off at a season boundary would make the first few matches of every season undersupplied for no statistical reason. `season` is season-scoped by definition, since its entire purpose is "how has this team performed so far this season."

---

## 2. Feature groups

### A. Team attacking strength (goals)

| Feature | Definition | Source | Window | Home/away | Missing-data handling | All seasons? | Leakage rule | V1? |
|---|---|---|---|---|---|---|---|---|
| `home_goals_for_per_match_{w}` | mean(goals scored) over home team's last-`w` matches (any venue) | `home_goals`/`away_goals` depending on which side the team played | last5, last10, season | Team-level (not venue-restricted) | NULL if `_n` < window minimum | Yes (goals have 0.01% null, single explained exception) | Strict `unix < T.unix` | Yes |
| `away_goals_for_per_match_{w}` | same, for the away team of T | same | last5, last10, season | Team-level | same | Yes | same | Yes |
| `home_goals_for_home_venue_{w}` | mean(goals scored) restricted to the home team's own past home matches only | `home_goals` where team was home | season only (last5/10 too thin once venue-restricted for most teams) | Home-venue only | NULL if `_n` < 2 | Yes | same | Yes |
| `away_goals_for_away_venue_{w}` | mean(goals scored) restricted to the away team's own past away matches only | `away_goals` where team was away | season only | Away-venue only | NULL if `_n` < 2 | Yes | same | Yes |

### B. Team defensive strength (goals)

Mirror of A using goals conceded (i.e. the opponent's goals in each of the team's past matches): `home_goals_against_per_match_{w}`, `away_goals_against_per_match_{w}`, `home_goals_against_home_venue`, `away_goals_against_away_venue`. Same rules, same thresholds. All V1.

### C. xG-enhanced strength

| Feature | Definition | Coverage policy | V1? |
|---|---|---|---|
| `home_xg_for_per_match_{w}`, `away_xg_for_per_match_{w}` | mean(`stat_home_xg`/`stat_away_xg` for the team's side) over window, **using only the subset of the window's matches where xG is non-NULL** | Emitted only if `_coverage_n >= 3` (last5/last10) or `>= 2` (season); else NULL. `_coverage_n` always stored alongside so a model/analyst can see e.g. "2 of 5" vs "5 of 5." | Yes, coverage-gated |
| `home_xg_against_per_match_{w}`, `away_xg_against_per_match_{w}` | mirror, opponent's xG in the team's past matches | same | Yes, coverage-gated |
| `xg_diff_{w}` (home minus away, both sides' xg-for-minus-against) | derived from the above | NULL if either side's underlying xg feature is NULL | Yes, coverage-gated |

**Explicit xG policy (per the Phase 2 audit's documented cutover):** xG is not backfilled, not zero-filled, and not treated as available just because a season is "recent enough on average." The per-match, per-window coverage count is the only gate — this naturally makes xG features NULL for any window sitting mostly or entirely before ~2024-03 (see audit §E) without needing a separate hardcoded season cutoff, and naturally starts producing values mid-way through the 2023/24 season as real coverage appears, exactly matching the gradual rollout the audit found. A hardcoded "season >= 2024/25 only" rule was considered and rejected because it would throw away real, valid xG data from the second half of 2023/24 for no reason.

### D. Shot-based features

`home_shots_for_per_match_{w}`, `away_shots_for_per_match_{w}`, `home_shots_against_per_match_{w}`, `away_shots_against_per_match_{w}`, and the same four for shots-on-target (`stat_home_shots_on`/`stat_away_shots_on`). Full historical depth (shots fields are ~99.6% populated dataset-wide, no cutover pattern found) — same `_n`/coverage-count mechanism as everything else, using per-match non-null filtering rather than assuming full coverage. All V1.

`shot_diff_{w}` = shots-for minus shots-against, derived. V1.

**Explicitly NOT built:** any feature implying shot timing (e.g. "seconds between shots," shot clustering, tempo-within-match). OddAlerts does not expose shot event timestamps (confirmed absent in `DATA_AUDIT.md` and the OddAlerts V1 sufficiency study) — no derivation of this exists in this codebase.

### E. Form

| Feature | Definition | Window | V1? |
|---|---|---|---|
| `points_{w}` | 3×wins + 1×draws over the team's last-`w` matches | last5, last10 | Yes |
| `win_rate_{w}`, `draw_rate_{w}`, `loss_rate_{w}` | fraction of last-`w` matches won/drawn/lost | last5, last10 | Yes |
| `goal_diff_{w}` | mean(goals for − goals against) over last-`w` | last5, last10 | Yes |
| `xg_diff_form_{w}` | mean(xg for − xg against) over last-`w`, coverage-gated exactly as in §C | last5, last10 | Yes, coverage-gated |

Result classification (win/draw/loss) for a historical match from a given team's perspective is derived from that match's own `home_goals`/`away_goals`/`winning_team` fields at the time it was itself a completed match — never from T. `winning_team` in the raw schema stores the **winning team's numeric ID** (not the string `"home"`/`"away"` as originally assumed in `DATA_AUDIT.md` — corrected here after direct DB inspection per Step 1's "verify from the database" instruction), so result classification is computed as `home_goals` vs `away_goals` directly (more robust than trusting `winning_team`, which is NULL for the ~25% of matches that were draws and would otherwise need special-casing anyway).

### F. Home advantage

| Feature | Definition | Leakage rule | V1? |
|---|---|---|---|
| `home_team_home_goal_diff_season` | mean(goals for − against) for the home team, restricted to that team's own past **home** matches this season | strict pre-T | Yes |
| `away_team_away_goal_diff_season` | mirror for the away team's past **away** matches this season | strict pre-T | Yes |
| `league_home_advantage_season` | mean(home_goals − away_goals) across **all** matches in T's league+season strictly before T (any team) | strict pre-T, league/season-scoped, T itself excluded | Yes |

`league_home_advantage_season` is a league-level context feature, not a per-team one — it answers "how much does home advantage matter in this league this season, based on evidence so far," which is itself a leakage-safe, evidence-based quantity rather than a fixed assumption (avoids hardcoding an arbitrary home-advantage constant per project instructions on validating rather than assuming weights).

### G. Match context

| Feature | Definition | V1? |
|---|---|---|
| `home_matches_played_before_target`, `away_matches_played_before_target` | count of each team's own past matches (any competition/season, all-time within the ingested dataset) strictly before T | Yes |
| `home_insufficient_history`, `away_insufficient_history` | boolean, `True` if `matches_played_before_target < 3` | Yes — see §5 (promotion policy) |
| `home_strength_score`, `away_strength_score`, `strength_diff` | from §4 (attack/defence methodology) | Yes |
| `competition_id`, `season_id`, `unix` | passed through as context, not "features" in the statistical sense | Yes |

**Not built:** any external ranking (explicitly forbidden), and no attempt to detect literal promotion/relegation — see §5 for why and what's used instead.

### H. Match tempo proxies

`home_attacks_per_match_{w}`, `away_attacks_per_match_{w}`, `home_dang_attacks_per_match_{w}`, `away_dang_attacks_per_match_{w}` (from `stat_*_attacks`/`stat_*_dang_attacks`, ~99.6% populated, same coverage mechanism), `home_pressure_per_match_season`, `away_pressure_per_match_season` (from `stat_home_pressure`/`stat_away_pressure`, 99.3% populated). Shots-for-per-match from §D doubles as a tempo proxy too — not duplicated here, just noted.

**Every one of these is documented, here and in code comments, as a volume/frequency proxy for tempo — not a timing measurement.** No field or feature in this codebase is ever labeled "seconds between shots" or implies match-second-level granularity, because that data does not exist in the source (confirmed absent, per the original audit).

### I. Discipline / supplementary features

| Feature | Decision | Rationale |
|---|---|---|
| `home_fouls_per_match_season`, `away_fouls_per_match_season` | **Built** (coverage-gated) | 0.39% null — no ambiguous-zero problem, essentially fully reliable. |
| `home_yellow_cards_per_match_season`, `away_yellow_cards_per_match_season` | **Built** (coverage-gated) | 0.95% null — same reasoning. |
| `home_red_cards_per_match_season`, `away_red_cards_per_match_season` | **Deferred, NOT built in V1** | 27.1% null dataset-wide. The Phase 2 audit flagged (YELLOW) that null likely means "zero red cards" but explicitly said this was "not independently re-verified" — per this phase's instruction ("do not assume null means zero unless evidence supports it"), that's insufficient evidence to build on. Building this feature would require either verifying the zero-vs-null hypothesis against an independent source or against internal consistency (e.g., checking whether `stat_cards` total ever implies a red card on a null-red-card row) before it can be trusted. |
| `home_offsides_per_match_season`, `away_offsides_per_match_season` | **Deferred, NOT built in V1** | Same reasoning, 19.3% null, same unresolved zero-vs-null ambiguity. |

### J. Odds

**Deferred entirely for a different and more fundamental reason: actual odds prices were never ingested.** Re-checked directly against the schema (Step 1's "verify from the database" instruction) rather than assumed from `DATA_AUDIT.md`: the `fixtures` table has a `has_odds` boolean flag only. Phase 1's ingestion deliberately used `include=stats` (per its own scope), not `include=odds` — so there is no `stat_odds_home`/`odds_draw`/`odds_away`-type column anywhere in `data/processed/matches.db`, and no odds-price raw data was saved under `data/raw/` either (only `include=stats` responses exist there). Building an odds feature group now would require re-ingesting with `include=odds`, which is an API call and explicitly out of scope for this phase ("Do NOT fetch additional data. Do NOT call any API."). This is therefore not a temporal-semantics design question yet (pre-match vs. closing odds, as the task anticipated) — it's a data-availability gap one level earlier than that. **V1 ships without any odds-derived feature.** If odds become a priority, the correct next step is a small, explicitly-scoped Phase 1 extension (re-run `/fixtures/between?include=odds` or a dedicated odds endpoint) with its own leakage-timing analysis, not something to bolt onto this phase.

---

## 3. Labels (not features — stored alongside, never fed back as input)

The feature table also stores the actual outcome of T, for training/evaluation use: `label_home_goals`, `label_away_goals`, `label_result` (`H`/`D`/`A`, derived from the goals). These are explicitly segregated (a `label_` prefix, and a "LABELS — NOT FEATURES" section in `storage.py`) so there is no ambiguity about which columns are legal model inputs. No feature in §2 may read T's own `home_goals`/`away_goals`/any `stat_*` column of T — only these dedicated label columns do, and only for the row's own target fixture.

---

## 4. Attack/defence strength methodology

**Chosen method: empirical-Bayes shrinkage of season-to-date per-match goal rate toward the league's own season-to-date mean, weighted by sample size.**

```
shrunk_attack(team) = (goals_for_sum(team, season, pre-T) + k * league_mean_gf(league, season, pre-T))
                       / (matches_played(team, season, pre-T) + k)
```

with `k = 5` (a team's own record is weighted equally to 5 "average league team" pseudo-matches). `league_mean_gf` is itself computed leakage-safely — all matches in that league+season strictly before T, same as `league_home_advantage_season` in §F. Same formula mirrored for defence using goals-against. `strength_diff = home_strength_score - away_strength_score` (attack-minus-defence combination deferred to the model itself, not baked into a single scalar here — keeping attack and defence as separate scored features preserves information a single combined "rating" would discard).

**Why shrinkage instead of a raw average:** a raw per-match average for a team with 2 games played is dominated by noise (project instructions explicitly require justifying any weighting rather than hardcoding one; `k=5` is not empirically tuned yet — it's a standard, conservative default for this class of problem and is documented as a candidate for backtesting-driven revision, not presented as final).

**Why NOT a full iterative opponent-adjusted rating (e.g., Elo-style) in V1:** per this phase's explicit instruction not to implement one "unless the design document shows it is justified" — it isn't, yet. An iterative system requires its own convergence/stability validation and backtesting infrastructure that doesn't exist until a later phase (model development, per the project's stated 12-step workflow). Shrinkage toward the league mean is the simplest method that already satisfies "league normalization + sample-size awareness" without that additional, currently-unvalidated machinery. This can be revisited in v1.1+ once there's a backtest to compare it against.

Home/away context is handled by keeping `home_goals_for_home_venue`/`away_goals_for_away_venue` (§A) as separate, non-shrunk supplementary features rather than folding venue into the shrinkage formula itself — venue splits have much smaller per-team sample sizes (roughly half of `season`), so shrinking them independently with the same `k` would over-smooth them; they're left as raw venue-restricted averages with their own `_n` and a documented 2-match minimum instead.

---

## 5. Promotion / new-team handling

**No literal promotion/relegation detection is implemented, because the ingested dataset cannot support one.** This project only ingested the 5 top-flight leagues themselves (per project scope) — there is no second-division data to check "was this team in the second tier last season." Treating a team as "newly promoted" by absence from this dataset in the prior season would also wrongly flag any team that was simply relegated-and-came-back, or a team from a league not covered at all (irrelevant here since all 5 leagues are covered, but the principle holds).

**V1 policy: use `matches_played_before_target` as the operative signal instead of a promotion label.** This already captures the real thing that matters for a prediction model — "how much reliable history do we actually have for this team, right now" — regardless of *why* the history is thin (newly promoted, mid-table team early in the season, a team with a data gap, etc.). Combined with the shrinkage formula in §4, a team with 0 matches played this season automatically gets `shrunk_attack = league_mean_gf` (the shrinkage formula naturally reduces to the league average when `matches_played = 0`), which is exactly the sensible fallback the task asked for — with no separate "if newly promoted then X" branch needed. `home_insufficient_history`/`away_insufficient_history` (§G) surfaces this explicitly to the model as a boolean, and `home_matches_played_before_target`/`away_matches_played_before_target` gives the continuous version.

---

## 6. Missing-data policy summary

| Field family | Reliable from | V1 treatment |
|---|---|---|
| Goals, shots, shots-on-target, corners, fouls, yellow cards, attacks, dang_attacks, possession, pressure | Full 2020/21-2025/26 (≤1% null throughout) | Built, per-match coverage-counted, NULL only when a window's `_coverage_n` is genuinely too thin |
| xG / xGOT | Gradual rollout Feb-May 2024, clean from ~2024/25 | Built, coverage-gated per §C — never backfilled, never zeroed |
| Tackles | Reliable from 2022/23 | Not built as a standalone V1 feature (redundant with shots/attacks for tempo purposes, and adds another coverage-gated dimension for modest marginal value) — deferred, noted as available for v1.1 if wanted |
| Goal kicks, throw-ins | Reliable from 2023/24 | Deferred — same reasoning as tackles, plus these are the weakest-justified "attacking strength" signals of the available fields |
| Red cards, offsides | Unresolved null-vs-zero ambiguity | Deferred (§I) until the ambiguity is independently verified |
| Odds | Never ingested (data gap, not a null-handling question) | Deferred (§J) |

For every family that **is** built: missing values remain `NULL` in the feature table (never `0`), the feature is excluded (NULL) for a specific target match only when that match's own window lacks sufficient coverage, there is no fallback-feature substitution across families (e.g. shots never silently substitutes for missing xG), and every coverage-gated feature carries its own `_n`/`_coverage_n` columns as the model's explicit confidence signal.

---

## 7. Storage model (detail in `src/features/storage.py`, summarized here)

A separate SQLite database, `data/processed/features.db`, table `feature_rows`, **not** written into `matches.db` — the raw fixture table stays the untouched, immutable source of truth, exactly as `raw_store.py`'s docstring establishes for the raw JSON layer. `feature_rows` is keyed by `fixture_id` (matching `matches.db`'s own primary key, joinable by that shared ID, not physically foreign-keyed across separate SQLite files). Columns: target metadata (`fixture_id`, `competition_id`, `season_id`, `unix`, `home_id`, `away_id`), every feature from §2 (roughly 70-80 columns given the window multiplication), every `_n`/`_coverage_n` companion column, the `label_*` columns from §3, `feature_version`, and `generated_at`. Raw match statistics are not duplicated into this table beyond what's needed to compute/interpret the features themselves.
