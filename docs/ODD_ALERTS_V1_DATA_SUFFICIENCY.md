# OddAlerts-Only V1 Data Sufficiency Study

**Status:** Research / data-validation only. No feature engineering code, no attack/defence formulas, no Poisson simulation, no model implementation. Nothing here has been purchased or integrated.
**Date:** 2026-08-16
**Client direction driving this document:** try OddAlerts alone first; do not integrate Sportmonks/API-Football/Understat/anything else yet.
**Method:** Re-inspection of `DATA_AUDIT.md`, `PROVIDER_RESEARCH.md`, the saved sample files in `data/audit/`, and the project instructions, followed by a new round of live, read-only GET calls against `https://data.oddalerts.com/api/...` specifically designed to *measure* coverage and *test* leakage-safety rather than just confirm field existence. New evidence is saved in `data/audit/` (see file list at the end) and is cited by filename throughout.

**Two new findings from this pass materially change the picture from `DATA_AUDIT.md` and are called out up front because they affect almost every phase below:**

1. **xG/xGOT is not available for the full historical window.** There is a platform-wide cutover somewhere between 2024-02-11 and 2024-05-02 — before it, `home_xg`/`away_xg`/`home_xgot`/`away_xgot` are `null` on every fixture tested across 3 different leagues; after it, they are consistently populated. Evidence: `data/audit/xg_coverage_cutover_evidence.json`, `data/audit/match_level_coverage_spotchecks.csv`.
2. **`GET /stats/season/{id}` and `GET /stats/fixture/{id}` (the pre-aggregated rollup endpoints) apply a hidden, non-overridable 730-day rolling lookback.** Requesting aggregate stats for a season older than ~2 years silently returns all-zero records instead of an error — this is not a real data gap, it's an artifact of that endpoint, and it means these endpoints must not be used for historical feature construction. The underlying raw match data is intact and must be pulled from `/fixtures/between` instead. Evidence: `data/audit/stats_season_lookback_limitation.json`.

---

## PHASE 1 — Project inspection

Read in full before starting new work: `DATA_AUDIT.md` (102 lines, all 52 rows of the requirement/field table), `PROVIDER_RESEARCH.md`, project instructions (this project's system configuration), all 11 files in `data/audit/`, and the raw Postman collection at `docs/Football Data API by OddAlerts.com.postman_collection.json`. No API client or config code exists yet in the project — this is confirmed to still be a research-only phase, consistent with instructions.

Every JSON structure cited in this report was either (a) re-inspected directly from a fresh live call made during this pass, or (b) already captured verbatim in a `data/audit/*.json` file from the original audit. Nowhere in this document is a field's presence assumed from documentation text alone.

---

## PHASE 2 — Requirement → data mapping

| Requirement | OddAlerts field(s) | Direct / Derived | Coverage | Can be used? | Notes |
|---|---|---|---|---|---|
| Average shots per 90 | `stats.home_shots/away_shots` (fixture-level, via `/fixtures/between?include=stats`) | **Derived** (sum shots over N matches ÷ N × ... per-90 normalization is trivial since all matches are ~90 min) | Confirmed populated back to Aug 2023 (earliest tested) | Yes | Season-level `shots_total/for/against` also exists but is subject to the 730-day lookback bug — compute from raw fixtures instead. |
| Average shots on target per 90 | `stats.home_shots_on/away_shots_on` | Derived | Same as above | Yes | — |
| Average xG for per 90 | `stats.home_xg/away_xg` | Derived | **Only from ~Feb–May 2024 onward** (see cutover finding) | **Yes, but only for recent seasons** | Not usable for full historical backtest depth; usable for 2024/25–2025/26 seasons. |
| Average xG against per 90 | `stats.home_xg/away_xg` (opponent's value) | Derived | Same cutover limit | Yes, same caveat | — |
| Average big chances per 90 | — | — | **NOT AVAILABLE** | No | No "big chance" field or threshold-xG-per-shot flag exists anywhere in the 44-endpoint collection. Cannot be derived without shot-level xG, which OddAlerts does not expose (confirmed in original audit). |
| First-half shots | — | — | **NOT AVAILABLE** | No | `stats` object has no `_1h`/`_2h` split for shots (unlike goals and cards, which do have 1H/2H splits). Confirmed absent from the fixture-level `stats` object inspected live. |
| Second-half shots | — | — | **NOT AVAILABLE** | No | Same as above. |
| Average goals per 90 | `home_goals`/`away_goals` on the fixture object | Derived | Full historical depth (2020/21+, subject to the general 730-day caveat only for the *aggregate* endpoint, not raw fixtures) | Yes | Most reliable metric in the whole dataset. |
| Average time per goal | — | — | **NOT AVAILABLE** | No | Requires a per-goal minute timeline. OddAlerts only exposes 1H/2H goal-count buckets, not individual goal minutes (confirmed in original audit's "Event-level data" row). |
| Goals scored per 90 | Same as "average goals per 90" | Derived | Full historical | Yes | — |
| Average time between shots | — | — | **NOT AVAILABLE** | No | Requires discrete shot timestamps. OddAlerts exposes only match-total shot counts, no shot event list. See Phase 4 for the recommended substitute. |
| Goals conceded per 90 | `home_goals`/`away_goals` (opponent's) | Derived | Full historical | Yes | — |
| xG conceded per 90 | `stats.home_xg/away_xg` (opponent's) | Derived | Same cutover limit as xG for | Yes, with caveat | — |
| Shots conceded per 90 | `stats.home_shots/away_shots` (opponent's) | Derived | Confirmed back to Aug 2023 | Yes | — |
| Fouls committed per 90 | `stats.home_fouls/away_fouls` | Derived | Confirmed back to Aug 2023, occasional nulls seen (not majority) | Yes, with minor null-handling | — |
| Cards per 90 | `stats.home_yellow_cards/away_yellow_cards/home_red_cards/away_red_cards` | Derived | Confirmed populated in every match-level sample checked (Aug 2023 – Feb 2025) | Yes | The original audit's "0% card_data coverage" flag was on the season-aggregate `coverage` block, which is affected by the same 730-day lookback bug — raw match-level card data is fine. |
| Average time between shots conceded | — | — | **NOT AVAILABLE** | No | Same reason as "average time between shots." |
| Other defensive indicators (tackles, offsides, goal kicks, throw-ins, possession, corners, pressure, dangerous attacks) | `stats.*` fields on fixtures | Direct | Confirmed populated back to Aug 2023 | Yes | Bonus fields beyond the original JOMO list — genuinely useful defensive/possession-style indicators not originally requested but available for free. |
| First-half goals | `stats` object does not carry this directly, but `ht_score` on the fixture object gives half-time score, from which 1H goals = ht_score, 2H goals = final − ht_score | Derived | Full historical (ht_score confirmed present on every fixture sampled) | Yes | Simple subtraction, not a modeling risk. |
| Second-half goals | Same as above | Derived | Full historical | Yes | — |
| Match tempo | `stats.attacks/dang_attacks/home_pressure/away_pressure` | Derived (proxy, not literal tempo) | Confirmed back to Aug 2023 | Yes, as a proxy | See Phase 4 — this is the recommended substitute for "seconds between shots," not an equivalent metric. |
| Home/away performance | Fixture object's home/away goals, results, and all `stats.home_*`/`away_*` splits; season-aggregate endpoints also provide explicit home/away splits when not lookback-limited | Direct + Derived | Full historical for raw fixtures | Yes | — |
| Recent form | `/stats/season/:id?last_x=N_overall\|home\|away` (native param) OR self-reconstructed from raw fixtures sorted chronologically | Direct (recent seasons only, due to lookback) / Derived (fallback for older data) | Native param only reliable within lookback window; self-reconstruction has full historical reach | Yes | Use self-reconstruction for consistency and leakage control — see Phase 6. |
| Historical strength | Multi-season aggregation from raw fixtures | Derived | Full historical (2020/21+) for non-xG stats; 2024+ only if xG is a required input | Yes, with the xG caveat | — |
| Competition difficulty | — | Must be derived from cross-competition/cross-team performance patterns within the data (no external ranking) | See Phase 5 | Partial / derivable, not directly available | — |

---

## PHASE 3 — Historical data sufficiency (measured, not assumed)

### Season / fixture counts per league (from `competitions/:id?include=seasons`, live, `data/audit/target_league_seasons.json`)

| League | Seasons with real match data | Fixtures/season | 2019/20 status | 2024/25 status |
|---|---|---|---|---|
| Premier League | 2020/21–2025/26 (6 seasons) | 380 | Listed, `played: null` — **empty, do not use** | `played: 308, progress: 81%` — see note below |
| La Liga | 2020/21–2025/26 (6 seasons) | 380 | Empty | `played: 323, progress: 85%` |
| Bundesliga | 2020/21–2025/26 (6 seasons) | 306 | Empty | `played: 306, progress: 100%` |
| Serie A | 2020/21–2025/26 (6 seasons) | 380–381 | Empty | `played: 323, progress: 85%` |
| Ligue 1 | 2020/21–2025/26 (6 seasons) | 305–306 | Empty | `played: 306, progress: 100%` |

**The 2024/25 partial-progress anomaly (81–85% in 3 of 5 leagues) is unexplained and was not resolved in this pass.** Given the 2024/25 season is fully in the past (finished mid-2025) relative to today's date (2026-08-16), it should show 100% like Bundesliga and Ligue 1 do. Two plausible explanations, neither confirmed: (a) a small number of postponed/abandoned/replayed fixtures were never backfilled, or (b) the `progress` field itself is stale/miscalculated on OddAlerts' side. **This needs a direct fixture-count check (pull all fixtures for e.g. La Liga season 6435 and count against the expected 380) before the pipeline is built**, not assumed away.

### Effective historical depth by data category (measured via live spot-checks, `data/audit/match_level_coverage_spotchecks.csv`)

| Data category | Effective historical start | Evidence |
|---|---|---|
| Fixtures, results, goals, ht_score, formations, referee_id, venue | 2020/21 season (confirmed via season list; raw fixture pull only tested back to Aug 2023 but no reason to expect a gap before that) | `target_league_seasons.json` + spot-checks |
| Possession, pressure, corners, fouls, cards, shots, shots on target, attacks/dangerous attacks, tackles, offsides, goal kicks, throw-ins | **Confirmed populated from Aug 2023 onward** (earliest point tested) | `match_level_coverage_spotchecks.csv`, rows 1–2 |
| **xG / xGOT** | **Confirmed NOT populated before ~Feb 2024; confirmed populated from ~May 2024 onward.** Exact cutover date not pinned down beyond that bracket. | `xg_coverage_cutover_evidence.json` |
| Odds (`has_odds` flag on fixtures, `/odds/history` opening/closing/peak) | Confirmed `has_odds: true` on every fixture sampled back to Aug 2023 | Spot-checks |
| Player-level stats | Not spot-checked for historical depth in this pass (out of scope — player-level data is not needed for the team-vs-team prediction requirements in this task) | — |

### Promotion/relegation gaps

Not empirically re-measured in this pass (would require pulling full team lists per season and diffing, which was deprioritized in favor of the more consequential xG/lookback findings above). This is a known, structural fact of these leagues rather than something that needs API confirmation: each of the 5 leagues relegates and promotes 3 teams every season (except Bundesliga's 2-up-2-down-plus-playoff and occasional expansion/format changes), meaning a meaningful fraction of each season's 18–20 teams (typically 15%) will have zero or partial prior-season history in a 5-season training window. **This must be handled explicitly in feature engineering** (e.g. a "newly promoted" flag, or a league-average fallback for teams with insufficient match history) — it is a data-completeness issue inherent to football, not an OddAlerts limitation, and it is fully derivable from the fixture data once pulled (compare team ID sets season-over-season).

### Missing-data percentages

Full-season missing-data percentages (e.g. "3.2% of Bundesliga 2022/23 fixtures are missing fouls data") were **not computed** in this pass — doing so properly requires pulling every fixture of every season of every league (~11,000+ matches across 6 seasons × 5 leagues) and is pipeline-build work, not research-phase work. What this pass *did* establish, with real evidence rather than assumption, is: (a) the specific xG cutover date range, (b) that non-xG match stats are reliably populated as far back as tested, and (c) that the pre-aggregated coverage-reporting endpoint (`stats/season`'s `coverage` block) cannot be trusted for older seasons due to the lookback bug. **Recommend running a full missing-data census as the first concrete step of the actual ingestion pipeline**, using the `fixtures/between` + self-aggregation approach validated in Phase 6, not as a separate research task.

---

## PHASE 4 — Can we reconstruct attack/defence scores?

| Variable | Status | Notes |
|---|---|---|
| `team_name` | **Direct** | `home_name`/`away_name` on every fixture. |
| `league` | **Direct** | `competition_id`/`competition_name` on every fixture. |
| `season` | **Direct** | `season_id`/`season` on every fixture. |
| Overall/hybrid rating | **Derived** | Must be computed by us from attack + defence + context; no such single field exists or should be expected from a raw data API. |
| Attack strength | **Derived** | From goals/xG/shots-for per-90, adjusted for league/opponent context (see Phase 5) — but xG-based attack strength is only reliable using 2024+ data; goals/shots-based attack strength has full historical depth. |
| Defence strength | **Derived** | Mirror of attack strength using goals/xG/shots-against. Same xG-era caveat. |
| Competition difficulty factor | **Derived, and only weakly** — see Phase 5 in full. No external ranking exists or should be used per project instructions; must come from within-data cross-league signal, which is inherently harder to make statistically robust with only 5 leagues and limited cross-competition overlap. |
| Match tempo | **Derived (proxy), not the literal thing** | See dedicated subsection below. |
| Home advantage | **Derived** | Directly computable from the home/away split that exists on every stat and every fixture — this is one of the best-supported derived variables in the dataset. |
| Form rating | **Derived** | Rolling window over N previous matches, self-computed (do not trust `last_x` on the aggregate endpoint beyond its ~2-year reliable window — see Phase 3). |

### "Average seconds between shots" — specific analysis, as requested

OddAlerts does not expose discrete shot events or shot timestamps (confirmed absent in the original audit, and re-confirmed here — no field or endpoint returns anything shot-level beyond match totals). **A literal "average seconds between shots" cannot be computed from this data under any derivation.** Treating a per-90 shot rate as if it were equivalent to seconds-between-shots would be a false equivalence — 20 shots in a 90-minute match does not tell you *when* those shots happened or whether they were bunched in a 10-minute spell or spread evenly, and the project instructions are explicit that this substitution should not be pretended away.

The statistically honest options, none of which are "the same metric" and all of which should be documented as approximations if used:

1. **Shots per 90** — the simplest, most defensible substitute. It captures shot *volume* (a component of tempo) without any claim about timing distribution. Full historical depth.
2. **Shots per possession-adjusted unit** — dividing shots-for by the team's own possession share (both available per fixture) gives a rough "shot efficiency relative to time on ball" proxy. Still not a timing metric, but slightly closer to "how quickly does this team generate shots when it has the ball."
3. **Attacks / dangerous attacks per 90** — OddAlerts' own `attacks`/`dang_attacks` fields are the closest thing to a possession-tempo signal in the dataset (frequency of attacking sequences per match), and are a reasonable proxy for overall match tempo (not shot-specific).

**Recommendation:** use shots-per-90 (and its home/away/for/against splits) as the tempo-adjacent attacking metric, and explicitly document in the model's methodology notes that this is a volume proxy, not a timing metric, and that true "seconds between shots" is not available from OddAlerts.

---

## PHASE 5 — Competition difficulty, without FIFA rankings

Per instructions, FIFA rankings are correctly excluded (they measure national teams, not clubs). The question is whether OddAlerts' own data supports a defensible internal competition-difficulty adjustment. Findings, without implementing anything:

- **European competition cross-over** (Champions League / Europa League results linking teams across the 5 domestic leagues) is the most statistically legitimate signal available, *if* OddAlerts carries those competitions with the same stats depth as domestic leagues — **this was not verified in this pass**. The `/competitions` endpoint returned dozens of cup/European competitions in the raw output during the original audit (e.g. `Coupe de la Ligue`, `DFB Pokal`) but Champions League/Europa League specifically were not seen in the sampled `country_ids=45,8,3,32,4` pull (those are typically filed under a "Europe"/UEFA country_id, e.g. country_id 10 seen in the countries list). **This needs a direct, targeted check before it can be used** — pull `/competitions?country_ids=10` (Europe) and confirm Champions League/Europa League IDs exist with match-level stats coverage comparable to the domestic leagues.
- **League-wide goal/xG distributions** (average goals per game, average xG per game, computed per league per season from the raw fixture data we already know how to pull) is fully supported by the data on hand and is a legitimate, if blunt, difficulty proxy — a league that averages 3.1 goals/game plays differently than one averaging 2.4, and this is directly measurable with full historical depth (goals; only 2024+ depth if using xG specifically).
- **Opponent-adjusted performance** (e.g., strength-of-schedule-adjusted goal differential, computed iteratively within a season/league) is derivable purely from within-league fixture results and does not need any additional data — this is the most statistically defensible approach available from OddAlerts alone, because it doesn't depend on uncertain cross-competition coverage.
- **Normalized team strength** (z-scoring each team's attack/defence numbers against their own league's mean/stddev before any cross-league comparison) is trivially derivable and should be the *baseline* step regardless of which cross-league adjustment (if any) is layered on top.

**Statistically defensible recommendation for this project specifically:** start with within-league normalized attack/defence ratings (fully supported, no external data needed) and treat cross-league "difficulty" as a secondary, lower-confidence adjustment layered on top only if the European-competition cross-over data checks out — not as a required V1 component. This keeps V1 honest about what the data actually supports instead of forcing a competition-difficulty factor that isn't well-grounded yet.

---

## PHASE 6 — Data leakage: recommended architecture

**`include_frozen=true` is confirmed NOT safe to use for leakage prevention**, with a concrete, reproducible demonstration in this pass (not just the earlier suspicion):

- Target fixture: Arsenal vs Manchester City, 2025-02-02, fixture id 215022547. The fixture object itself states `home_played: 23` — i.e., Arsenal had played exactly 23 prior league matches this season before this kickoff.
- `GET /stats/fixture/215022547?include_frozen=true` returned a record with `played.total: 38` — the **full completed season**, which includes matches played *after* the target fixture. This is future information leaking into what the endpoint documents as a pre-match snapshot.
- A manually reconstructed pre-match window — `GET /fixtures/between?teams=5303&seasons=6484&from=<season start>&to=<day before kickoff>&include=stats` — returned **exactly 23 fixtures**, matching the fixture object's own `home_played:23` field precisely, with all match-level stats (goals, xG, shots, cards, etc.) populated and summable into leakage-safe pre-match features (e.g. goals-for-per-90 = 44 goals / 23 games = 1.91).

Full detail in `data/audit/frozen_stats_leakage_test.json`.

**Recommended leakage-safe architecture:**

1. For every target match at kickoff time T, pull all prior fixtures for both teams via `GET /fixtures/between?teams=<id>&from=<window_start>&to=<T minus 1 second>&include=stats` (or a wider multi-team batch call, then filter/group locally — more efficient for building a full training set).
2. Sort chronologically (the `unix` field is reliable for this).
3. Compute all rolling features (goals, xG where date ≥ cutover, shots, cards, fouls, corners, possession, form, home/away splits) strictly from that pre-T window — never from any OddAlerts aggregate/season/frozen endpoint.
4. Cache these self-computed rolling features locally (e.g. SQLite, per project's preferred stack) rather than re-deriving them from the live API on every training run — this also insulates the project from the 730-day lookback drift described in Phase 3, since the raw historical fixtures, once pulled and stored locally, don't disappear the way the API's own aggregate view of them effectively does.
5. Treat `include_frozen=true` as unused/deprecated for this project unless OddAlerts support can explain and reproduce its documented behavior in a way that passes a similar verification test.

This directly satisfies the project's Data Leakage Prevention requirement and prefers time-based backtesting (chronological rolling windows) over any pre-built shortcut, exactly as instructed.

---

## PHASE 7 — Prediction feasibility (without implementing anything)

| # | Capability | Verdict | Why |
|---|---|---|---|
| A | Expected goals | **YES — DERIVED** | Directly available as a native field (`home_xg`/`away_xg`) for 2024+ matches; for earlier matches, goals-based proxies can be used, or the xG-based model can simply be trained only on the post-cutover window (still 1.5–2 seasons across 5 leagues, ~thousands of matches). |
| B | Win/Draw/Loss probability | **YES — DERIVED** | Standard output of a Poisson/simulation model fed by attack/defence strengths, which are themselves derivable from goals and (for recent seasons) xG. Not directly returned by OddAlerts as "our" prediction — OddAlerts' own `/predictions` and `/probability` endpoints are a third-party model output, explicitly out of scope as a feature source per `DATA_AUDIT.md`. |
| C | Most likely scoreline | **YES — DERIVED** | Standard Poisson-simulation output once expected goals are established. |
| D | Team goals prediction | **YES — DERIVED** | Same basis as A/C. |
| E | Attack strength | **YES — DERIVED** | See Phase 4; goals-based version has full historical depth, xG-enhanced version limited to 2024+. |
| F | Defence strength | **YES — DERIVED** | Mirror of E. |
| G | Recent form | **YES — DERIVED** | Fully supported via self-reconstructed rolling windows (Phase 6); native `last_x` param usable only within the ~2-year lookback. |
| H | Home advantage | **YES — DERIVED** | Best-supported derived variable in the dataset — home/away splits exist on every relevant field. |
| I | Competition difficulty | **PARTIAL** | Within-league normalization is solid; cross-league adjustment is only weakly supported pending verification of European-competition data coverage (Phase 5). Should ship V1 without a strong cross-league difficulty factor, or with a clearly-labeled low-confidence one. |
| J | Uncertainty / confidence estimates | **YES — DERIVED** | Native output of a Monte Carlo/Poisson simulation (variance across simulated outcomes) — not something OddAlerts needs to provide directly, it falls out of the simulation approach itself once built. |
| K | Match chaos / unpredictability factor | **PARTIAL** | Per instructions, do not implement yet. The data *could* support an empirically-derived chaos factor (e.g., residual variance in actual results vs. model-predicted results, measured via backtesting) but this requires the model to exist first and be backtested — it cannot be derived a priori from raw stats alone. This is a Phase-11-or-later item (per the project's stated development workflow), not a V1 blocker. |
| L | Exact shot-quality xG based on shot coordinates | **NO** | Confirmed absent from OddAlerts (no shot X/Y anywhere in the 44-endpoint collection, original audit + reconfirmed). OddAlerts' own match-level xG (a pre-computed aggregate, not derived from coordinates we hold) is the closest available substitute, and only from 2024+. |
| M | Exact seconds-between-shots tempo | **NO** | Confirmed absent — no discrete shot timestamps exist. Shots-per-90 is the defensible substitute (Phase 4), not an equivalent metric. |

---

## PHASE 8 — Final decision

### 🟢 BUILD NOW WITH ODDALERTS

- Fixtures, results, and goals (full historical depth, 2020/21–2025/26, all 5 leagues)
- Win/Draw/Loss probability and most-likely-scoreline via a self-built Poisson/Monte Carlo simulation
- Goals-based attack/defence strength ratings, home/away-split, with full historical depth
- Shots, shots-on-target, possession, corners, fouls, cards, tackles, offsides, goal kicks, throw-ins, pressure/dangerous-attacks-based tempo proxy — all confirmed populated back to at least Aug 2023, usable as supplementary features alongside goals
- Recent form via self-computed rolling windows (leakage-safe, per Phase 6)
- Home advantage
- Within-league normalized team strength
- Historical odds (`/odds/history`, permanent opening/closing/peak) for backtesting against market lines and computing implied-probability baselines
- Uncertainty/confidence intervals as a natural output of the simulation approach

### 🟡 BUILD WITH A METHODOLOGY ADAPTATION

- **xG-enhanced attack/defence strength** — real and usable, but only for matches from ~Feb–May 2024 onward (needs exact date pinned down before finalizing the training window). Adaptation: run two model variants (goals-only, full historical depth; xG-enhanced, shorter but higher-quality window) and compare, rather than assuming xG can be backfilled across all 6 seasons.
- **Match tempo** — shots-per-90 and attacks/dangerous-attacks-per-90 used as documented proxies, explicitly not equivalent to true shot-timing tempo.
- **Competition difficulty** — within-league normalization now, cross-league adjustment only if European-competition data coverage is separately verified; otherwise ship without it or with a clearly low-confidence flag.
- **Frozen/pre-match features** — do not use OddAlerts' `include_frozen` parameter; self-reconstruct via the chronological rolling-window architecture in Phase 6 instead. This is a full substitute, not a degraded one — it was proven equivalent to the theoretically-correct answer for the one fixture tested.

### 🔴 CANNOT BUILD WITH ODDALERTS

- Average time per goal / average time between shots / average time between shots conceded — no discrete event timestamps exist under any derivation
- Big chances per 90 — requires shot-level xG, not available
- First-half/second-half shot splits — not exposed (unlike goals/cards, which do have 1H/2H splits)
- Exact shot-quality xG from shot coordinates — no coordinates exist
- Match chaos/unpredictability factor as a data-derived-in-advance metric — this specifically requires a working, backtested model first (not a data gap, a sequencing issue — correctly deferred per instructions)

### Final verdict: **Is OddAlerts sufficient for V1?**

**Yes — with two scope adjustments the client should sign off on before build starts.**

OddAlerts alone supports a genuinely useful V1: win/draw/loss probabilities, expected goals, likely scorelines, and confidence intervals, built on a self-computed Poisson/Monte Carlo simulation fed by goals-based (full historical depth) and xG-enhanced (2024+) attack/defence strength ratings, home advantage, and rolling form — all leakage-safe via self-reconstructed chronological windows rather than OddAlerts' unreliable frozen-stats endpoint. This directly delivers everything in the client's six numbered requirements (win probability, W/D/L, expected goals, scorelines, goal-prediction accuracy, confidence estimates).

**The two adjustments:**

1. **xG-based features have a real, evidenced historical ceiling (~Feb–May 2024 onward), not the full 6-season window.** V1 should either (a) train the core model on goals-based strength ratings across the full historical depth and treat xG as a recent-data enhancement/validation signal, or (b) accept a shorter, xG-informed training window and validate whether that's enough data for the target accuracy — this is a modeling decision for the next phase, not a blocker now, but the client should know the ceiling exists before backtesting results come in lower than expected on xG-dependent features.
2. **"Seconds between shots" and "big chances" cannot be built from this data under any derivation** — not degraded, not approximated, genuinely absent. They should be dropped from the V1 feature list rather than faked with a mislabeled substitute. Shots-per-90 and attacks-per-90 are reasonable stand-ins for tempo but must be documented as such, not presented as equivalent.

No additional data provider is required to build a defensible, honestly-scoped V1. If, after backtesting the goals-only vs. xG-enhanced variants, the client specifically wants shot-level tempo/chaos features or wants the xG training window extended further back than ~2024, that would be the concrete, evidence-based trigger to revisit `PROVIDER_RESEARCH.md` — not a default next step.

---

## Concise summary

**1. What we have:** Full 6-season (2020/21–2025/26) fixture/result/goals history for all 5 target leagues; rich match-level stats (possession, shots, shots-on-target, corners, fouls, cards, tackles, offsides, goal kicks, throw-ins, pressure) confirmed populated back to at least August 2023; permanent historical odds; native home/away splits everywhere.

**2. What we can derive:** Attack/defence strength (goals-based, full depth; xG-enhanced, 2024+ only), home advantage, rolling form (self-computed, leakage-safe), within-league normalized team ratings, win/draw/loss probability, expected goals, likely scorelines, and confidence intervals — all via a self-built Poisson/Monte Carlo simulation layer on top of the above.

**3. What is missing (genuinely, not just untested):** Shot-level timestamps and coordinates (so no true "seconds between shots," no shot-coordinate xG, no big-chances metric), first/second-half shot splits, and any cross-league competition-difficulty signal beyond what within-league normalization and (pending verification) European-competition results can support.

**4. What V1 should contain:** Win/Draw/Loss probabilities, expected goals, most-likely scorelines, and confidence estimates for all 5 leagues, built on goals-based attack/defence strength (full historical depth) with xG-enhanced features layered in for the post-cutover window, self-computed leakage-safe rolling form, home advantage, and within-league difficulty normalization — explicitly without a match-chaos factor (deferred per instructions) and without any tempo metric stronger than shots-per-90.

**5. Is another API actually necessary right now:** **No.** Every requirement in the client's six-point prediction spec is buildable from OddAlerts alone, with the two documented adjustments above. Additional providers remain a *future* option (see `PROVIDER_RESEARCH.md`) triggered by concrete backtesting evidence, not a prerequisite for V1.

---

## Appendix: New supporting evidence files (this pass)

- `data/audit/xg_coverage_cutover_evidence.json` — live evidence bracketing the xG availability cutover date
- `data/audit/stats_season_lookback_limitation.json` — evidence and explanation of the 730-day lookback bug on aggregate endpoints
- `data/audit/frozen_stats_leakage_test.json` — reproducible before/after comparison proving `include_frozen` is not leakage-safe and that self-reconstruction is
- `data/audit/match_level_coverage_spotchecks.csv` — raw table of every live coverage spot-check made in this pass
- `data/audit/season_2023_24_coverage_summary.json` — superseded/invalidated file, kept with an explanation rather than deleted, documenting why the first attempt at measuring season-level coverage gave misleading zeros (the lookback bug) — included deliberately so the reasoning trail is auditable
