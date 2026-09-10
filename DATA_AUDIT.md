# OddAlerts API — Data Capability Audit

**Status:** Read-only audit. No model, feature engineering, or simulation logic has been implemented.
**Date:** 2026-08-16
**Method:** (1) Full inspection of the client's exported Postman collection (`docs/Football Data API by OddAlerts.com.postman_collection.json`, 44 endpoints, official saved examples). (2) Live, read-only GET calls against `https://data.oddalerts.com/api/...` using the credential stored in the project's `.env` (never printed, never committed — see Security section). All findings below are traceable to either a documented example response in the collection or a live call made during this audit. Nothing in this document is invented.

**Base URL:** `https://data.oddalerts.com/api`
**Auth:** query parameter `api_token=<token>` on every request.
**Pagination:** every list endpoint returns `{"info": {...page/count/pages...}, "data": [...]}`. `info` also carries `next_page_url` where relevant.
**Target leagues confirmed present**, with official OddAlerts competition IDs (live call, `GET /competitions?country_ids=45,8,3,32,4`):

| League | Competition ID | Country ID | Current Season ID |
|---|---|---|---|
| Premier League | 423 | 45 (England) | 2263973 |
| La Liga | 419 | 8 (Spain) | 2244302 |
| Bundesliga | 477 | 3 (Germany) | 2305071 |
| Serie A | 499 | 32 (Italy) | 2224807 |
| Ligue 1 | 200 | 4 (France) | 2254143 |

---

## 1. Requirement Capability Table

| Requirement | Available? | Endpoint | Actual Field | Evidence | Notes |
|---|---|---|---|---|---|
| Fixtures (single/multiple) | CONFIRMED | `GET /fixtures/:id`, `/fixtures/multiple?ids=` | `id, home_name, away_name, status, unix, date, ko_human, home_goals, away_goals, corners` (+ richer fields via `include=`) | Doc example (Multiple Fixtures) | Base fixture object is thin by default; richer data requires `include=`. |
| Upcoming fixtures | CONFIRMED | `GET /fixtures/upcoming` | Same as above + `season_id, competition_id, home_id, away_id, home_formation, away_formation, home_position, away_position` | Doc example | Next 7 days, sorted by kickoff. Filterable by `teams=`, `seasons=`, `competitions=` (max 25 IDs each). |
| Historical matches / Results | CONFIRMED | `GET /fixtures/between?from=&to=` | Same fixture object, `status: "FT"` for completed | **Live call**, `competitions=423, from=1738368000, to=1738972800` → 10 real EPL fixtures (Feb 2025), e.g. Arsenal 5–1 Man City | `from`/`to` are Unix seconds. Confirmed working with `competitions=` filter for a single target league. |
| Season / historical coverage per league | CONFIRMED | `GET /competitions/:id?include=seasons` | `season_id, season_name, played, progress` | **Live call** for all 5 target leagues (saved to `data/audit/target_league_seasons.json`) | All 5 leagues: seasons **2020/21 through 2025/26** have `played` fixture data (100% progress, except 2024/25 which shows 81–85% progress in 3 of the 5 leagues — flagged below as needing investigation). A `2019/2020` season row exists per league but `played: null` (listed, **no data**). `2026/2027` exists as an empty upcoming-season shell. |
| Team statistics | CONFIRMED | `GET /stats/season/:season_id`, `/stats/fixture/:fixture_id` | See full field dump in `data/audit/stats_team_by_fixture_example.json` | Doc example is real EPL 2025/26 data (season_id 667780 = confirmed current PL season) | Extremely rich: W/D/L, points, goals (total/for/against, split home/away, 1H/2H, "over X" buckets), BTTS, scored/conceded first, fouls, cards (yellow/red, split for/against), corners (+over buckets), tackles, offsides, goal kicks, throw-ins, **xg_total/xg_for/xg_against**, **shots_total/for/against**, **shots_on_total/for/against**, "most_X" league-leader flags, and a `coverage` block reporting % of underlying fixtures that actually have xg/card/corner/shots/tackle/offside/goal-kick/throw-in data. |
| Player statistics | CONFIRMED | `GET /players/search`, `/players/:id`, `/players/rank`, `/players/fixture/:id`, `/players/season/:id`, `/players/competition/:id`, `/players/meta` | Full stat whitelist in `data/audit/players_meta_live.json` | **Live call** to `/players/meta` | Marked "Beta" in the collection. Stats include apps, minutes, goals, assists, **shots_total/on_target/off_target**, tackles, fouls (+drawn), cards, passes, key passes, crosses, dribbles, duels, interceptions, blocks, GK saves/cleansheets, plus rolling-form buckets (`last_5/10/15/20/all`) and threshold "over %" fields. **No player-level xG or xA field exists** (xG is team/match-level only — see below). |
| Goals | CONFIRMED | `/fixtures/*`, `/stats/*` | `home_goals, away_goals`, extensive goal-timing buckets (1H/2H, over-X) | Doc + live | No per-goal minute/scorer event list found (see Event-level row). |
| xG / xGA | CONFIRMED (match & season level only) | `/fixtures/between?include=stats` → `stats.home_xg/away_xg/home_xgot/away_xgot`; `/stats/season|fixture` → `xg_total/xg_for/xg_against` | **Live call**, Arsenal v Man City (Feb 2025): `home_xg: 1.2704, away_xg: 0.9779, home_xgot: 3.2507, away_xgot: 0.9663` | Confirmed populated (non-null) for target-league matches. `xgot` = xG on target, a field not explicitly requested in the audit but discovered — potentially useful. `coverage.xg` in the stats endpoint reports what % of a team's sampled fixtures actually have xG on file (was 100% in the sample checked, but must be verified per-league/per-season, not assumed universal). | Not available at player level. |
| Shots | CONFIRMED | `/fixtures/*` `stats.shots/home_shots/away_shots`; `/stats/*` `shots_total/for/against`; `/players/*` `shots_total` | Live call (fixture-level), doc example (team/player aggregate level) | Present at match, season-aggregate, and player-aggregate level. |
| Shots on target | CONFIRMED | `stats.shots_on/home_shots_on/away_shots_on`; `shots_on_total/for/against`; player `shots_on_target` | Live + doc | — |
| Shot X/Y coordinates | **NOT AVAILABLE** | — | — | Not present in any fixture/stats response field, and not listed in the documented `include=` options for `/fixtures` (`probability, stats, correctScores, odds, odds.live, h2h`). No shot-map / coordinate endpoint exists anywhere in the 44-endpoint collection. | Confirmed absent, not just untested. |
| Possession | CONFIRMED | `stats.home_possession/away_possession` (fixture-level only) | Live call, e.g. `home_possession: 46, away_possession: 54` | Null in low-tier/lower-popularity fixtures (seen in doc example for a non-target league); populated for the EPL sample tested. Not present in the season-aggregate `/stats/season` object — only at individual-fixture level via `include=stats`. |
| Corners | CONFIRMED | `stats.corners/home_corners/away_corners` (fixture); `corners_total/for/against` (+over buckets) (season) | Live + doc | — |
| Fouls | CONFIRMED | `stats.home_fouls/away_fouls` (fixture); `fouls_total/won/committed` (season) | Live + doc | Fixture-level fouls null in some lower-tier sample data; populated in EPL sample tested. |
| Cards | CONFIRMED | `stats.cards/home_yellow_cards/away_yellow_cards/home_red_cards/away_red_cards` (fixture); `yellow_cards_total/for/opponent`, `red_cards_total/for/opponent`, `cards_over` buckets (season) | Live + doc | `coverage.card_data` was 0% in the one doc example inspected (Arsenal, mixed real+placeholder sample) — **coverage must be checked per league/season before relying on cards as a feature**, do not assume universal availability. |
| Lineups | **UNKNOWN / NOT CONFIRMED** | — | `home_formation`/`away_formation` strings are present on fixtures (e.g. `"4-3-3"`) | Live call | Formation string only — no starting XI, no player-by-position lineup, no substitutions list found in the collection. If lineups exist they are not exposed via any documented endpoint. |
| Historical form | CONFIRMED | `/stats/season/:id?last_x=N_overall\|home\|away` | `last_x` param confirmed in doc description and `info.last_x` field | Doc | Supports 1–25 games, scoped to home/away/overall, `all_comps` toggle for cross-competition form. |
| Pre-match statistics | CONFIRMED | `/fixtures/*?include=stats` on upcoming fixtures; `/stats/*?include_frozen=true` | — | Doc + live | See "Frozen historical statistics" row for an important caveat found during live testing. |
| Frozen historical statistics (pre-kickoff snapshot) | **PARTIAL — needs further verification before relying on it for leakage prevention** | `GET /stats/fixture/:fixture_id?include_frozen=true`, `GET /stats/season/:season_id?include_frozen=true` | `info.include_frozen: true` | **Live call** on a finished EPL fixture (`/stats/fixture/215022547?include_frozen=true`) | Documentation states this returns "stats as they were before kick-off in that game" with `fixture_id` populated on the record. In the live test, the returned Arsenal record showed `fixture_id: null` and `played.total: 38` — i.e. **full-season end totals**, not a stats-as-of-matchday-24 snapshot. Two *other* fixture_id values did appear elsewhere in the same response (unexplained — possibly frozen snapshots for different teams/fixtures bundled in). **This must be treated as unverified pending a focused backtest before being trusted for leakage-safe historical features.** Do not build the leakage-prevention pipeline on the assumption this parameter behaves exactly as documented — verify empirically per project's Data Leakage Prevention requirement. |
| Historical odds / probabilities | CONFIRMED | `GET /odds/history/:id` (permanent), `GET /odds/movement/:id` (rolling 21-day window), `GET /probability/:market_id` | `opening, closing, peak` per bookmaker per market (history); `odds, unix, datetime` per movement tick | Doc + live | `/odds/history` is described as the **permanent** record (opening/closing/peak) — correct source for backtesting. `/odds/movement` is explicitly a **rolling 21-day retention window** (`info.retention.retention_days: 21`, confirmed live) — not suitable for deep historical backtesting, only recent/live odds-drift analysis. Odds timestamps are **second-precision** (`unix` epoch seconds + `"datetime": "2026-02-01 07:10:56"`). |
| Event-level data (goal minute, card minute, sub minute, etc.) | **NOT AVAILABLE (as discrete events)** | — | A `coverage.goal_timings` flag exists in `/stats/*` responses, and goal buckets are split 1H/2H | Doc | This confirms the *underlying* data pipeline has goal-timing information, but no endpoint in the 44-endpoint collection exposes a per-event list (no `/fixtures/:id/events`, no minute-by-minute goal/card log). Only aggregated 1H/2H splits are exposed. Treat as NOT AVAILABLE for event-level feature engineering; re-open with OddAlerts support if finer granularity is later required. |
| Event timestamp precision | N/A (no discrete events exposed) | — | Odds and bet-tracking timestamps are second-precision Unix epoch (`unix`) + human `datetime` strings | Live | Fixture kickoff time (`unix`) is also second precision, though naturally minute-granular in practice (kickoffs land on the minute). |
| Referees | CONFIRMED | `/referees`, `/referees/:id`, `/referees/:id/stats`, `/referees/:id/seasons`, `/referees/:id/fixtures`, `/referees/upcoming` | `referee_id` on fixtures; card stats broken down by season | Doc (descriptions only, no saved examples — untested live) | Marked here as CONFIRMED-by-documentation; recommend a live smoke test before depending on it, since no example response exists in the collection. |
| Value Bets / Probability model output | CONFIRMED (third-party model, not raw data) | `/value/:type`, `/probability/:id`, `/correctScores`, `/predictions/generate/:id` | `home_win_percentage, draw_percentage, away_win_percentage, scorelines{...}` (Predictions); `scores{"1-0":"12.57",...}` (Correct Scores, OA's own probability, not actual results) | Doc examples | **Important distinction for methodology:** these endpoints return OddAlerts' own proprietary Monte-Carlo/probability model output, not raw underlying data. Per project instructions to build our own validated Poisson simulation rather than depend on a third-party black-box, these should be treated as a **benchmarking/comparison source only**, not a feature-engineering input, and not something our model's predictions should be derived from. |
| Bookmakers / Markets metadata | CONFIRMED | `/bookmakers`, `/odds/markets`, `/probability/markets` | Bookmaker IDs 1–6 documented: Pinnacle, Bet365, 1xBet, WilliamHill, Betfair Exchange, Kambi Group | Doc | Needed to interpret `bookmaker_id` on odds records. |
| Teams metadata | CONFIRMED (by documentation) | `/teams/all`, `/teams/find/:id`, `/teams/country/:id` | — | Doc (no saved example; untested live) | No example response in collection — recommend a live smoke test. |
| Countries / Competitions metadata | CONFIRMED | `/countries`, `/competitions`, `/competitions/:id`, `/competitions/search` | `id, name, code, slug` (countries); `id, name, country, country_id, type, current_season` (competitions) | **Live calls**, saved to `data/audit/target_league_ids.json` and `target_league_seasons.json` | Used to derive the 5 target league IDs above. |

---

## 2. Rate Limits & Pagination

- All list endpoints paginate via `{info: {page, per_page, count/total, pages, next_page_url}, data: [...]}`.
- No explicit numeric **rate limit** (requests/minute) is documented anywhere in the collection description, endpoint descriptions, or response headers captured. **This is UNKNOWN** — recommend asking OddAlerts support directly, or empirically probing with backoff once we're ready to build the ingestion pipeline (out of scope for this audit).
- `/odds/dropping` is explicitly capped at the top 5,000 records.
- `/odds/history` multi-ID lookups are capped at 50 fixture IDs per call.
- Most filter parameters (`teams=`, `seasons=`, `competitions=`, `fixtures=`) are capped at 25 IDs per call (some endpoints cap at 50 or 100 — see individual endpoint descriptions in the collection).

## 3. Data Retention Caveats (important for backtesting design)

- **Odds movement** (`/odds/movement`) is a rolling **21-day** window only — not usable for deep historical backtests.
- **Odds history** (`/odds/history`) is described as the permanent opening/closing/peak record — this is the correct endpoint for historical odds-based features.
- **Fixture/season data**: confirmed present back to 2020/21 across all 5 target leagues; a 2019/20 season row exists but is empty (`played: null`). Effective usable historical window is **~6 completed seasons (2020/21–2025/26)** per league, pending investigation of the anomalous partial `progress` (81–85%) seen on three leagues' 2024/25 season.

## 4. Security

- The API key was read once from the project's `.env` file to perform live calls and was never printed to chat, logs, or any file in this repository. All URLs and saved sample files in `data/audit/` have been checked and contain **no token values** (the exported Postman collection's `auth` block itself only contains the placeholder `{{api_key}}` / `YOUR_API_TOKEN`, not a real credential).
- `.env` is not committed to any file tracked here beyond its existing location; no code has been written yet that reads it (that will follow the project's standard config-loading approach when ingestion is built).

## 5. Open Items / Follow-ups Before Building Anything

1. **Frozen/pre-match stats behavior is unverified** — the one live test did not match the documented "stats as of kickoff" behavior. Needs a focused test (pick several fixtures across a season, compare `include_frozen=true` output against known pre-match state) before this is trusted as our leakage-prevention mechanism.
2. **`coverage.card_data` and other coverage sub-fields should be checked per target league/season**, not assumed uniformly high — the one sample checked showed 0% card-data coverage despite cards being populated elsewhere in the same response.
3. **Referees and Teams endpoints have no saved example responses** in the collection — recommend a quick live smoke test before depending on them.
4. **No documented rate limit** — needs clarification from OddAlerts or empirical testing during pipeline design.
5. **Event-level (goal/card minute) data is not exposed by any endpoint** — if the project later needs this, it is a genuine gap in the current subscription, not something to work around by guessing.

## 6. Explicitly Out of Scope for This Audit (per instructions)

No xG model, attack/defence strength calculation, Poisson simulation, feature engineering, or ML training has been implemented. This document only establishes what data OddAlerts' API can supply.

---

## Appendix: Sample Responses

Redacted, real sample responses (no API key present in any file) are saved in `data/audit/`:

- `target_league_ids.json` — the 5 target leagues with confirmed competition IDs
- `target_league_seasons.json` — season coverage per target league
- `fixtures_between_epl_sample_note.json` — real EPL fixture with full stats block (incl. xG)
- `stats_team_by_fixture_example.json` — full team-stats field dump (season-aggregate)
- `fixtures_live_example.json` — live fixture with stats/probability/odds nested
- `predictions_generate_simulation_example.json` — OddAlerts' own Monte Carlo output (benchmarking only)
- `correct_scores_example.json` — OddAlerts' own correct-score probability model output
- `trends_example_truncated.json` — trends endpoint sample (2 records)
- `odds_history_example.json` — permanent odds history sample
- `odds_movement_retention.json` — confirms 21-day rolling retention window
- `players_meta_live.json` — full whitelist of player-level stats