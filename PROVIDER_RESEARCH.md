# Complementary Data Provider Feasibility Study

**Status:** Research only. No provider has been purchased, no code written, no changes made to the existing prediction system.
**Date:** 2026-08-16
**Method:** Official documentation, pricing pages, and Terms of Service for each provider were read directly (fetched and quoted below). Where a provider's own docs were not machine-readable (client-rendered SPA docs blocked automated fetching), that is stated explicitly and the specific claim is flagged as lower-confidence rather than presented as fact.

## What we're trying to fill

Per `DATA_AUDIT.md`, OddAlerts is strong on fixtures, historical results, aggregate team/player stats, match-level xG/xGOT, shots, shots on target, possession, corners, fouls, cards, and historical odds. It does **not** provide: shot X/Y coordinates, discrete shot events, goal/card/substitution event streams with timestamps, or reliable event-level timestamps. That gap is the only thing this research is scoring providers against — general fixture/stats/odds overlap with OddAlerts does not count as "complementary" per your instruction.

---

## Provider-by-provider findings

### 1. Sportmonks

**What OddAlerts is missing that Sportmonks provides:** a discrete match-event timeline — goals, cards, substitutions, VAR decisions, each tied to `player_id`, `minute`/`extra_minute`, and a `type_id`/`sub_type_id` (e.g. left-foot shot, header, penalty) — plus confirmed starting lineups/formations and a second, independently-computed xG source (13 metrics: xG, xGoT, npxG, xPTS, xG split by open play/set play/corner/free kick/penalty, xG Prevented for keepers) at both team and player level.

**Shot X/Y coordinates — NOT available.** Verified directly from Sportmonks' own engineering blog post ["Modeling Expected Goals (xG) from Match Event Data"](https://www.sportmonks.com/blogs/modeling-expected-goals-xg-from-match-event-data/): *"Shot x/y coordinates: The events endpoint does not return per-shot coordinates."* Sportmonks does offer a **`ballCoordinates`** include — continuous ball-position tracking (~6–12 samples/minute) for top-tier leagues only — but this is explicitly *not* shot-linked and has no `player_id`, so it cannot be attributed to a specific shot without manually aligning it to event timestamps (which Sportmonks' own docs describe as a workaround for building your own model, not a supported feature).

**Coverage of target leagues:** confirmed — Sportmonks maintains dedicated pages for [Premier League](https://www.sportmonks.com/football-api/premier-league-api/), [La Liga](https://www.sportmonks.com/football-api/la-liga-api/), [Bundesliga](https://www.sportmonks.com/football-api/bundesliga-api/), [Serie A](https://www.sportmonks.com/football-api/serie-a-api/), [Ligue 1](https://www.sportmonks.com/football-api/ligue-1-api/), all within the 2,200+ leagues covered.

**Historical coverage:** general fixture/event history goes back multiple seasons, but the **xG data specifically is only available from the 2024 season onward** (Sportmonks FAQ, xG product page: *"xG metrics were not collected for earlier seasons, so they are not available historically before that point"*). Separately, the base subscription plans only include the **last 3 seasons**; anything older requires a one-time "historical data" add-on starting at €29.

**API access & rate limits:** self-serve, instant token, no sales call. REST API, uniform JSON, `include=` pattern similar to OddAlerts. Starter plan: 2,000 calls/hour per entity; Growth: 2,500/hour; Pro: 3,000/hour.

**Pricing (published, no quote needed):**
- Starter — €29/mo (€24/mo paid yearly): pick any 5 leagues — enough to cover exactly our 5 target leagues on the cheapest tier.
- xG add-on (Basic, post-match only) — €15/mo extra, or the "xG & Pressure Index" bundle at €24/mo (24 was €29 discounted).
- 14-day free trial, no contract, cancel any month.
- **Realistic all-in cost for this project: ~€45–55/month (~$50–60/month)** — well inside a ~$700 budget even run for several months.

**Licensing (from Sportmonks' own [Terms of Service](https://www.sportmonks.com/terms-of-service/)):** data may be stored, transformed, and used to build a product you charge for — *"if you use our data to create something based on our data and start earning money from your creation, everything is fine."* Reselling the raw data itself is prohibited; using it as an input to your own prediction system is explicitly the intended use case (Sportmonks lists "Sports Betting Platforms" as one of its named solution verticals). Local storage/derivation of model features is allowed (distribution/storage of the data "is allowed," only reselling the raw product is not).

**Data quality signal:** independent case studies from betting-product customers (Packball, Fine Line, BF Bot Manager — 7+ years, no major problems) are published with named contacts, which is a stronger reliability signal than marketing copy alone, though obviously still vendor-selected testimonials.

**Integration difficulty:** low. Same REST/JSON/`include=` shape you're already handling for OddAlerts; a Python client can reuse most of the same ingestion pattern.

---

### 2. API-Football (api-sports.io)

**What OddAlerts is missing that API-Football provides:** the same category of gap-filler as Sportmonks — a `/fixtures/events` endpoint returning goals (scorer + assist + minute), cards, substitutions, and VAR events, plus a `/fixtures/lineups` endpoint with confirmed starting XI and formations, plus a `/fixtures/statistics` endpoint with granular shot counts (shots on goal, shots off goal, shots inside box, shots outside box, total shots, blocked shots) at match level.

**Shot X/Y coordinates — no evidence found anywhere in available documentation or third-party technical write-ups.** Not claimed by the provider anywhere I could verify. Treat as NOT AVAILABLE, consistent with Sportmonks and consistent with this simply not being a feature either budget-tier provider offers.

**xG — unconfirmed with low confidence.** Some third-party/aggregator pages reference `/expected/fixtures` and `/expected/lineups` endpoints "for subscribers with xG packages," but I could not verify this against api-football.com's own documentation page directly — it's a client-rendered single-page app that returned no readable content to automated fetching, and the Chrome extension needed to render it wasn't available in this session. **This should be verified directly in the API-Football dashboard/docs (or by asking their support) before counting on xG from this provider.** Their published pricing/feature list (which *did* load) does not list "xG" or "Expected Goals" as a named feature on any tier — only "Statistics" and "Predictions."

**Coverage of target leagues:** confirmed generally — API-Football advertises 1,200+ leagues/cups worldwide, which includes the top 5 European leagues (this is corroborated by widespread community/tutorial usage referencing exactly these leagues), though I did not pull a live coverage list the way I did for OddAlerts.

**Historical coverage:** the free plan is explicitly "limited in terms of available seasons" per their own pricing page; paid plans (Pro and up) are described as including full competition/endpoint access, but the exact number of historical seasons per league isn't stated as a hard number on the pricing page — would need confirming from the dashboard "Coverage" tool before relying on it for a multi-season backtest.

**API access & rate limits:** header-based auth (`x-apisports-key`) to `https://v3.football.api-sports.io`. Free: 100 requests/day. Pro: $19/mo, 7,500/day. Ultra: $29/mo, 75,000/day. Mega: $39/mo, 150,000/day. All paid tiers include every endpoint (no feature-gating by tier, only volume).

**Pricing:** cheapest of the realistic candidates — **$19/month** gets full endpoint access at 7,500 requests/day, comfortably enough to backfill 5 leagues × 6 seasons of events/lineups/stats within a few days and then run ongoing ingestion.

**Licensing (from their own [Terms of Service](https://www.api-football.com/terms), fetched in full):**
- Explicit prohibition on **reselling the raw data** to third parties, but explicit permission to use it to build "applications, websites, fantasy soccer games etc." — same shape of permission as Sportmonks.
- Important caveat specific to our use case, quoted directly: *"The use of our data for betting platforms, television broadcasting, fantasy sports platforms, or any mass media distribution may require additional licenses from the relevant rights holders. Users must ensure their use complies with applicable legal and commercial restrictions."* This is about league/competition IP (names, logos, official data rights) rather than the API data itself, but it's a more explicit betting-specific warning than Sportmonks gives, and worth flagging to the client since this is a football *prediction* system.
- Data provided "as is," no accuracy guarantee, no refunds for data errors.
- Governed by French law.

**Integration difficulty:** low — simple REST/JSON, well-documented in community tutorials even though the primary docs site didn't render for automated fetching in this session.

---

### 3. StatsBomb (Hudl)

**What it would add:** this is the *only* candidate researched that has genuine, best-in-class shot-level data — including shot location (x/y), shot freeze-frame data (defender/attacker positions at the moment of the shot), and a shot-level xG model built from that positional data, which is exactly the category of data OddAlerts is missing.

**Access model:** two tiers, confirmed from StatsBomb's own GitHub org (`github.com/statsbomb/open-data`, `github.com/statsbomb/statsbombpy`):
1. **Free Open Data** — a fixed set of historical competitions/seasons released publicly on GitHub for research and personal analytics use. This does **not** include current-season Premier League/La Liga/Bundesliga/Serie A/Ligue 1 coverage on a rolling basis — it's a curated, static archive (famous tournaments, some historical leagues, women's football, etc.), and its license is explicitly for **non-commercial research use**, not for building a paid client deliverable.
2. **Commercial API** — full current coverage of the top leagues with shot/event/freeze-frame data, but "API access is for paying customers only" and pricing is not published; it requires a sales conversation.

**Realistic for a ~$700 budget project:** **no.** The free tier is legally unusable for a paid client system (research/personal use only) and doesn't have live current-season coverage of all 5 target leagues; the commercial tier is enterprise-priced (industry reporting on comparable elite providers puts this well into four-to-five-figure annual territory, consistent with StatsBomb's positioning as an elite/pro-club-grade provider). Flagging this honestly rather than guessing a number: **no public price exists, and every available signal points to it being far outside this project's budget.**

---

### 4. Wyscout / Hudl

Same story as StatsBomb, one level more opaque: Hudl's own product pages for the [Wyscout Data API](https://www.hudl.com/en_gb/products/wyscout/data-api) and pricing page do not publish numbers at all — access requires filling out a sales contact form. No self-serve signup, no published tier, no free trial found. **Not realistic for a $700 local-dev budget** — this is an enterprise/agency-tier product (Wyscout is primarily a scouting/video platform for professional clubs; the data API is a secondary commercial offering on top of that).

### 5. Opta / Stats Perform

Confirmed via Stats Perform's own [Pricing & Licensing FAQ](https://www.statsperform.com/faqs/stats-perform-faqs-pricing-licensing/): *"products and pricing are tailored to meet the requirements of large and small enterprise-level clients... designed for organisations rather than individual use... contact our expert sales team for a customised quote."* No self-serve tier, no published price, sales-quote-only. This is the data source most professional broadcasters and top-flight clubs use, and it shows in the access model — **not realistic for this budget.**

### 6. Sportradar

Confirmed no public pricing; third-party sports-data market analysis (LSports' own published pricing guide, cross-referenced against multiple sports-data comparison sites) consistently cites Sportradar's typical commercial contracts in the low-thousands-of-euros-per-month range and up, with annual minimum commitments and no self-serve signup. **Not realistic for this budget**, and would be significant overkill for a local-development project at this stage regardless.

### 7. Understat

**Technical fit is actually the best of all seven** for the specific gap: understat.com computes and publishes its own shot-level xG model with shot x/y coordinates, specifically for the top 5 European leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1 — an exact match to our target leagues) going back several seasons, and it's free.

**Why it's disqualified anyway:** there is no official API. Every access method found (`understatAPI`, `understatscraper`, and similar PyPI/GitHub packages) is an **unofficial scraper against the public website**, and every one of them explicitly disclaims any affiliation with or endorsement by Understat. I found no published Terms of Service or data-licensing statement from Understat covering commercial use, redistribution, or reliance in a paid product — which means:
- **Legal risk (criterion #9):** we cannot confirm this data can legally be used in a paid client project. Scraping a website with no stated commercial-use license is a real, unquantified legal exposure for a project being delivered to a paying client, not a hypothetical one.
- **Reliability risk (criterion #12):** no SLA, no support channel, no guarantee the site structure won't change and silently break ingestion (this happens routinely to scraper-based sports data tools).

**Recommendation on Understat:** do not use it as a production data source for the client deliverable. It's worth keeping in mind as a free *validation* reference during development (e.g., spot-checking whether our OddAlerts-derived xG numbers for a handful of known matches look reasonable against Understat's independently-computed values) — but that's a developer sanity-check, not a data pipeline dependency, and should not be presented to the client as part of the supported data stack.

---

## Ranked recommendation

### #1 Recommended additional provider: Sportmonks

**Why:** it's the only researched provider that is simultaneously (a) self-serve and affordably priced, (b) confirmed to cover exactly our 5 target leagues on its cheapest plan, (c) confirmed via its own engineering documentation (not marketing copy) to provide the discrete goal/card/substitution/VAR event timeline with player and minute that OddAlerts is missing, (d) provides real lineups/formations, and (e) offers a second, independently-computed xG model as a cross-check against OddAlerts' xG — valuable for the data-quality validation your project instructions require, separate from the event-timeline gap-fill. Its Terms of Service are the clearest and most favorable of the realistic candidates about building a paid product (including betting-adjacent products) on top of the data.

### #2 Recommended additional provider: API-Football

**Why:** near-identical feature set to Sportmonks (events, lineups, statistics) at a lower headline price ($19/mo vs. ~€45-55/mo all-in for Sportmonks with xG), with a genuinely free tier (100 req/day) that's useful for early prototyping before committing spend. Its main weaknesses relative to Sportmonks: xG availability is unconfirmed (needs direct verification, possibly gated behind a separate paid package), its documentation site could not be independently verified in this session (client-rendered, blocked automated access), and its Terms of Service carry a more explicit betting-specific licensing caveat that should be read carefully before use in a prediction product.

### Why these two complement OddAlerts

Both close the same specific, confirmed gap: **discrete match events with player attribution and minute-level timestamps, and real starting lineups** — the two things DATA_AUDIT.md flagged as genuinely absent from OddAlerts (OddAlerts only exposes 1H/2H aggregate splits and a formation string, never a per-event timeline or confirmed XI). Sportmonks additionally provides an independent xG source, useful for cross-validating OddAlerts' own xG numbers per your project's "validate rather than assume" principle. **Neither one closes the shot X/Y coordinate / freeze-frame gap** — that requires StatsBomb, Opta, or a similarly-tiered provider, all of which are enterprise-priced and out of scope for this budget (see below).

### What each provider adds (summary)

| | Sportmonks | API-Football |
|---|---|---|
| Discrete goal/card/sub/VAR events, w/ player+minute | Yes | Yes |
| Confirmed starting lineups/formations | Yes | Yes |
| Shot x/y coordinates | **No** (confirmed) | No evidence found |
| Independent xG (secondary to OddAlerts) | Yes, 13 metrics, from 2024 season only | Unconfirmed |
| Ball-tracking (non-shot-linked) | Yes, top leagues only | Not found |
| Target league coverage on cheapest tier | Yes (pick-5 plan) | Yes (all-tier) |

### Estimated cost

- **Sportmonks:** ~€45–55/month (~$50–60) for Starter (5 leagues) + xG add-on. Annual billing saves ~20%.
- **API-Football:** $19/month (Pro tier) for full endpoint access at a volume more than sufficient for 5 leagues of backfill + ongoing updates. Free tier available for initial testing at $0.
- **Combined, both providers, one full month of overlap testing:** roughly $70–80 — a small fraction of the ~$700 project budget, leaving most of it for development time rather than data costs.

### Licensing risk

- **Sportmonks:** Low. Clear written permission to build and monetize a product on the data; only raw resale is prohibited.
- **API-Football:** Low-to-moderate. Same core permission, but its own ToS flags betting-platform use as potentially needing "additional licenses from relevant rights holders" — worth a deliberate read-through (and possibly a support email to API-Football) before finalizing, specifically because this project is a prediction system.
- **Both:** neither grants a "license" to publish official league/team logos or names without separately confirming rights with the leagues — irrelevant for an internal client prediction system, relevant if this ever becomes a public-facing product.
- **StatsBomb/Wyscout/Opta/Sportradar:** would be resolved via a negotiated commercial contract (out of scope now).
- **Understat:** unresolved/unquantifiable legal risk — this is the reason it's excluded despite the best technical fit.

### Technical integration difficulty

Both are low-difficulty: standard REST/JSON APIs with a similar shape to what's already been audited for OddAlerts (paginated list endpoints, `include=`/query-param-driven field selection). No SDKs are required; a lightweight Python client per provider, following the same pattern as the OddAlerts client, is sufficient. The main engineering work is not "talking to the API," it's building an ID-mapping layer so that a fixture/team/player in OddAlerts can be reliably matched to the same fixture/team/player in Sportmonks and/or API-Football (different providers use different internal IDs) — this should be scoped as its own small piece of work before any event-level feature engineering begins.

### Whether we actually need both

**Probably not both, on an ongoing basis.** Sportmonks and API-Football overlap almost completely on the specific gap we're trying to close (discrete events + lineups). Running both long-term would mostly buy redundancy/cross-validation, not new information. A more cost- and effort-efficient path:

1. Start with **Sportmonks only** (better-documented ToS, confirmed xG methodology write-up, cleaner licensing language for a betting-adjacent product, and the pick-5-leagues plan is a precise fit for this project).
2. Use **API-Football's free tier** opportunistically during development — e.g., to spot-check a handful of matches against Sportmonks for data-quality validation (three-way comparison against OddAlerts too) — without paying for a second full subscription.
3. Only upgrade to a paid API-Football subscription (or reconsider adding it permanently) if Sportmonks' data turns out to have gaps or quality issues for our specific 5 leagues once we start using it for real — a decision to make with evidence, not in advance.

This keeps monthly data spend near $50–60 instead of $70–80+, and follows the project's stated principle of validating assumptions empirically rather than provisioning for redundancy we haven't shown we need.

### Recommended final data stack

**Primary recommendation:**

```
OddAlerts (fixtures, results, aggregate stats, match xG/xGOT, odds)
+
Sportmonks (discrete goal/card/sub/VAR events with player+minute, confirmed lineups, secondary xG for cross-validation)
```

**API-Football held in reserve**, free-tier only, as a spot-check/backup data source rather than a second paid subscription — promote it to a full paid provider only if a concrete data-quality gap in Sportmonks is found for our 5 leagues.

**Explicitly not recommended for this phase:** StatsBomb, Wyscout/Hudl, Opta/Stats Perform, Sportradar (all enterprise-priced, sales-quote-only, out of scope for a ~$700 local-development budget) and Understat (best technical fit for shot X/Y data, but no official API and no confirmable license for commercial/client use — real legal exposure, not a hypothetical one).

If the client later decides shot-level X/Y coordinates and freeze-frame data are a hard requirement (e.g., for a more sophisticated positional xG model beyond what OddAlerts + Sportmonks can support), that is a separate, larger budget conversation involving StatsBomb or Opta and should be scoped and approved on its own — it does not fit inside this project's current budget or "local development" framing.

---

## Confidence notes / what to verify before spending money

- API-Football's xG availability and exact per-league historical depth were **not** confirmed against their own primary documentation (the docs site is a client-rendered SPA that didn't return readable content to this session's tools). Before subscribing, either render the docs in a browser directly or email API-Football support to confirm (a) whether xG is included in the Pro tier or requires a separate package, and (b) how many historical seasons per target league are actually populated.
- Sportmonks' xG is confirmed available only from the 2024 season onward — if the backtesting plan needs xG going further back than that, Sportmonks doesn't solve that either, and neither does OddAlerts' own coverage-completeness caveat noted in DATA_AUDIT.md.
- No live API test calls were made to either Sportmonks or API-Football in this research pass (no account/token exists yet for either) — all findings here are from official docs, pricing pages, and ToS. A short paid or free-tier trial should be run against real fixtures for our 5 target leagues before committing to a monthly subscription, the same way OddAlerts was live-tested in DATA_AUDIT.md.
