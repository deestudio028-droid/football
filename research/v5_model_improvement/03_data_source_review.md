# 03 — External Data Source Audit: Coverage, Quality, & Causal Viability

## 1. Executive Summary

This audit evaluates external football data sources across the five target European leagues:
1. **Premier League** (England)
2. **La Liga** (Spain)
3. **Bundesliga** (Germany)
4. **Serie A** (Italy)
5. **Ligue 1** (France)

We assess their historical depth, update latency, licensing terms, free vs. paid availability, and—most importantly—**their causal pre-match timestamp reliability**. 

**Policy Constraint**: Under no circumstances will paid commercial APIs be introduced into this project. All candidate datasets must be obtainable via free research tiers, open datasets, or public domain archives.

---

## 2. Comprehensive Source Evaluations

### Source 1: Football-Data.co.uk (Joseph Buchdahl Archive)
- **Primary URL**: [https://www.football-data.co.uk/data.php](https://www.football-data.co.uk/data.php)
- **Data Available**:
  - Full match outcomes: Full-time goals (`FTHG`, `FTAG`), Halftime goals (`HTHG`, `HTAG`), Full-time result (`FTR`), Halftime result (`HTR`).
  - Match statistics: Total shots (`HS`, `AS`), Shots on Target (`HST`, `AST`), Fouls (`HF`, `AF`), Corners (`HC`, `AC`), Yellow cards (`HY`, `AY`), Red cards (`HR`, `AR`).
  - Referee names (`Referee`).
  - Multi-bookmaker odds: Bet365, Pinnacle, Bwin, William Hill, Interwetten, Ladbrokes, Betfair Exchange, Market Average (`AvgH`, `AvgD`, `AvgA`), Market Maximum (`MaxH`, `MaxD`, `MaxA`).
- **Historical Depth**: 1993/94 to present (30+ consecutive seasons for all 5 top leagues).
- **Update Frequency**: Updated weekly (typically Monday and Friday evenings following match rounds).
- **API Availability**: Static CSV download per league/season; easily automated via script.
- **Cost**: 100% Free / Public Domain for research and analytical use.
- **Rate Limits**: None (static files hosted on web server).
- **Data Quality & Reliability**: Exceptional for Top 5 leagues. Team name conventions are standard and well-documented.
- **Causal Availability**:
  - Match statistics are strictly post-match, but can be aggregated causally into rolling pre-match indicators (e.g., $HST_{\text{last 5}}$, $AST_{\text{last 5}}$, rolling fouls, cards).
  - Pre-match odds reflect historical market closing lines.
- **Usage Recommendation**:
  - **Research**: **MANDATORY CORE DATASET**. Perfect for historical shot-on-target, referee, and market odds benchmarking back to 2000.
  - **Production**: Suitable for weekly offline historical table synchronization.

---

### Source 2: Understat Open $xG$ Data
- **Primary URL**: [https://understat.com](https://understat.com)
- **Data Available**:
  - Match-level Expected Goals ($xG_{\text{home}}, xG_{\text{away}}$), Expected Goals Against ($xGA$).
  - Non-penalty $xG$ ($npxG$), Expected Points ($xPTS$), Passes Per Defensive Action ($PPDA$).
  - Deep completions (passes completed within 20 yards of goal excluding crosses).
  - Shot-level details: $(x, y)$ coordinates, shot result, situation (open play, corner, direct free kick, set piece, penalty), player name, body part, individual shot $xG$.
- **Historical Depth**: 2014/15 season to present (12+ seasons for all 5 leagues).
- **Update Frequency**: Within 1–2 hours after match completion.
- **API Availability**: Web accessible JSON endpoints (parsed easily via `understatapi` or Python `requests`).
- **Cost**: Free for research and non-commercial analytical use.
- **Rate Limits**: Gentle scraping required (1–2 requests/sec to prevent IP rate-limiting).
- **Data Quality & Reliability**: High consistency. Model applies a uniform neural network/gradient boosting $xG$ algorithm across all leagues from 2014 to present.
- **Causal Availability**:
  - Post-match $xG$ is available 2 hours post-match.
  - Pre-match rolling features ($xG_{\text{last 5}}, xGA_{\text{last 5}}, npxG_{\text{diff}}$) can be constructed causally with zero future leakage.
- **Usage Recommendation**:
  - **Research**: **PRIMARY $xG$ DATASET**. Allows constructing high-signal rolling expected goal features from 2014/15 onward.
  - **Production**: Rolling $xG$ states updated after every completed round.

---

### Source 3: StatsBomb Open Data
- **Primary URL**: [https://github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)
- **Data Available**:
  - Deep event stream data: complete pitch coordinates for every pass, pressure, carry, tackle, interception, and shot.
  - StatsBomb 360 freeze frames (locations of all 22 players at the moment of shot or key pass).
  - StatsBomb proprietary $xG$, shot impact metrics, goalkeeper post-shot $xG$ ($PSxG$).
- **Historical Depth**: Selected leagues/seasons (e.g., complete La Liga 2004–2021 for Lionel Messi career, UEFA Champions League, selected Premier League and Bundesliga seasons).
- **Update Frequency**: Periodic static releases.
- **API Availability**: GitHub repository with JSON data dump.
- **Cost**: 100% Free under StatsBomb Open Data User Agreement.
- **Rate Limits**: None (direct Git clone).
- **Data Quality & Reliability**: Gold-standard in sports analytics research.
- **Causal Availability**: Rich post-match event data, but lacks continuous season-by-season coverage for all 5 current leagues.
- **Usage Recommendation**:
  - **Research**: Excellent for validating tactical/spatial feature engineering concepts (e.g., testing whether pressure metrics predict goal concession).
  - **Production**: **Not feasible** due to incomplete season coverage across current active leagues.

---

### Source 4: ClubElo Archive (Lars Schiefler)
- **Primary URL**: [http://clubelo.com/API](http://clubelo.com/API)
- **Data Available**:
  - Daily historical Elo ratings for all European football clubs dating back to the 1950s.
  - Expected match probabilities and home advantage constants.
  - Club Elo updates after every official domestic and European match.
- **Historical Depth**: 1950s to present (complete coverage of all 5 target leagues).
- **Update Frequency**: Daily (overnight after matches).
- **API Availability**: Clean public REST API (e.g., `http://api.clubelo.com/2026-08-22` or `http://api.clubelo.com/Arsenal`).
- **Cost**: Free for personal/research use.
- **Rate Limits**: Standard web access (unmetered within reasonable bounds).
- **Data Quality & Reliability**: Highly regarded in academic and industry benchmarks.
- **Causal Availability**:
  - Daily snapshots provide strictly causal pre-match Elo ratings ($R_{\text{club}}(t)$ before match $t$).
- **Usage Recommendation**:
  - **Research**: High-value benchmark for comparing against our internal causal Elo (`src/features/elo.py`).
  - **Production**: Optional external validation anchor for team rating cross-checks.

---

### Source 5: OddAlerts API (Our Active Live Feed Provider)
- **Primary URL**: `https://data.oddalerts.com/api/`
- **Data Available**:
  - Live upcoming fixture schedules, kickoff UTC timestamps, competition metadata, venue details.
  - Real-time match state: statuses (`NS`, `1H`, `HT`, `2H`, `FT`), live scores, final completed scores.
  - In-play match statistics: possession, shots, shots on target, dangerous attacks, fouls, cards, corners.
  - Pre-match odds feed.
- **Historical Depth**: Current active season (2026/27) and recent rolling windows.
- **Update Frequency**: Real-time (1–60 seconds for in-play and upcoming schedules).
- **API Availability**: REST API authenticated via API token.
- **Cost**: Active project subscription/token in place.
- **Rate Limits**: Multi-page queries require paginated caching (`ttl = 180s`).
- **Data Quality & Reliability**: High reliability for live schedule discovery, fixture ID tracking, and score reconciliation.
- **Causal Availability**:
  - Timestamps are provided in ISO 8601 UTC.
  - Serves as the primary ground-truth source for pre-kickoff cutoffs and post-match outcome reconciliation.
- **Usage Recommendation**:
  - **Research**: Live prospective evaluation pipeline (`research/v4_promotion/live_v46_prospective.sqlite`).
  - **Production**: Active live feed for the Model Lab dashboard (`src/data/providers/football_fixture_provider.py`).

---

### Source 6: Kaggle European Soccer Database
- **Primary URL**: [https://www.kaggle.com/datasets/hugomathien/soccer](https://www.kaggle.com/datasets/hugomathien/soccer)
- **Data Available**:
  - 25,000+ matches from 11 European countries (2008 to 2016).
  - Match details, betting odds, line-up player IDs, coordinates, and FIFA player attributes (updated weekly from EA Sports).
- **Historical Depth**: 2008/09 to 2015/16.
- **Update Frequency**: Static archive (stopped in 2016).
- **Cost**: Free / Public Domain (CC BY-NC-SA 4.0).
- **Usage Recommendation**:
  - **Research**: Useful historical sandbox for testing player lineup aggregation algorithms and FIFA rating proxies on older seasons.
  - **Production**: Obsolete (not updated for current seasons).

---

## 3. Data Source Evaluation Matrix

| Source | Historical Depth | Leagues Covered | Key Predictive Features | Cost / Limits | Causal Reliability | Recommended Role in $V_5$ |
|---|---|---|---|---|---|---|
| **Football-Data.co.uk** | 1993–2026 (30+ yrs) | All 5 Leagues | Shots on Target, Corners, Fouls, Referee, Historical Odds | **100% Free** (No limits) | **High** (Strict date ordering) | **Core Historical Backbone** (Shots, Referees, Odds) |
| **Understat** | 2014–2026 (12 yrs) | All 5 Leagues | Match $xG$, $xGA$, $npxG$, Deep Completions, PPDA, Shot Maps | **100% Free** (Gentle scraping) | **High** (Pre-match rolling states) | **Core $xG$ Feature Engine** (Rolling $xG$ ratings) |
| **ClubElo** | 1950–2026 (70+ yrs) | All 5 Leagues | Official European Club Elo Ratings & Probabilities | **100% Free** (Public API) | **High** (Daily pre-match snapshots) | **External Elo Benchmark & Validation** |
| **OddAlerts API** | 2024–2026 (Live) | All 5 Leagues | Live Fixture Feeds, Real-Time Kickoffs, Completed Scores | **Active / In Use** | **High** (Real-time UTC timestamps) | **Live Matchday Discovery & Outcome Reconciliation** |
| **StatsBomb Open Data** | Selective (2004–2021) | Partial (La Liga/UCL) | 360 Event Streams, Freeze Frames, Pressure Events | **100% Free** (GitHub repo) | **High** (Event-level) | **Exploratory Tactical Research Only** |
| **Kaggle European DB** | 2008–2016 (Static) | 11 Leagues | Historical Lineups, FIFA Player Attributes | **100% Free** (Archive) | **Moderate** (Dated) | **Lineup Aggregation Feasibility Sandbox** |

---

## 4. Key Takeaways & Ingestion Strategy for $V_5$

1. **The "Holy Trinity" of Free Football Data**:
   By uniting **Football-Data.co.uk** (1993–2026: shots on target, referee names, multi-bookmaker odds), **Understat** (2014–2026: $xG, xGA, npxG, PPDA$), and **OddAlerts API** (live 2026/27 fixtures and outcomes), we acquire complete, high-fidelity data across all five leagues at **zero incremental monetary cost**.
2. **Standardized Team Name Aliasing**:
   Each provider uses distinct naming conventions (e.g., `Marseille` vs `Olympique Marseille`, `Athletic Bilbao` vs `Athletic Club`, `PSG` vs `Paris Saint Germain`). We must expand our centralized team entity resolver (`src/dashboard/prediction_service.py` and `src/features/context.py`) with a unified cross-source entity mapping table.
3. **Causal Data Provenance & Ingestion Separation**:
   All external datasets will be compiled into immutable, versioned SQLite and Parquet files in `data/research/` before running any $V_5$ experiments, completely isolated from production databases (`matches.db`, `features.db`).
