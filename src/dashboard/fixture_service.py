"""Fixture Service for Football Prediction Lab Dashboard.

Queries real API feeds (OddAlerts), local SQLite database, or test harnesses across the 5 target leagues.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"
COMPETITIONS_JSON = PROJECT_ROOT / "config" / "competitions.json"

from .matchweek_service import MatchweekService
from data.providers.football_fixture_provider import (
    BaseFixtureProvider,
    FixtureProviderError,
    LocalDatabaseFixtureProvider,
    NoProviderCredentialsError,
    OddAlertsFixtureProvider,
    ProviderConnectionError,
    TARGET_COMPETITIONS,
    UpcomingFixture,
)

logger = logging.getLogger("dashboard_fixture_service")

TARGET_LEAGUES: Dict[int, str] = {
    423: "Premier League",
    419: "La Liga",
    477: "Bundesliga",
    499: "Serie A",
    200: "Ligue 1",
}

# Canonical Single Source of Truth for Weekly League Fixture Targets
WEEKLY_LEAGUE_TARGETS: Dict[str, int] = {
    "Premier League": 10,
    "Serie A": 10,
    "La Liga": 10,
    "Bundesliga": 9,
    "Ligue 1": 9,
}

WEEKLY_COMPETITION_TARGETS: Dict[int, int] = {
    423: 10,  # Premier League (20 teams = 10 matches)
    499: 10,  # Serie A (20 teams = 10 matches)
    419: 10,  # La Liga (20 teams = 10 matches)
    477: 9,   # Bundesliga (18 teams = 9 matches)
    200: 9,   # Ligue 1 (18 teams = 9 matches)
}

TOTAL_WEEKLY_TARGET: int = sum(WEEKLY_LEAGUE_TARGETS.values())  # exactly 48


@dataclass
class DashboardFixture:
    """Fixture record formatted for dashboard display and evaluation."""
    fixture_id: int
    competition_id: int
    competition_name: str
    season_name: str
    home_team: str
    away_team: str
    scheduled_kickoff: str
    status: str  # 'FT', 'SCHEDULED', 'TIMED', 'NS', 'UPCOMING', etc.
    provider: str  # 'OddAlerts API' or 'Local Database'
    home_id: Optional[int] = None
    away_id: Optional[int] = None
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    actual_outcome: Optional[str] = None  # 'H', 'D', 'A'
    matchweek: Optional[str] = None


class FixtureService:
    """Service managing fixture discovery across real API and local database providers."""

    def __init__(self, db_path: Path = MATCHES_DB, custom_provider: Optional[BaseFixtureProvider] = None):
        self.db_path = Path(db_path)
        self.custom_provider = custom_provider
        self._local_provider = LocalDatabaseFixtureProvider(db_path=self.db_path)
        self._api_provider: Optional[OddAlertsFixtureProvider] = None
        self._api_init_error: Optional[str] = None

        try:
            self._api_provider = OddAlertsFixtureProvider()
        except NoProviderCredentialsError:
            self._api_init_error = "OddAlerts_API credentials not configured in environment or .env."
        except Exception as exc:
            self._api_init_error = f"OddAlerts initialization failed: {exc}"

        self._matchweek_service = MatchweekService()

    def get_matchweek(self, league_name: str, scheduled_kickoff: str) -> Optional[str]:
        """Resolve matchweek from league name and scheduled kickoff date."""
        return self._matchweek_service.get_matchweek(league_name, scheduled_kickoff)

    def get_supported_leagues(self) -> Dict[int, str]:
        return dict(TARGET_LEAGUES)

    def get_teams_by_league(self, competition_id: int) -> List[str]:
        """Fetch all unique team names for a given competition."""
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT home_name FROM fixtures WHERE competition_id = ?
            UNION
            SELECT DISTINCT away_name FROM fixtures WHERE competition_id = ?
            ORDER BY 1 ASC
        """, (competition_id, competition_id))
        teams = [r[0] for r in cur.fetchall() if r[0]]
        conn.close()
        return teams

    def get_todays_matches(
        self,
        date_str: str = "2026-08-22",
        provider_name: str = "oddalerts",
        competition_ids: Optional[List[int]] = None,
        custom_provider: Optional[BaseFixtureProvider] = None,
    ) -> Tuple[List[DashboardFixture], Dict[str, Any]]:
        """Fetch fixtures scheduled for a specific date across target leagues."""
        comp_ids = competition_ids or list(TARGET_LEAGUES.keys())
        provider_key = provider_name.lower().strip()

        # 1. Select provider
        provider: BaseFixtureProvider
        active_provider_display: str

        if custom_provider or self.custom_provider:
            provider = custom_provider or self.custom_provider
            active_provider_display = "Mock/Test Provider"
        elif provider_key in ("oddalerts", "api", "live"):
            if self._api_provider is not None:
                provider = self._api_provider
                active_provider_display = "OddAlerts API"
            else:
                return ([], {
                    "provider_display": "OddAlerts API",
                    "provider_key": "oddalerts",
                    "status": "NO_CREDENTIALS",
                    "error_message": self._api_init_error or "Missing API credentials.",
                    "total_returned": 0,
                    "eligible_count": 0,
                })
        else:
            provider = self._local_provider
            active_provider_display = "Local Database"

        # 2. Fetch fixtures from active provider
        try:
            raw_fixtures: List[UpcomingFixture] = provider.get_fixtures_by_date(
                date_str=date_str,
                competition_ids=comp_ids,
                include_completed=True,
            )
        except NoProviderCredentialsError as e:
            return ([], {
                "provider_display": active_provider_display,
                "provider_key": provider_key,
                "status": "NO_CREDENTIALS",
                "error_message": "Missing live API token in environment.",
                "total_returned": 0,
                "eligible_count": 0,
            })
        except ProviderConnectionError as e:
            return ([], {
                "provider_display": active_provider_display,
                "provider_key": provider_key,
                "status": "CONNECTION_ERROR",
                "error_message": f"Network connection error contacting provider: {e}",
                "total_returned": 0,
                "eligible_count": 0,
            })
        except Exception as e:
            return ([], {
                "provider_display": active_provider_display,
                "provider_key": provider_key,
                "status": "ERROR",
                "error_message": f"Error querying fixtures from provider: {e}",
                "total_returned": 0,
                "eligible_count": 0,
            })

        # 3. 5-League filtering & Deduplication
        results: List[DashboardFixture] = []
        seen_fids: Set[int] = set()

        for rf in raw_fixtures:
            if rf.league_id not in comp_ids:
                continue
            if rf.fixture_id in seen_fids:
                continue
            seen_fids.add(rf.fixture_id)

            act_outcome = None
            is_completed = rf.status in ("FT", "AET", "PEN", "AWARDED")
            if is_completed and rf.home_goals is not None and rf.away_goals is not None:
                hg, ag = int(rf.home_goals), int(rf.away_goals)
                act_outcome = "H" if hg > ag else ("D" if hg == ag else "A")

            results.append(DashboardFixture(
                fixture_id=rf.fixture_id,
                competition_id=rf.league_id,
                competition_name=rf.league_name,
                season_name=str(rf.raw_data.get("season", "2026/2027") if rf.raw_data else "2026/2027"),
                home_team=rf.home_team,
                away_team=rf.away_team,
                home_id=rf.home_team_id,
                away_id=rf.away_team_id,
                scheduled_kickoff=rf.scheduled_kickoff,
                status=rf.status,
                provider=active_provider_display,
                home_goals=rf.home_goals,
                away_goals=rf.away_goals,
                actual_outcome=act_outcome,
                matchweek=self.get_matchweek(rf.league_name, rf.scheduled_kickoff),
            ))

        # Build league breakdown dictionary
        league_counts = {lname: 0 for lname in TARGET_LEAGUES.values()}
        completed_cnt = 0
        upcoming_cnt = 0

        for r in results:
            if r.competition_name in league_counts:
                league_counts[r.competition_name] += 1
            if r.status in ("FT", "AET", "PEN", "AWARDED"):
                completed_cnt += 1
            else:
                upcoming_cnt += 1

        meta = {
            "provider_display": active_provider_display,
            "provider_key": provider_key,
            "status": "SUCCESS" if results else "NO_FIXTURES",
            "error_message": None if results else f"No fixtures returned by provider '{active_provider_display}' for date {date_str}.",
            "total_returned": len(raw_fixtures),
            "eligible_count": len(results),
            "league_breakdown": league_counts,
            "completed_count": completed_cnt,
            "upcoming_count": upcoming_cnt,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        return (results, meta)

    def get_all_available_matches(
        self,
        provider_name: str = "oddalerts",
        competition_ids: Optional[List[int]] = None,
        days_back: int = 7,
        days_ahead: int = 14,
    ) -> Tuple[List[DashboardFixture], Dict[str, Any]]:
        """Fetch ALL currently available fixtures across the active feed without date restrictions."""
        # 1. Resolve Provider
        provider: BaseFixtureProvider
        active_provider_display = "OddAlerts API"
        provider_key = "oddalerts"

        if provider_name.lower().startswith("local"):
            provider = self._local_provider
            active_provider_display = "Local Database"
            provider_key = "local_matches"
        elif self.custom_provider is not None:
            provider = self.custom_provider
            active_provider_display = "Custom Provider"
            provider_key = "custom"
        elif self._api_provider is not None:
            provider = self._api_provider
            active_provider_display = "OddAlerts API"
            provider_key = "oddalerts"
        else:
            provider = self._local_provider
            active_provider_display = "Local Database (Fallback)"
            provider_key = "local_matches"

        comp_ids = set(competition_ids or list(TARGET_LEAGUES.keys()))

        # 2. Query all available fixtures
        try:
            raw_fixtures: List[UpcomingFixture] = provider.get_all_available_fixtures(
                competition_ids=list(comp_ids),
                days_back=days_back,
                days_ahead=days_ahead,
            )
        except NoProviderCredentialsError as e:
            return ([], {
                "provider_display": active_provider_display,
                "provider_key": provider_key,
                "status": "NO_CREDENTIALS",
                "error_message": "Missing live API token in environment.",
                "total_returned": 0,
                "eligible_count": 0,
            })
        except ProviderConnectionError as e:
            return ([], {
                "provider_display": active_provider_display,
                "provider_key": provider_key,
                "status": "CONNECTION_ERROR",
                "error_message": f"Network connection error contacting provider: {e}",
                "total_returned": 0,
                "eligible_count": 0,
            })
        except Exception as e:
            return ([], {
                "provider_display": active_provider_display,
                "provider_key": provider_key,
                "status": "ERROR",
                "error_message": f"Error querying fixtures from provider: {e}",
                "total_returned": 0,
                "eligible_count": 0,
            })

        # 3. 5-League filtering & Deduplication
        results: List[DashboardFixture] = []
        seen_fids: Set[int] = set()

        for rf in raw_fixtures:
            if rf.league_id not in comp_ids:
                continue
            if rf.fixture_id in seen_fids:
                continue
            seen_fids.add(rf.fixture_id)

            act_outcome = None
            is_completed = rf.status in ("FT", "AET", "PEN", "AWARDED")
            if is_completed and rf.home_goals is not None and rf.away_goals is not None:
                hg, ag = int(rf.home_goals), int(rf.away_goals)
                act_outcome = "H" if hg > ag else ("D" if hg == ag else "A")

            results.append(DashboardFixture(
                fixture_id=rf.fixture_id,
                competition_id=rf.league_id,
                competition_name=rf.league_name,
                season_name=str(rf.raw_data.get("season", "2026/2027") if rf.raw_data else "2026/2027"),
                home_team=rf.home_team,
                away_team=rf.away_team,
                home_id=rf.home_team_id,
                away_id=rf.away_team_id,
                scheduled_kickoff=rf.scheduled_kickoff,
                status=rf.status,
                provider=active_provider_display,
                home_goals=rf.home_goals,
                away_goals=rf.away_goals,
                actual_outcome=act_outcome,
            ))

        # Sort by scheduled_kickoff ascending
        results.sort(key=lambda x: x.scheduled_kickoff)

        # Build league breakdown dictionary
        league_counts = {lname: 0 for lname in TARGET_LEAGUES.values()}
        completed_cnt = 0
        upcoming_cnt = 0

        for r in results:
            if r.competition_name in league_counts:
                league_counts[r.competition_name] += 1
            if r.status in ("FT", "AET", "PEN", "AWARDED") or (r.home_goals is not None and r.away_goals is not None):
                completed_cnt += 1
            else:
                upcoming_cnt += 1

        meta = {
            "provider_display": active_provider_display,
            "provider_key": provider_key,
            "status": "SUCCESS" if results else "NO_FIXTURES",
            "error_message": None if results else f"No fixtures returned by provider '{active_provider_display}'.",
            "total_returned": len(raw_fixtures),
            "eligible_count": len(results),
            "league_breakdown": league_counts,
            "completed_count": completed_cnt,
            "upcoming_count": upcoming_cnt,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        return (results, meta)

    @staticmethod
    def _select_league_active_gameweek(
        fixtures: List[DashboardFixture],
        target_count: int,
        now_dt: Optional[datetime] = None,
        offset_weeks: int = 0,
    ) -> List[DashboardFixture]:
        """Select up to target_count matches belonging to this league's active/upcoming matchday cluster.

        Determines the active gameweek independently per league:
        1. If upcoming/in-progress matches exist, anchors on the earliest upcoming match.
        2. Selects the matchday cluster spanning [anchor - 2 days, anchor + 5 days].
        3. If cluster has fewer matches than target, supplements with subsequent upcoming matches.
        4. If no upcoming matches exist at all in feed, falls back to most recent completed matches.
        """
        if not fixtures:
            return []

        sorted_fixtures = sorted(fixtures, key=lambda x: x.scheduled_kickoff)
        if len(sorted_fixtures) <= target_count:
            return sorted_fixtures

        if now_dt is None:
            now_dt = datetime.now(timezone.utc)

        # Identify upcoming or live matches
        upcoming = [
            f for f in sorted_fixtures
            if f.status not in ("FT", "AET", "PEN", "AWARDED") and f.home_goals is None
        ]

        if upcoming:
            # Anchor on earliest upcoming match in this league
            try:
                anchor_dt = datetime.fromisoformat(upcoming[0].scheduled_kickoff.replace("Z", "+00:00"))
                if anchor_dt.tzinfo is None:
                    anchor_dt = anchor_dt.replace(tzinfo=timezone.utc)
            except Exception:
                anchor_dt = now_dt

            # Gameweek cluster window around anchor: [anchor - 2 days, anchor + 5 days]
            c_start = anchor_dt - timedelta(days=2)
            c_end = anchor_dt + timedelta(days=5)

            if offset_weeks > 0:
                for _ in range(offset_weeks):
                    # Find next upcoming match after c_end
                    next_upcoming = None
                    for f in upcoming:
                        try:
                            f_dt = datetime.fromisoformat(f.scheduled_kickoff.replace("Z", "+00:00"))
                            if f_dt.tzinfo is None:
                                f_dt = f_dt.replace(tzinfo=timezone.utc)
                            if f_dt > c_end:
                                next_upcoming = f
                                anchor_dt = f_dt
                                break
                        except Exception:
                            pass
                    
                    if next_upcoming:
                        c_start = anchor_dt - timedelta(days=2)
                        c_end = anchor_dt + timedelta(days=5)
                    else:
                        c_start = anchor_dt + timedelta(days=7)
                        c_end = anchor_dt + timedelta(days=14)
                        break

            cluster_matches = []
            for f in sorted_fixtures:
                try:
                    f_dt = datetime.fromisoformat(f.scheduled_kickoff.replace("Z", "+00:00"))
                    if f_dt.tzinfo is None:
                        f_dt = f_dt.replace(tzinfo=timezone.utc)
                    if c_start <= f_dt <= c_end:
                        cluster_matches.append(f)
                except Exception:
                    continue

            chosen = cluster_matches[:target_count]

            # If cluster has fewer than target_count, supplement with subsequent upcoming matches
            if len(chosen) < target_count:
                chosen_fids = {f.fixture_id for f in chosen}
                for f in upcoming:
                    try:
                        f_dt = datetime.fromisoformat(f.scheduled_kickoff.replace("Z", "+00:00"))
                        if f_dt.tzinfo is None:
                            f_dt = f_dt.replace(tzinfo=timezone.utc)
                        if (offset_weeks == 0 or f_dt >= c_start) and f.fixture_id not in chosen_fids and len(chosen) < target_count:
                            chosen.append(f)
                            chosen_fids.add(f.fixture_id)
                    except Exception:
                        pass

            # If still under target, supplement with chronological fixtures from sorted list that are >= c_start
            if len(chosen) < target_count:
                chosen_fids = {f.fixture_id for f in chosen}
                for f in sorted_fixtures:
                    try:
                        f_dt = datetime.fromisoformat(f.scheduled_kickoff.replace("Z", "+00:00"))
                        if f_dt.tzinfo is None:
                            f_dt = f_dt.replace(tzinfo=timezone.utc)
                        if (offset_weeks == 0 or f_dt >= c_start) and f.fixture_id not in chosen_fids and len(chosen) < target_count:
                            chosen.append(f)
                            chosen_fids.add(f.fixture_id)
                    except Exception:
                        pass

            chosen.sort(key=lambda x: x.scheduled_kickoff)
            return chosen[:target_count]
        else:
            # Fallback when no upcoming matches exist: take the most recent completed matches
            return sorted_fixtures[-target_count:]

    def get_weekly_prediction_fixtures(
        self,
        provider_name: str = "oddalerts",
        league_targets: Optional[Dict[str, int]] = None,
        days_back: int = 7,
        days_ahead: int = 21,
        offset_weeks: int = 0,
    ) -> Tuple[List[DashboardFixture], Dict[str, Any]]:
        """Select exactly the weekly fixture distribution (10 EPL, 10 Serie A, 10 La Liga, 9 Bundesliga, 9 Ligue 1 = 48 total).

        Determines the active gameweek independently per league so staggered season start dates
        (e.g., Bundesliga GW1 starting alongside other leagues' GW2) are handled cleanly.
        """
        targets = league_targets or WEEKLY_LEAGUE_TARGETS
        total_target = sum(targets.values())

        # 1. Fetch all available target fixtures
        all_fixtures, meta_all = self.get_all_available_matches(
            provider_name=provider_name,
            days_back=days_back,
            days_ahead=days_ahead,
        )

        if not all_fixtures:
            return ([], {
                "provider_display": meta_all.get("provider_display", "OddAlerts API"),
                "provider_key": meta_all.get("provider_key", "oddalerts"),
                "status": meta_all.get("status", "NO_FIXTURES"),
                "error_message": meta_all.get("error_message", "No fixtures available."),
                "total_target": total_target,
                "total_selected": 0,
                "league_targets": targets,
                "league_counts": {lname: 0 for lname in targets},
                "shortfalls": {lname: targets[lname] for lname in targets},
                "is_full_gameweek": False,
                "completed_count": 0,
                "upcoming_count": 0,
                "fetched_at": meta_all.get("fetched_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
            })

        # 2. Group fixtures by target league
        by_league: Dict[str, List[DashboardFixture]] = {lname: [] for lname in targets}
        for f in all_fixtures:
            if f.competition_name in by_league:
                by_league[f.competition_name].append(f)

        # 3. Select active gameweek independently per league
        selected: List[DashboardFixture] = []
        league_counts: Dict[str, int] = {}
        shortfalls: Dict[str, int] = {}
        now_dt = datetime.now(timezone.utc)

        for lname, target_count in targets.items():
            league_all = by_league[lname]
            chosen = self._select_league_active_gameweek(league_all, target_count, now_dt=now_dt, offset_weeks=offset_weeks)
            selected.extend(chosen)
            league_counts[lname] = len(chosen)
            shortfalls[lname] = max(0, target_count - len(chosen))

        # 4. Sort selected collection chronologically by scheduled_kickoff ascending
        selected.sort(key=lambda x: x.scheduled_kickoff)

        completed_cnt = sum(1 for f in selected if f.status in ("FT", "AET", "PEN", "AWARDED") or (f.home_goals is not None and f.away_goals is not None))
        upcoming_cnt = len(selected) - completed_cnt
        is_full = (len(selected) == total_target) and all(s == 0 for s in shortfalls.values())

        meta = {
            "provider_display": meta_all.get("provider_display", "OddAlerts API"),
            "provider_key": meta_all.get("provider_key", "oddalerts"),
            "status": "SUCCESS" if selected else "NO_FIXTURES",
            "error_message": None,
            "total_target": total_target,
            "total_selected": len(selected),
            "league_targets": targets,
            "league_counts": league_counts,
            "shortfalls": shortfalls,
            "is_full_gameweek": is_full,
            "completed_count": completed_cnt,
            "upcoming_count": upcoming_cnt,
            "fetched_at": meta_all.get("fetched_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        }
        return (selected, meta)

    def discover_2026_27_fixtures(self) -> Dict[str, Any]:
        """Query and categorize all 2026/27 season fixtures from database."""
        if not self.db_path.exists():
            return {
                "total_discovered": 0,
                "completed_ft": 0,
                "upcoming": 0,
                "completed_fixtures": [],
                "upcoming_fixtures": [],
            }

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT fixture_id, competition_id, competition_name, season, date, unix,
                   home_name, away_name, home_id, away_id, status, home_goals, away_goals
            FROM fixtures
            WHERE (season = '2026/2027' OR date >= '2026-07-01')
              AND competition_id IN (200, 419, 423, 477, 499)
            ORDER BY unix ASC, date ASC
        """)
        rows = cur.fetchall()
        conn.close()

        completed: List[DashboardFixture] = []
        upcoming: List[DashboardFixture] = []

        for r in rows:
            st = str(r["status"]).upper()
            act_outcome = None
            if st == "FT" and r["home_goals"] is not None and r["away_goals"] is not None:
                hg, ag = int(r["home_goals"]), int(r["away_goals"])
                act_outcome = "H" if hg > ag else ("D" if hg == ag else "A")

            fix_obj = DashboardFixture(
                fixture_id=int(r["fixture_id"]),
                competition_id=int(r["competition_id"]),
                competition_name=str(r["competition_name"]),
                season_name=str(r["season"]),
                home_team=str(r["home_name"]),
                away_team=str(r["away_name"]),
                home_id=int(r["home_id"]),
                away_id=int(r["away_id"]),
                scheduled_kickoff=str(r["date"]),
                status=st,
                provider="Local Database",
                home_goals=int(r["home_goals"]) if r["home_goals"] is not None else None,
                away_goals=int(r["away_goals"]) if r["away_goals"] is not None else None,
                actual_outcome=act_outcome,
            )

            if st == "FT":
                completed.append(fix_obj)
            else:
                upcoming.append(fix_obj)

        return {
            "total_discovered": len(rows),
            "completed_ft": len(completed),
            "upcoming": len(upcoming),
            "completed_fixtures": completed,
            "upcoming_fixtures": upcoming,
        }

    def get_fixture_details(self, fixture_id: int) -> Optional[DashboardFixture]:
        """Fetch details for a specific fixture ID from database or live providers."""
        if self.custom_provider:
            rf = self.custom_provider.get_fixture_details(fixture_id)
            if rf:
                act_outcome = None
                if rf.status in ("FT", "AET", "PEN", "AWARDED") and rf.home_goals is not None and rf.away_goals is not None:
                    hg, ag = int(rf.home_goals), int(rf.away_goals)
                    act_outcome = "H" if hg > ag else ("D" if hg == ag else "A")
                return DashboardFixture(
                    fixture_id=rf.fixture_id,
                    competition_id=rf.league_id,
                    competition_name=rf.league_name,
                    season_name=str(rf.raw_data.get("season", "2026/2027") if rf.raw_data else "2026/2027"),
                    home_team=rf.home_team,
                    away_team=rf.away_team,
                    home_id=rf.home_team_id,
                    away_id=rf.away_team_id,
                    scheduled_kickoff=rf.scheduled_kickoff,
                    status=rf.status,
                    provider="Mock/Test Provider",
                    home_goals=rf.home_goals,
                    away_goals=rf.away_goals,
                    actual_outcome=act_outcome,
                )

        rf = self._local_provider.get_fixture_details(fixture_id)
        if rf is None and self._api_provider is not None and hasattr(self._api_provider, "get_fixture_details"):
            try:
                rf = self._api_provider.get_fixture_details(fixture_id)
            except Exception:
                rf = None

        if rf is None:
            return None

        act_outcome = None
        if rf.status in ("FT", "AET", "PEN", "AWARDED") and rf.home_goals is not None and rf.away_goals is not None:
            hg, ag = int(rf.home_goals), int(rf.away_goals)
            act_outcome = "H" if hg > ag else ("D" if hg == ag else "A")

        return DashboardFixture(
            fixture_id=rf.fixture_id,
            competition_id=rf.league_id,
            competition_name=rf.league_name,
            season_name=str(rf.raw_data.get("season", "2026/2027") if rf.raw_data else "2026/2027"),
            home_team=rf.home_team,
            away_team=rf.away_team,
            home_id=rf.home_team_id,
            away_id=rf.away_team_id,
            scheduled_kickoff=rf.scheduled_kickoff,
            status=rf.status,
            provider="OddAlerts API" if rf.provider == "OddAlerts" else "Local Database",
            home_goals=rf.home_goals,
            away_goals=rf.away_goals,
            actual_outcome=act_outcome,
        )

    def get_previous_week_completed_fixtures(
        self,
        start_date: str = "2026-08-29",
        end_date: str = "2026-09-01",
        provider_name: str = "oddalerts",
        competition_ids: Optional[List[int]] = None,
    ) -> Tuple[List[DashboardFixture], Dict[str, Any]]:
        """Fetch completed fixtures strictly within the previous week UTC date window."""
        comp_ids = set(competition_ids or list(TARGET_LEAGUES.keys()))
        start_dt = datetime.fromisoformat(f"{start_date}T00:00:00").date()
        end_dt = datetime.fromisoformat(f"{end_date}T00:00:00").date()

        date_list = []
        curr = start_dt
        while curr <= end_dt:
            date_list.append(curr.isoformat())
            curr += timedelta(days=1)

        raw_fixtures: List[DashboardFixture] = []
        seen_fids: Set[int] = set()

        for d_str in date_list:
            day_matches, _ = self.get_todays_matches(
                date_str=d_str,
                provider_name=provider_name,
                competition_ids=list(comp_ids),
            )
            for m in day_matches:
                if m.fixture_id not in seen_fids:
                    seen_fids.add(m.fixture_id)
                    raw_fixtures.append(m)

        # Filter strictly to completed matches with known actual goals within [start_date, end_date]
        completed: List[DashboardFixture] = []
        for f in raw_fixtures:
            if f.competition_id not in comp_ids:
                continue
            if f.status not in ("FT", "AET", "PEN", "AWARDED") and (f.home_goals is None or f.away_goals is None):
                continue
            ko_date = f.scheduled_kickoff[:10]
            if ko_date < start_date or ko_date > end_date:
                continue
            completed.append(f)

        # Fallback to persistent offline ledger if provider returned 0
        if not completed:
            offline_ledger = PROJECT_ROOT / "reports" / "previous_week_2026_08_29_to_2026_09_01_ledger.jsonl"
            if offline_ledger.exists():
                try:
                    with open(offline_ledger, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                r = json.loads(line.strip())
                                ko = r.get("scheduled_kickoff_utc", "")
                                if start_date <= ko[:10] <= end_date:
                                    completed.append(DashboardFixture(
                                        fixture_id=r["fixture_id"],
                                        competition_id=r.get("competition_id", 0),
                                        competition_name=r.get("competition_name", "Unknown"),
                                        season_name=r.get("season", "2026/2027"),
                                        home_team=r["home_team"],
                                        away_team=r["away_team"],
                                        scheduled_kickoff=ko,
                                        status="FT",
                                        provider="offline_ledger",
                                        home_goals=r.get("actual_home_goals"),
                                        away_goals=r.get("actual_away_goals"),
                                        actual_outcome=r.get("actual_outcome"),
                                    ))
                except Exception as e:
                    logger.warning(f"Error reading offline previous week ledger: {e}")

        completed.sort(key=lambda x: x.scheduled_kickoff)
        meta = {
            "start_date": start_date,
            "end_date": end_date,
            "total_completed": len(completed),
            "dates_covered": sorted(list({f.scheduled_kickoff[:10] for f in completed})),
            "status": "SUCCESS" if completed else "NO_FIXTURES",
        }
        return (completed, meta)

    def get_previous_week_performance_records(
        self,
        start_date: str = "2026-08-29",
        end_date: str = "2026-09-01",
        provider_name: str = "oddalerts",
        competition_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve pre-kickoff predictions joined with verified outcomes for the previous week."""
        # 1. Try loading directly from verified ledger if available
        offline_ledger = PROJECT_ROOT / "reports" / "previous_week_2026_08_29_to_2026_09_01_ledger.jsonl"
        if offline_ledger.exists() and start_date == "2026-08-29" and end_date == "2026-09-01":
            recs = []
            with open(offline_ledger, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        recs.append(json.loads(line.strip()))
            if recs:
                return recs

        # 2. Dynamic generation joining live/mock completed fixtures with immutable snapshots
        fixtures, _ = self.get_previous_week_completed_fixtures(
            start_date=start_date,
            end_date=end_date,
            provider_name=provider_name,
            competition_ids=competition_ids,
        )

        from dashboard.prediction_snapshot_store import get_prediction_snapshot_store
        store = get_prediction_snapshot_store()

        records: List[Dict[str, Any]] = []
        for f in fixtures:
            snap = store.get_snapshot(f.fixture_id)
            if not snap:
                continue

            hg = int(f.home_goals) if f.home_goals is not None else 0
            ag = int(f.away_goals) if f.away_goals is not None else 0
            act_outcome = "H" if hg > ag else ("D" if hg == ag else "A")
            act_score = f"{hg}-{ag}"

            p_h = float(snap.p_home)
            p_d = float(snap.p_draw)
            p_a = float(snap.p_away)
            dec = str(snap.model_decision)

            lh = float(snap.expected_home_goals)
            la = float(snap.expected_away_goals)
            pred_score = getattr(snap, "canonical_predicted_score", None) or getattr(snap, "predicted_score", None)
            if not pred_score:
                try:
                    import numpy as np
                    from models.poisson import modal_scoreline, _grid_size
                    K = _grid_size(float(max(lh, la)), 1e-4)
                    sh, sa, _ = modal_scoreline(np.array([lh]), np.array([la]), K)
                    pred_score = f"{int(sh[0])}-{int(sa[0])}"
                except Exception:
                    pred_score = f"{max(0, int(round(lh)))}-{max(0, int(round(la)))}"

            d_tier = str(snap.draw_risk_tier or "LOW").upper()
            is_sh = bool(p_h >= 0.60 and d_tier == "LOW")
            is_20 = bool(pred_score == "2-0" and d_tier == "LOW")
            sig_prof = "2-0_PROFILE" if is_20 else ("STRONG_HOME_PROFILE" if is_sh else None)

            is_corr = (dec == act_outcome)
            is_exact = (pred_score == act_score)

            y_h = 1.0 if act_outcome == "H" else 0.0
            y_d = 1.0 if act_outcome == "D" else 0.0
            y_a = 1.0 if act_outcome == "A" else 0.0
            brier = (p_h - y_h)**2 + (p_d - y_d)**2 + (p_a - y_a)**2
            p_act = p_h if act_outcome == "H" else (p_d if act_outcome == "D" else p_a)
            log_loss = -math.log(max(1e-15, p_act))

            k_utc = f.scheduled_kickoff
            date_str = k_utc[:10]
            time_str = k_utc[11:16] if len(k_utc) >= 16 else "--"

            badge = "🟢 LOW" if d_tier == "LOW" else ("🟡 MEDIUM" if d_tier in ("MEDIUM", "MODERATE") else "🟠 HIGH")

            records.append({
                "fixture_id": f.fixture_id,
                "competition_name": f.competition_name,
                "competition_id": f.competition_id,
                "home_team": f.home_team,
                "away_team": f.away_team,
                "scheduled_kickoff_utc": k_utc,
                "Date (UTC)": date_str,
                "Kickoff (UTC)": time_str,
                "League": f.competition_name,
                "Home Team": f.home_team,
                "Away Team": f.away_team,
                "p_home": round(p_h, 3),
                "p_draw": round(p_d, 3),
                "p_away": round(p_a, 3),
                "P(H)": f"{p_h*100:.1f}%",
                "P(D)": f"{p_d*100:.1f}%",
                "P(A)": f"{p_a*100:.1f}%",
                "predicted_outcome": dec,
                "V4.0 Pred": dec,
                "canonical_predicted_score": pred_score,
                "baseline_predicted_score": pred_score,
                "Predicted Score": pred_score,
                "strong_home_profile": is_sh,
                "is_2_0_profile": is_20,
                "signal_profile": sig_prof,
                "Signal": "🟢 2-0 Signal" if is_20 else ("🔵 Strong Home" if is_sh else "--"),
                "expected_home_goals": round(lh, 2),
                "expected_away_goals": round(la, 2),
                "xG": f"{lh:.2f}-{la:.2f}",
                "actual_home_goals": hg,
                "actual_away_goals": ag,
                "actual_score": act_score,
                "Final Score": act_score,
                "actual_outcome": act_outcome,
                "Actual Outcome": act_outcome,
                "prediction_correct": is_corr,
                "Evaluation": "✅ CORRECT" if is_corr else "❌ WRONG",
                "exact_score_correct": is_exact,
                "Exact Score": "🎯 HIT" if is_exact else "--",
                "draw_risk_tier": d_tier,
                "Draw Risk": badge,
                "draw_risk_score": round(float(snap.draw_risk_score or p_d), 3),
                "memory_evidence_level": "INSUFFICIENT",
                "Historical Memory": "⚪ INSUFFICIENT",
                "brier_score": round(brier, 4),
                "log_loss": round(log_loss, 4),
            })

        return records

