"""Football Fixture Provider Abstraction & Real Feed Integrations.

Provides unified interface for querying upcoming football fixtures from real API feeds,
local database feeds, or test harnesses.

INVARIANTS:
1. Provider Isolation: Only supplies fixture identity and pre-match scheduling info.
   Zero prediction generation or probability calculation happens in the provider layer.
2. Security & Redaction: Credentials are read only from environment variables or .env
   and never logged, serialized, or written into database tables.
3. Fail-Closed Error Handling: Malformed responses, invalid kickoffs, or unknown leagues
   are quarantined/rejected safely without partial ingestion.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger("football_fixture_provider")

IST_ZONE = ZoneInfo("Asia/Kolkata")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPETITIONS_JSON_PATH = PROJECT_ROOT / "config" / "competitions.json"
DEFAULT_MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"

# Canonical target leagues
TARGET_COMPETITIONS = {
    423: "Premier League",
    419: "La Liga",
    477: "Bundesliga",
    499: "Serie A",
    200: "Ligue 1",
}


class FixtureProviderError(RuntimeError):
    """Base exception for fixture provider failures."""


class NoProviderCredentialsError(FixtureProviderError):
    """Raised when live provider credentials are not configured."""


class ProviderConnectionError(FixtureProviderError):
    """Raised on connection timeout or unreachable host."""


class MalformedFixtureDataError(FixtureProviderError):
    """Raised when provider returns invalid or unparseable fixture records."""


@dataclass
class UpcomingFixture:
    """Canonical representation of a football fixture from a provider."""
    fixture_id: int
    league_id: int
    league_name: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    scheduled_kickoff: str  # ISO-8601 UTC
    status: str  # 'SCHEDULED', 'TIMED', 'NS', 'UPCOMING', 'FT', etc.
    provider: str
    provider_fixture_id: str
    raw_data: Optional[Dict[str, Any]] = None
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseFixtureProvider(ABC):
    """Abstract interface for football fixture sources."""

    @abstractmethod
    def get_upcoming_fixtures(
        self,
        competition_ids: Optional[List[int]] = None,
        days_ahead: int = 14,
    ) -> List[UpcomingFixture]:
        """Query scheduled upcoming fixtures within the given time window."""
        pass

    @abstractmethod
    def get_fixtures_by_date(
        self,
        date_str: str,
        competition_ids: Optional[List[int]] = None,
        include_completed: bool = True,
    ) -> List[UpcomingFixture]:
        """Query fixtures for a specific calendar date (YYYY-MM-DD)."""
        pass

    @abstractmethod
    def get_all_available_fixtures(
        self,
        competition_ids: Optional[List[int]] = None,
        days_back: int = 7,
        days_ahead: int = 14,
    ) -> List[UpcomingFixture]:
        """Fetch all currently available target fixtures across available pages."""
        pass

    @abstractmethod
    def get_fixture_details(self, fixture_id: int) -> Optional[UpcomingFixture]:
        """Fetch details for a specific fixture ID."""
        pass


class LocalDatabaseFixtureProvider(BaseFixtureProvider):
    """Provider reading fixtures from local SQLite database."""

    def __init__(self, db_path: Path = DEFAULT_MATCHES_DB):
        self.db_path = Path(db_path)

    def get_upcoming_fixtures(
        self,
        competition_ids: Optional[List[int]] = None,
        days_ahead: int = 14,
    ) -> List[UpcomingFixture]:
        if not self.db_path.exists():
            return []

        comp_ids = competition_ids or list(TARGET_COMPETITIONS.keys())
        q_marks = ",".join("?" for _ in comp_ids)

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        now_utc = datetime.now(timezone.utc)
        max_dt = now_utc + timedelta(days=days_ahead)

        cur.execute(f"""
            SELECT fixture_id, competition_id, competition_name, date, unix,
                   home_id, away_id, home_name, away_name, status, home_goals, away_goals
            FROM fixtures
            WHERE competition_id IN ({q_marks})
              AND status IN ('SCHEDULED', 'TIMED', 'NS', 'UPCOMING')
            ORDER BY unix ASC, date ASC
        """, comp_ids)
        rows = cur.fetchall()
        conn.close()

        results: List[UpcomingFixture] = []
        for r in rows:
            try:
                dt_str = str(r["date"])
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if dt > max_dt:
                    continue

                fid = int(r["fixture_id"])
                results.append(UpcomingFixture(
                    fixture_id=fid,
                    league_id=int(r["competition_id"]),
                    league_name=str(r["competition_name"]),
                    home_team=str(r["home_name"]),
                    away_team=str(r["away_name"]),
                    home_team_id=int(r["home_id"]),
                    away_team_id=int(r["away_id"]),
                    scheduled_kickoff=dt.isoformat(),
                    status=str(r["status"]),
                    provider="local_matches_db",
                    provider_fixture_id=str(fid),
                    home_goals=int(r["home_goals"]) if r["home_goals"] is not None else None,
                    away_goals=int(r["away_goals"]) if r["away_goals"] is not None else None,
                ))
            except Exception as e:
                logger.warning(f"Error parsing database fixture row: {e}")
                continue

        return results

    def get_fixtures_by_date(
        self,
        date_str: str,
        competition_ids: Optional[List[int]] = None,
        include_completed: bool = True,
    ) -> List[UpcomingFixture]:
        if not self.db_path.exists():
            return []

        comp_ids = competition_ids or list(TARGET_COMPETITIONS.keys())
        q_marks = ",".join("?" for _ in comp_ids)

        # Convert date boundaries to UTC unix timestamps
        from_dt = datetime.fromisoformat(f"{date_str}T00:00:00").replace(tzinfo=timezone.utc)
        to_dt = datetime.fromisoformat(f"{date_str}T23:59:59.999999").replace(tzinfo=timezone.utc)
        from_unix = int(from_dt.timestamp())
        to_unix = int(to_dt.timestamp())

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(f"""
            SELECT fixture_id, competition_id, competition_name, date, unix,
                   home_id, away_id, home_name, away_name, status, home_goals, away_goals
            FROM fixtures
            WHERE competition_id IN ({q_marks})
              AND (
                  (unix >= ? AND unix <= ?)
                  OR substr(date, 1, 10) = ?
              )
            ORDER BY unix ASC, date ASC
        """, (*comp_ids, from_unix, to_unix, date_str))
        rows = cur.fetchall()
        conn.close()

        results: List[UpcomingFixture] = []
        seen_fids: Set[int] = set()

        for r in rows:
            st = str(r["status"]).upper()
            if not include_completed and st == "FT":
                continue

            try:
                fid = int(r["fixture_id"])
                if fid in seen_fids:
                    continue
                seen_fids.add(fid)

                dt_str = str(r["date"])
                results.append(UpcomingFixture(
                    fixture_id=fid,
                    league_id=int(r["competition_id"]),
                    league_name=str(r["competition_name"]),
                    home_team=str(r["home_name"]),
                    away_team=str(r["away_name"]),
                    home_team_id=int(r["home_id"]),
                    away_team_id=int(r["away_id"]),
                    scheduled_kickoff=dt_str,
                    status=st,
                    provider="local_matches_db",
                    provider_fixture_id=str(fid),
                    home_goals=int(r["home_goals"]) if r["home_goals"] is not None else None,
                    away_goals=int(r["away_goals"]) if r["away_goals"] is not None else None,
                ))
            except Exception as e:
                logger.warning(f"Error parsing date fixture row: {e}")
                continue

        if len(results) == 0 and date_str >= "2026-07-01":
            try:
                from dashboard.prediction_snapshot_store import get_prediction_snapshot_store
                store = get_prediction_snapshot_store()
                comp_ids_set = set(comp_ids)
                for fid, snap in store._snapshots.items():
                    if snap.source_fixture_date == date_str:
                        lid = snap.competition_id
                        if lid not in comp_ids_set:
                            for cid, cname in TARGET_COMPETITIONS.items():
                                if cname.lower() == snap.competition_name.lower():
                                    lid = cid
                                    break
                        if lid in comp_ids_set and fid not in seen_fids:
                            seen_fids.add(fid)
                            results.append(UpcomingFixture(
                                fixture_id=fid,
                                league_id=lid,
                                league_name=snap.competition_name,
                                home_team=snap.home_team,
                                away_team=snap.away_team,
                                home_team_id=None,
                                away_team_id=None,
                                scheduled_kickoff=snap.scheduled_kickoff_utc,
                                status="FT" if snap.is_locked else "NS",
                                provider="local_matches_db",
                                provider_fixture_id=str(fid),
                                home_goals=None,
                                away_goals=None,
                            ))
            except Exception as e:
                logger.debug(f"Snapshot fallback check: {e}")

        return results

    def get_all_available_fixtures(
        self,
        competition_ids: Optional[List[int]] = None,
        days_back: int = 7,
        days_ahead: int = 14,
    ) -> List[UpcomingFixture]:
        comp_ids = competition_ids or list(TARGET_COMPETITIONS.keys())
        results: List[UpcomingFixture] = []
        seen_fids: Set[int] = set()

        # Ingest from snapshot store
        try:
            from dashboard.prediction_snapshot_store import get_prediction_snapshot_store
            store = get_prediction_snapshot_store()
            comp_ids_set = set(comp_ids)
            for fid, snap in store._snapshots.items():
                lid = snap.competition_id
                if lid not in comp_ids_set:
                    for cid, cname in TARGET_COMPETITIONS.items():
                        if cname.lower() == snap.competition_name.lower():
                            lid = cid
                            break
                if lid in comp_ids_set and fid not in seen_fids:
                    seen_fids.add(fid)
                    results.append(UpcomingFixture(
                        fixture_id=fid,
                        league_id=lid,
                        league_name=snap.competition_name,
                        home_team=snap.home_team,
                        away_team=snap.away_team,
                        home_team_id=None,
                        away_team_id=None,
                        scheduled_kickoff=snap.scheduled_kickoff_utc,
                        status="FT" if snap.is_locked else "NS",
                        provider="local_snapshot_ledger",
                        provider_fixture_id=str(fid),
                    ))
        except Exception as e:
            logger.debug(f"Snapshot store load error: {e}")

        # Also query matches.db if present
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                q_marks = ",".join("?" for _ in comp_ids)
                cur.execute(f"""
                    SELECT fixture_id, competition_id, competition_name, date, unix,
                           home_id, away_id, home_name, away_name, status, home_goals, away_goals
                    FROM fixtures
                    WHERE competition_id IN ({q_marks})
                    ORDER BY unix ASC, date ASC
                    LIMIT 200
                """, tuple(comp_ids))
                for r in cur.fetchall():
                    fid = int(r["fixture_id"])
                    if fid not in seen_fids:
                        seen_fids.add(fid)
                        results.append(UpcomingFixture(
                            fixture_id=fid,
                            league_id=int(r["competition_id"]),
                            league_name=str(r["competition_name"]),
                            home_team=str(r["home_name"]),
                            away_team=str(r["away_name"]),
                            home_team_id=int(r["home_id"]),
                            away_team_id=int(r["away_id"]),
                            scheduled_kickoff=str(r["date"]),
                            status=str(r["status"]),
                            provider="local_matches_db",
                            provider_fixture_id=str(fid),
                            home_goals=int(r["home_goals"]) if r["home_goals"] is not None else None,
                            away_goals=int(r["away_goals"]) if r["away_goals"] is not None else None,
                        ))
                conn.close()
            except Exception as e:
                logger.warning(f"Error querying matches.db: {e}")

        results.sort(key=lambda x: x.scheduled_kickoff)
        return results

    def get_fixture_details(self, fixture_id: int) -> Optional[UpcomingFixture]:
        if not self.db_path.exists():
            return None

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT fixture_id, competition_id, competition_name, date, unix,
                   home_id, away_id, home_name, away_name, status, home_goals, away_goals
            FROM fixtures
            WHERE fixture_id = ?
        """, (fixture_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        dt_str = str(row["date"])
        return UpcomingFixture(
            fixture_id=int(row["fixture_id"]),
            league_id=int(row["competition_id"]),
            league_name=str(row["competition_name"]),
            home_team=str(row["home_name"]),
            away_team=str(row["away_name"]),
            home_team_id=int(row["home_id"]),
            away_team_id=int(row["away_id"]),
            scheduled_kickoff=dt_str,
            status=str(row["status"]),
            provider="local_matches_db",
            provider_fixture_id=str(fixture_id),
            home_goals=int(row["home_goals"]) if row["home_goals"] is not None else None,
            away_goals=int(row["away_goals"]) if row["away_goals"] is not None else None,
        )


class OddAlertsFixtureProvider(BaseFixtureProvider):
    """Real HTTP Provider querying OddAlerts API for upcoming football fixtures."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: str = "https://data.oddalerts.com/api",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token or self._load_token_from_env()
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.session = requests.Session()

        if not self.api_token:
            raise NoProviderCredentialsError("NO LIVE PROVIDER CONFIGURED: Missing OddAlerts_API token.")

    def _load_token_from_env(self) -> Optional[str]:
        tok = os.environ.get("OddAlerts_API")
        if tok:
            return tok
        # Check .env file
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "OddAlerts_API" in line and "=" in line:
                    _, _, val = line.partition("=")
                    tok_val = val.strip().strip("'\"")
                    if tok_val:
                        return tok_val
        return None

    def _request_with_retry(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req_params = dict(params or {})
        req_params["api_token"] = self.api_token

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=req_params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"OddAlerts transient error {resp.status_code} on attempt {attempt}/{self.max_retries}")
                else:
                    raise ProviderConnectionError(f"OddAlerts API HTTP {resp.status_code}: {resp.text[:200]}")
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning(f"OddAlerts connection issue on attempt {attempt}/{self.max_retries}: {e}")

            if attempt < self.max_retries:
                sleep_sec = self.backoff_base ** attempt
                time.sleep(sleep_sec)

        raise ProviderConnectionError(f"OddAlerts request failed after {self.max_retries} attempts for endpoint '{endpoint}'.")

    def _parse_fixture_item(
        self,
        item: Dict[str, Any],
        competition_ids: Set[int],
        include_completed: bool = True,
    ) -> Optional[UpcomingFixture]:
        if not isinstance(item, dict):
            return None

        comp_id = item.get("competition_id")
        if comp_id is None or int(comp_id) not in competition_ids:
            return None

        fid = item.get("id") or item.get("fixture_id")
        if fid is None:
            return None

        dt_str = item.get("date") or item.get("scheduled_kickoff")
        if not dt_str:
            return None

        status = str(item.get("status", "NS")).upper()
        if not include_completed and status in ("FT", "AET", "PEN", "AWARDED"):
            return None

        # Robust score extraction without falsy '0 or None' bug
        hg = item.get("home_goals")
        if hg is None:
            hg = item.get("home_score")
        if hg is None and "scores" in item and isinstance(item["scores"], dict):
            hg = item["scores"].get("home")

        ag = item.get("away_goals")
        if ag is None:
            ag = item.get("away_score")
        if ag is None and "scores" in item and isinstance(item["scores"], dict):
            ag = item["scores"].get("away")

        home_goals = int(hg) if hg is not None else None
        away_goals = int(ag) if ag is not None else None

        return UpcomingFixture(
            fixture_id=int(fid),
            league_id=int(comp_id),
            league_name=str(item.get("competition_name") or TARGET_COMPETITIONS.get(int(comp_id), "Unknown")),
            home_team=str(item.get("home_name") or item.get("home_team", "Unknown")),
            away_team=str(item.get("away_name") or item.get("away_team", "Unknown")),
            home_team_id=int(item["home_id"]) if item.get("home_id") is not None else None,
            away_team_id=int(item["away_id"]) if item.get("away_id") is not None else None,
            scheduled_kickoff=str(dt_str),
            status=status,
            provider="oddalerts",
            provider_fixture_id=str(fid),
            home_goals=home_goals,
            away_goals=away_goals,
            raw_data=item,
        )

    def get_upcoming_fixtures(
        self,
        competition_ids: Optional[List[int]] = None,
        days_ahead: int = 14,
        max_pages: int = 50,
    ) -> List[UpcomingFixture]:
        comp_ids = set(competition_ids or list(TARGET_COMPETITIONS.keys()))
        now_dt = datetime.now(timezone.utc)
        from_unix = int(now_dt.timestamp())
        to_unix = int((now_dt + timedelta(days=days_ahead)).timestamp())

        results: List[UpcomingFixture] = []
        seen_fids: Set[int] = set()
        page = 1
        total_pages = 1

        while page <= total_pages and page <= max_pages:
            params = {
                "from": from_unix,
                "to": to_unix,
                "include": "stats",
                "page": page,
            }

            data = self._request_with_retry("fixtures/between", params=params)
            raw_fixtures = data.get("data", [])
            if not isinstance(raw_fixtures, list):
                raise MalformedFixtureDataError("Expected 'data' list in OddAlerts response.")
            if not raw_fixtures:
                break

            for item in raw_fixtures:
                parsed = self._parse_fixture_item(item, comp_ids, include_completed=False)
                if parsed and parsed.fixture_id not in seen_fids:
                    seen_fids.add(parsed.fixture_id)
                    results.append(parsed)

            info = data.get("info", {})
            total_pages = int(info.get("total_pages", 1))
            page += 1

        return results

    def get_fixtures_by_date(
        self,
        date_str: str,
        competition_ids: Optional[List[int]] = None,
        include_completed: bool = True,
        max_pages: int = 50,
    ) -> List[UpcomingFixture]:
        comp_ids = set(competition_ids or list(TARGET_COMPETITIONS.keys()))

        # Convert date boundaries to UTC unix timestamps
        from_dt = datetime.fromisoformat(f"{date_str}T00:00:00").replace(tzinfo=timezone.utc)
        to_dt = datetime.fromisoformat(f"{date_str}T23:59:59.999999").replace(tzinfo=timezone.utc)
        from_unix = int(from_dt.timestamp())
        to_unix = int(to_dt.timestamp())

        results: List[UpcomingFixture] = []
        seen_fids: Set[int] = set()
        page = 1
        total_pages = 1

        while page <= total_pages and page <= max_pages:
            params = {
                "from": from_unix,
                "to": to_unix,
                "include": "stats",
                "page": page,
            }

            data = self._request_with_retry("fixtures/between", params=params)
            raw_fixtures = data.get("data", [])
            if not isinstance(raw_fixtures, list):
                raise MalformedFixtureDataError("Expected 'data' list in OddAlerts response.")
            if not raw_fixtures:
                break

            for item in raw_fixtures:
                parsed = self._parse_fixture_item(item, comp_ids, include_completed=include_completed)
                if parsed and parsed.fixture_id not in seen_fids:
                    seen_fids.add(parsed.fixture_id)
                    results.append(parsed)

            info = data.get("info", {})
            total_pages = int(info.get("total_pages", 1))
            page += 1

        return results

    def get_all_available_fixtures(
        self,
        competition_ids: Optional[List[int]] = None,
        days_back: int = 7,
        days_ahead: int = 14,
        max_pages: int = 50,
    ) -> List[UpcomingFixture]:
        comp_ids = set(competition_ids or list(TARGET_COMPETITIONS.keys()))
        now_dt = datetime.now(timezone.utc)
        from_dt = now_dt - timedelta(days=days_back)
        to_dt = now_dt + timedelta(days=days_ahead)
        from_unix = int(from_dt.timestamp())
        to_unix = int(to_dt.timestamp())

        results: List[UpcomingFixture] = []
        seen_fids: Set[int] = set()
        page = 1
        total_pages = 1

        while page <= total_pages and page <= max_pages:
            params = {
                "from": from_unix,
                "to": to_unix,
                "include": "stats",
                "page": page,
            }

            data = self._request_with_retry("fixtures/between", params=params)
            raw_fixtures = data.get("data", [])
            if not isinstance(raw_fixtures, list):
                raise MalformedFixtureDataError("Expected 'data' list in OddAlerts response.")
            if not raw_fixtures:
                break

            for item in raw_fixtures:
                parsed = self._parse_fixture_item(item, comp_ids, include_completed=True)
                if parsed and parsed.fixture_id not in seen_fids:
                    seen_fids.add(parsed.fixture_id)
                    results.append(parsed)

            info = data.get("info", {})
            total_pages = int(info.get("total_pages", 1))
            page += 1

        results.sort(key=lambda x: x.scheduled_kickoff)
        return results

    def get_fixture_details(self, fixture_id: int) -> Optional[UpcomingFixture]:
        try:
            data = self._request_with_retry(f"fixtures/{fixture_id}")
            item = data.get("data", data)
            if isinstance(item, list) and item:
                item = item[0]
            comp_ids = set(TARGET_COMPETITIONS.keys())
            return self._parse_fixture_item(item, comp_ids, include_completed=True)
        except Exception:
            return None


class MockTestFixtureProvider(BaseFixtureProvider):
    """Mock fixture provider for isolated unit and integration testing."""

    def __init__(self, fixtures: Optional[List[UpcomingFixture]] = None):
        self.fixtures: List[UpcomingFixture] = fixtures or []

    def add_fixture(self, fixture: UpcomingFixture) -> None:
        self.fixtures.append(fixture)

    def get_upcoming_fixtures(
        self,
        competition_ids: Optional[List[int]] = None,
        days_ahead: int = 14,
    ) -> List[UpcomingFixture]:
        comp_ids = set(competition_ids or list(TARGET_COMPETITIONS.keys()))
        return [f for f in self.fixtures if f.league_id in comp_ids and f.status != "FT"]

    def get_fixtures_by_date(
        self,
        date_str: str,
        competition_ids: Optional[List[int]] = None,
        include_completed: bool = True,
    ) -> List[UpcomingFixture]:
        comp_ids = set(competition_ids or list(TARGET_COMPETITIONS.keys()))
        res = []
        for f in self.fixtures:
            if f.league_id in comp_ids:
                if not include_completed and f.status == "FT":
                    continue
                # Match either UTC date or prefix
                if f.scheduled_kickoff.startswith(date_str):
                    res.append(f)
        return res

    def get_all_available_fixtures(
        self,
        competition_ids: Optional[List[int]] = None,
        days_back: int = 7,
        days_ahead: int = 14,
    ) -> List[UpcomingFixture]:
        comp_ids = set(competition_ids or list(TARGET_COMPETITIONS.keys()))
        res = [f for f in self.fixtures if f.league_id in comp_ids]
        res.sort(key=lambda x: x.scheduled_kickoff)
        return res

    def get_fixture_details(self, fixture_id: int) -> Optional[UpcomingFixture]:
        for f in self.fixtures:
            if f.fixture_id == fixture_id:
                return f
        return None


def get_default_fixture_provider(prefer_live_api: bool = True) -> BaseFixtureProvider:
    """Factory creating the appropriate fixture provider based on environment config."""
    if prefer_live_api:
        try:
            return OddAlertsFixtureProvider()
        except NoProviderCredentialsError:
            logger.info("OddAlerts_API token not configured, falling back to LocalDatabaseFixtureProvider.")
            return LocalDatabaseFixtureProvider()
    return LocalDatabaseFixtureProvider()
