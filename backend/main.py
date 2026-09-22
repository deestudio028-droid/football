"""FastAPI Live Score Updating Backend for Football Prediction Dashboard.

Architecture:
- OddAlerts API -> FastAPI Backend (Railway) -> Cached /api/live-scores -> Frontend (index.html)
- Implements 20-second server-side TTL cache to safeguard OddAlerts 300 req/min quota.
- Strict token security: API key is loaded only from environment variable (OddAlerts_API),
  never exposed in responses, and redacted from all error logs.
- Preserves 100% data integrity for frozen pre-kickoff predictions.
- Persistent completed scores cache: once a match finishes (FT/AET/PEN), its score is
  permanently retained and never reverts to UPCOMING or blank.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging with redaction
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("live_scores_backend")

ODD_ALERTS_BASE_URL = "https://data.oddalerts.com/api"
DEFAULT_CACHE_TTL_SECONDS = 20.0

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
COMPLETED_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "completed_scores_cache.json"
OUTCOME_LEDGER_PATH = Path(__file__).resolve().parent.parent / "research" / "external_consensus" / "phase13" / "02_outcome_ledger.jsonl"

# Canonical Phase 15 fixtures (2026-09-18 -> 2026-09-21)
DEFAULT_PHASE15_48_FIXTURE_IDS = [
    420656757, 420657505, 420656763, 420654561, 420656764, 420657711, 420654558,
    420657746, 420657745, 420657833, 420657834, 420657832, 420657831, 420657845,
    420657844, 420657841, 420654607, 420657503, 420657949, 420658031, 420654568,
    420658030, 420657509, 420657507, 420657506, 420657502, 420658129, 420654582,
    420659043, 420654594, 420657508, 420659114, 420657843, 420657842, 420659113,
    420659112, 420659393, 420654564, 420657504, 420659447, 420659446, 420659455,
    420654708, 420654580, 420659463, 420659485, 420659484, 420654586
]

# Canonical 55 fixtures from earlier upcoming batch (2026-09-10 -> 2026-09-16)
DEFAULT_55_FIXTURE_IDS = [
    420629293, 420629297, 420629296, 420636977, 420636980, 420634429, 420634488,
    420634485, 420634484, 420634486, 420634487, 420634512, 420634511, 420634510,
    420634509, 420634508, 420636981, 420634716, 420638007, 420634771, 420636985,
    420634770, 420634862, 420634861, 420634860, 420634859, 420634858, 420636979,
    420634863, 420636984, 420637600, 420637601, 420637598, 420637864, 420636982,
    420637968, 420637973, 420637972, 420637982, 420636983, 420638008, 420634723,
    420636978, 420637599, 420637534, 420644169, 420636976, 420644172, 420644690,
    420644698, 420644694, 420644697, 420644695, 420644693, 420644692
]

# Canonical Oct 9-13, 2026 fixtures (48 fixtures)
DEFAULT_OCT_48_FIXTURE_IDS = [
    420679102, 420660812, 420654657, 420665726, 420654552, 420669055, 420678999,
    420679178, 420679177, 420679170, 420679131, 420665971, 420665977, 420665902,
    420665723, 420654587, 420660809, 420668930, 420679145, 420654560, 420665724,
    420669040, 420660811, 420660813, 420660810, 420660806, 420654590, 420668928,
    420654608, 420669061, 420668931, 420665875, 420665914, 420660808, 420679118,
    420654705, 420660888, 420665834, 420678977, 420668934, 420654530, 420669035,
    420660807, 420654589, 420668921, 420668929, 420665848, 420654559
]


def load_target_fixture_ids() -> List[int]:
    """Load target fixture IDs from all upcoming batch ledgers, falling back to static lists."""
    fids_set: Dict[int, None] = {}

    # Check reports directory for any upcoming ledgers
    if REPORTS_DIR.exists():
        for ledger_file in sorted(REPORTS_DIR.glob("upcoming_*ledger.jsonl")):
            try:
                with open(ledger_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if "fixture_id" in data:
                                fids_set[int(data["fixture_id"])] = None
            except Exception as exc:
                logger.warning(f"Could not read fixture IDs from {ledger_file}: {exc}")

    # Ensure all Phase 15, earlier batch, and Oct fixtures are included
    for fid in DEFAULT_PHASE15_48_FIXTURE_IDS + DEFAULT_55_FIXTURE_IDS + DEFAULT_OCT_48_FIXTURE_IDS:
        fids_set[fid] = None

    return list(fids_set.keys())


def load_api_token() -> Optional[str]:
    """Load OddAlerts API token exclusively from environment, with .env fallback for local dev."""
    tok = os.environ.get("OddAlerts_API")
    if tok and tok.strip():
        return tok.strip()

    # Local development fallback: check .env files
    for search_dir in [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
        env_file = search_dir / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "OddAlerts_API" in line and "=" in line:
                        _, _, val = line.partition("=")
                        cand = val.strip().strip("'\"")
                        if cand:
                            return cand
            except Exception:
                pass
    return None


def redact(text: str, token: Optional[str] = None) -> str:
    """Safely redact API token from log messages, URLs, and exception strings."""
    if not text:
        return text
    if token and token in text:
        text = text.replace(token, "[REDACTED_API_TOKEN]")
    text = re.sub(r"api_token=[a-zA-Z0-9_\-]+", "api_token=[REDACTED_API_TOKEN]", text)
    return text


class FixtureScoreItem(BaseModel):
    fixture_id: int
    status: str
    status_label: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    display_score: str = "—"
    elapsed: Optional[int] = None
    time_added: Optional[int] = None
    is_finished: bool = False


class LiveScoresResponse(BaseModel):
    last_updated_utc: str
    active_fixtures_count: int
    cached: bool = False
    scores: Dict[str, FixtureScoreItem]


def dump_item(item: FixtureScoreItem) -> Dict[str, Any]:
    return item.model_dump() if hasattr(item, "model_dump") else item.dict()


class LiveScoreManager:
    """Manages upstream OddAlerts queries with in-memory caching, rate-limit defense and retry."""

    def __init__(self, cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS):
        self.cache_ttl = cache_ttl
        self.fixture_ids = load_target_fixture_ids()
        self._cache_lock = threading.Lock()
        self._cached_data: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0.0
        self._completed_cache: Dict[str, FixtureScoreItem] = {}
        self._session = requests.Session()
        self._load_completed_cache()

    def _load_completed_cache(self) -> None:
        """Load known finished match scores from persistent cache file and outcome ledgers."""
        # 1. Load from data/completed_scores_cache.json
        if COMPLETED_CACHE_PATH.exists():
            try:
                with open(COMPLETED_CACHE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for str_fid, item_data in data.items():
                    self._completed_cache[str(str_fid)] = FixtureScoreItem(**item_data)
                logger.info(f"Loaded {len(self._completed_cache)} completed scores from disk cache.")
            except Exception as exc:
                logger.warning(f"Error loading completed scores cache from {COMPLETED_CACHE_PATH}: {exc}")

        # 2. Ingest from phase13 outcome ledger if available
        if OUTCOME_LEDGER_PATH.exists():
            try:
                with open(OUTCOME_LEDGER_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("match_status") in ("FT", "AET", "PEN") and rec.get("actual_home_score") is not None and rec.get("actual_away_score") is not None:
                            fid = int(rec["fixture_id"])
                            str_fid = str(fid)
                            if str_fid not in self._completed_cache:
                                hg = int(rec["actual_home_score"])
                                ag = int(rec["actual_away_score"])
                                item = FixtureScoreItem(
                                    fixture_id=fid,
                                    status="FT",
                                    status_label="FINISHED",
                                    home_goals=hg,
                                    away_goals=ag,
                                    display_score=f"{hg}-{ag}",
                                    elapsed=90,
                                    time_added=None,
                                    is_finished=True,
                                )
                                self._completed_cache[str_fid] = item
                logger.info(f"Ingested completed outcomes from ledger. Total completed in cache: {len(self._completed_cache)}")
            except Exception as exc:
                logger.warning(f"Error loading from outcome ledger {OUTCOME_LEDGER_PATH}: {exc}")

    def _save_completed_cache(self) -> None:
        """Persist completed scores to disk."""
        try:
            COMPLETED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            serializable = {k: dump_item(v) for k, v in self._completed_cache.items()}
            with open(COMPLETED_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not persist completed scores cache: {exc}")

    def _normalize_fixture(self, raw_item: Dict[str, Any]) -> FixtureScoreItem:
        fid = int(raw_item.get("id") or raw_item.get("fixture_id"))
        raw_status = str(raw_item.get("status", "NS")).upper()

        hg = raw_item.get("home_goals")
        if hg is None:
            hg = raw_item.get("home_score")
        if hg is None and isinstance(raw_item.get("scores"), dict):
            hg = raw_item["scores"].get("home")

        ag = raw_item.get("away_goals")
        if ag is None:
            ag = raw_item.get("away_score")
        if ag is None and isinstance(raw_item.get("scores"), dict):
            ag = raw_item["scores"].get("away")

        home_goals = int(hg) if hg is not None else None
        away_goals = int(ag) if ag is not None else None

        elapsed = raw_item.get("elapsed")
        elapsed = int(elapsed) if elapsed is not None else None

        time_added = raw_item.get("time_added")
        time_added = int(time_added) if time_added is not None else None

        # Status & Label Normalization
        if raw_status in ("FT", "AET", "PEN"):
            norm_status = raw_status
            norm_label = "FINISHED" if raw_status == "FT" else raw_status
            score_disp = f"{home_goals}-{away_goals}" if home_goals is not None and away_goals is not None else "—"
            is_finished = True
            if home_goals is not None and away_goals is not None:
                item = FixtureScoreItem(
                    fixture_id=fid,
                    status=norm_status,
                    status_label=norm_label,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    display_score=score_disp,
                    elapsed=elapsed or 90,
                    time_added=time_added,
                    is_finished=True,
                )
                self._completed_cache[str(fid)] = item
                return item
        else:
            # Check if this fixture was already completed in cache
            if str(fid) in self._completed_cache:
                return self._completed_cache[str(fid)]

        if raw_status == "LIVE":
            norm_status = "LIVE"
            if elapsed is not None:
                if time_added:
                    norm_label = f"🔴 LIVE {elapsed}+{time_added}'"
                else:
                    norm_label = f"🔴 LIVE {elapsed}'"
            else:
                norm_label = "🔴 LIVE"
            hg_val = home_goals if home_goals is not None else 0
            ag_val = away_goals if away_goals is not None else 0
            score_disp = f"{hg_val}-{ag_val}"
            is_finished = False
        elif raw_status == "HT":
            norm_status = "HT"
            norm_label = "⏸ HT"
            hg_val = home_goals if home_goals is not None else 0
            ag_val = away_goals if away_goals is not None else 0
            score_disp = f"{hg_val}-{ag_val}"
            is_finished = False
        elif raw_status == "POSTP":
            norm_status = "POSTP"
            norm_label = "POSTPONED"
            score_disp = "—"
            is_finished = False
        elif raw_status == "CANC":
            norm_status = "CANC"
            norm_label = "CANCELLED"
            score_disp = "—"
            is_finished = False
        elif raw_status == "SUSP":
            norm_status = "SUSP"
            norm_label = "SUSPENDED"
            score_disp = f"{home_goals}-{away_goals}" if home_goals is not None and away_goals is not None else "—"
            is_finished = False
        else:  # NS, TIMED, UPCOMING, SCHEDULED
            norm_status = "UPCOMING"
            norm_label = "UPCOMING"
            score_disp = "—"
            is_finished = False

        return FixtureScoreItem(
            fixture_id=fid,
            status=norm_status,
            status_label=norm_label,
            home_goals=home_goals,
            away_goals=away_goals,
            display_score=score_disp,
            elapsed=elapsed,
            time_added=time_added,
            is_finished=is_finished,
        )

    def _build_fallback_item(self, fid: int) -> FixtureScoreItem:
        if str(fid) in self._completed_cache:
            return self._completed_cache[str(fid)]
        return FixtureScoreItem(
            fixture_id=fid,
            status="UPCOMING",
            status_label="UPCOMING",
            home_goals=None,
            away_goals=None,
            display_score="—",
            elapsed=None,
            time_added=None,
            is_finished=False,
        )

    def fetch_live_scores(self, token: Optional[str] = None, requested_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Query OddAlerts using cached bulk endpoint with retry, circuit protection, and completed match retention."""
        now = time.time()

        target_ids = list(self.fixture_ids)
        if requested_ids:
            for rid in requested_ids:
                if rid not in target_ids:
                    target_ids.append(rid)

        with self._cache_lock:
            # Check cache freshness
            if self._cached_data is not None and (now - self._cache_timestamp) < self.cache_ttl:
                cached_scores = self._cached_data.get("scores", {})
                if all(str(fid) in cached_scores for fid in target_ids):
                    cached_copy = dict(self._cached_data)
                    cached_copy["cached"] = True
                    return cached_copy

        # Upstream query needed
        tok = token or load_api_token()
        if not tok:
            logger.warning("No OddAlerts API token available; returning cached/fallback fixtures.")
            return self._generate_fallback_response("NO_TOKEN_CONFIGURED", target_ids=target_ids)

        ids_param = ",".join(str(fid) for fid in target_ids)
        url = f"{ODD_ALERTS_BASE_URL}/fixtures/multiple"
        params = {"api_token": tok, "ids": ids_param}

        max_retries = 3
        backoff_base = 1.5
        resp_data: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    try:
                        resp_data = resp.json()
                        break
                    except Exception:
                        logger.warning("OddAlerts returned non-JSON 200 response (invalid/expired credentials).")
                        break
                elif resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else (backoff_base ** attempt)
                    logger.warning(f"OddAlerts HTTP 429 rate limit. Backing off for {sleep_time:.2f}s (attempt {attempt}/{max_retries})")
                    time.sleep(sleep_time)
                elif resp.status_code in (500, 502, 503, 504):
                    sleep_time = backoff_base ** attempt + random.uniform(0, 0.5)
                    logger.warning(f"OddAlerts HTTP {resp.status_code} transient error. Retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"OddAlerts non-retryable HTTP error: {resp.status_code}")
                    break
            except (requests.Timeout, requests.ConnectionError) as err:
                sleep_time = backoff_base ** attempt
                logger.warning(f"Connection issue on attempt {attempt}/{max_retries}: {redact(str(err), tok)}")
                time.sleep(sleep_time)

        scores_dict: Dict[str, FixtureScoreItem] = {}

        if resp_data and isinstance(resp_data.get("data"), list):
            items_list = resp_data["data"]
            for item in items_list:
                try:
                    norm = self._normalize_fixture(item)
                    scores_dict[str(norm.fixture_id)] = norm
                except Exception as parse_err:
                    logger.warning(f"Error parsing fixture item: {parse_err}")
            self._save_completed_cache()

        # Ensure all target fixture IDs exist in dictionary and preserve completed scores
        for fid in target_ids:
            str_fid = str(fid)
            if str_fid not in scores_dict:
                if str_fid in self._completed_cache:
                    scores_dict[str_fid] = self._completed_cache[str_fid]
                elif self._cached_data and str_fid in self._cached_data.get("scores", {}):
                    cached_item_dict = self._cached_data["scores"][str_fid]
                    scores_dict[str_fid] = FixtureScoreItem(**cached_item_dict)
                else:
                    scores_dict[str_fid] = self._build_fallback_item(fid)
            else:
                # If item is returned as UPCOMING/POSTP/CANC, but known completed in _completed_cache:
                if str_fid in self._completed_cache and scores_dict[str_fid].status in ("UPCOMING", "POSTP", "CANC"):
                    scores_dict[str_fid] = self._completed_cache[str_fid]

        active_count = sum(1 for v in scores_dict.values() if v.status in ("LIVE", "HT"))

        result = {
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            "active_fixtures_count": active_count,
            "cached": False,
            "scores": {k: dump_item(v) for k, v in scores_dict.items()}
        }

        with self._cache_lock:
            self._cached_data = result
            self._cache_timestamp = time.time()

        return result

    def _generate_fallback_response(self, reason: str, target_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        t_ids = target_ids or self.fixture_ids
        scores = {}
        for fid in t_ids:
            str_fid = str(fid)
            if str_fid in self._completed_cache:
                scores[str_fid] = dump_item(self._completed_cache[str_fid])
            elif self._cached_data and str_fid in self._cached_data.get("scores", {}):
                scores[str_fid] = self._cached_data["scores"][str_fid]
            else:
                scores[str_fid] = dump_item(self._build_fallback_item(fid))
        return {
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            "active_fixtures_count": 0,
            "cached": False,
            "scores": scores,
            "note": reason
        }


# FastAPI Application
app = FastAPI(
    title="Football Prediction Live Score Service",
    description="Secure, rate-shielded live score updating API for client dashboard.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

manager = LiveScoreManager()


@app.get("/health")
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint for Railway platform monitoring."""
    cache_age = None
    if manager._cache_timestamp > 0:
        cache_age = round(time.time() - manager._cache_timestamp, 1)
    return {
        "status": "healthy",
        "service": "football-prediction-live-scores",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures_tracked": len(manager.fixture_ids),
        "completed_cached": len(manager._completed_cache),
        "cached": bool(manager._cached_data),
        "cache_age_seconds": cache_age,
    }


@app.get("/api/live-scores", response_model=LiveScoresResponse)
def get_live_scores(ids: Optional[str] = None) -> LiveScoresResponse:
    """Return live scores and match statuses for all upcoming and completed fixtures."""
    try:
        requested_ids: Optional[List[int]] = None
        if ids:
            parsed = []
            for part in ids.split(","):
                part = part.strip()
                if part.isdigit():
                    parsed.append(int(part))
            if parsed:
                requested_ids = parsed
        data = manager.fetch_live_scores(requested_ids=requested_ids)
        return LiveScoresResponse(**data)
    except Exception as exc:
        logger.error(f"Unexpected error in get_live_scores: {redact(str(exc))}")
        fallback = manager._generate_fallback_response("INTERNAL_ERROR_FALLBACK")
        return LiveScoresResponse(**fallback)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
