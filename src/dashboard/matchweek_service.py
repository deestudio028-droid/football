"""Official Matchweek Calendar Service for 2026/27 Domestic Season.

Provides deterministic Matchweek resolution based strictly on client-provided
calendar windows across the Top 5 European leagues.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALENDAR_CONFIG_PATH = PROJECT_ROOT / "config" / "matchweek_calendar_2026_27.json"


class MatchweekService:
    """Service for resolving official Matchweeks for European Top 5 leagues."""

    def __init__(self, config_path: Path = CALENDAR_CONFIG_PATH):
        self.config_path = Path(config_path)
        self._data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Matchweek configuration not found at {self.config_path}")
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    @property
    def leagues(self) -> Dict[str, Any]:
        return self._data.get("leagues", {})

    def get_matchweek(self, league: str, date_str: str) -> Optional[str]:
        """Resolve matchweek from league name and scheduled kickoff date (YYYY-MM-DD).

        Returns matchweek string (e.g. 'MW3', 'MW4') or None if not within configured windows.
        """
        league_conf = self.leagues.get(league)
        if not league_conf:
            return None

        # Clean date string to YYYY-MM-DD
        dt_clean = str(date_str).strip()[:10]

        for entry in league_conf.get("calendar", []):
            if entry["start_date"] <= dt_clean <= entry["end_date"]:
                return entry["matchweek"]

        return None

    def get_current_matchweek(self, league: str) -> Optional[str]:
        """Return the current matchweek for the league as of the Sep 2026 international break."""
        league_conf = self.leagues.get(league)
        if not league_conf:
            return None
        return league_conf.get("current_matchweek")

    def get_calendar_summary(self) -> List[Dict[str, Any]]:
        """Return all matchweek calendar windows as structured dicts."""
        summary = []
        for league, conf in self.leagues.items():
            for entry in conf.get("calendar", []):
                summary.append({
                    "league": league,
                    "matchweek": entry["matchweek"],
                    "start_date": entry["start_date"],
                    "end_date": entry["end_date"],
                    "date_window": f"{entry['start_date']} → {entry['end_date']}",
                    "is_current": (entry["matchweek"] == conf.get("current_matchweek")),
                    "notes": entry.get("notes", ""),
                })
        return summary
