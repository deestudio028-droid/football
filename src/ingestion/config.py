"""Configuration and credential loading.

The API token is read once from the environment (populated from a local
.env file that is never committed) and held only in memory as a plain
Python string on the Config object. Nothing in this module or any module
that consumes Config may print, log, or serialize the token. See
`api_client.RedactingFilter` for the log-side enforcement of this rule.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _load_dotenv(dotenv_path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY = VALUE per line, '#' comments, no quoting
    rules beyond simple strip). Avoids adding a python-dotenv dependency
    for a one-line file.
    """
    values: dict[str, str] = {}
    if not dotenv_path.exists():
        return values
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


@dataclass
class Season:
    season_id: int
    season_name: str


@dataclass
class Competition:
    name: str
    competition_id: int
    country_id: int
    seasons: list[Season]


@dataclass
class Config:
    api_token: str
    base_url: str
    competitions: list[Competition]
    project_root: Path
    raw_data_dir: Path
    db_path: Path
    checkpoint_path: Path
    log_dir: Path
    request_timeout_seconds: int = 30
    max_retries: int = 5
    backoff_base_seconds: float = 1.5
    page_size: int = 250  # OddAlerts' observed default per_page

    def __repr__(self) -> str:  # never let a stray print() leak the token
        return (
            f"Config(base_url={self.base_url!r}, "
            f"competitions={len(self.competitions)}, "
            f"api_token=<redacted, len={len(self.api_token)}>)"
        )

    __str__ = __repr__


def load_config(
    project_root: str | Path,
    env_var_name: str = "OddAlerts_API",
    competitions_file: str | Path | None = None,
) -> Config:
    """Build a Config from the project's .env file and competitions.json.

    project_root: the directory containing .env, data/, config/, logs/.
    """
    project_root = Path(project_root)
    dotenv_values = _load_dotenv(project_root / ".env")
    # Real environment variables take precedence over .env, consistent
    # with typical 12-factor behavior, but .env is the primary source
    # for this project since it runs locally.
    api_token = os.environ.get(env_var_name) or dotenv_values.get(env_var_name)
    if not api_token:
        raise ConfigError(
            f"Missing API token. Expected env var '{env_var_name}' in the "
            f"environment or in {project_root / '.env'}."
        )

    competitions_path = Path(competitions_file) if competitions_file else (
        project_root / "config" / "competitions.json"
    )
    if not competitions_path.exists():
        raise ConfigError(f"Missing competitions config at {competitions_path}")

    raw = json.loads(competitions_path.read_text(encoding="utf-8"))
    competitions = [
        Competition(
            name=c["name"],
            competition_id=c["competition_id"],
            country_id=c["country_id"],
            seasons=[Season(s["season_id"], s["season_name"]) for s in c["seasons"]],
        )
        for c in raw["competitions"]
    ]

    data_root = project_root / "data"
    return Config(
        api_token=api_token,
        base_url="https://data.oddalerts.com/api",
        competitions=competitions,
        project_root=project_root,
        raw_data_dir=data_root / "raw",
        db_path=data_root / "processed" / "matches.db",
        checkpoint_path=data_root / "checkpoints" / "ingestion_state.json",
        log_dir=project_root / "logs",
    )
