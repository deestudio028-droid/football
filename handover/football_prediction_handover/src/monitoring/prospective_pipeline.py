"""Production Monitoring & Prospective Validation Pipeline for the Frozen Draw Champion.

Model Identity:
    Model ID:           v4_draw_champion
    Model Version:      v4.0-champion-dc-elo-stacking
    Methodology Hash:   9c396e7e5364f93f079313726c1ba499
    Protocol Hash:      1311eb7fa75f51c77a1fc09c0cf4df68
    Minimum Cohort:     N >= 1,050 fresh fixtures (preferred >= 1,500)

INVARIANTS & SAFETY CONTRACTS:
1. Strict Pre-Match Prediction Lock: Predictions must be cryptographically hashed (SHA-256)
   and locked strictly BEFORE match kickoff.
2. Two-Stage Outcome Separation: Match outcomes are recorded separately after match completion;
   locked prediction records are completely immutable.
3. Historical Reused 300 OOS Exclusion: The 300 fixtures from `fresh_extended_fixture_ids.json`
   (REUSED_HISTORICAL_RESEARCH_OOS) are strictly excluded from the prospective cohort count.
4. Fail-Closed Validation: Rejects post-kickoff predictions, duplicate fixtures, invalid
   probabilities, NaN/Inf, missing causal features, and methodology mismatches.
5. Zero Market Contamination: Market odds remain reference-only benchmarks and never enter
   the model feature matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from models.baselines import CLASS_ORDER

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_CHAMPION_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "draw_champion_method_frozen.json"
FROZEN_PROTOCOL_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "prospective_validation_protocol.json"
REUSED_300_IDS_PATH = PROJECT_ROOT / "research" / "v4_promotion" / "fresh_extended_fixture_ids.json"

EXPECTED_MODEL_ID = "v4_draw_champion"
EXPECTED_MODEL_VERSION = "v4.0-champion-dc-elo-stacking"
EXPECTED_METHODOLOGY_HASH = "9c396e7e5364f93f079313726c1ba499"
EXPECTED_PROTOCOL_HASH = "1311eb7fa75f51c77a1fc09c0cf4df68"

MIN_PROSPECTIVE_COHORT = 1050
PREFERRED_PROSPECTIVE_COHORT = 1500
PROB_TOL = 1e-6


class CohortStatus(str, Enum):
    PROSPECTIVE_PENDING = "PROSPECTIVE_PENDING"
    PROSPECTIVE_ACTIVE = "PROSPECTIVE_ACTIVE"
    PROSPECTIVE_COMPLETE = "PROSPECTIVE_COMPLETE"
    PROSPECTIVE_VALIDATED = "PROSPECTIVE_VALIDATED"
    PROSPECTIVE_REJECTED = "PROSPECTIVE_REJECTED"


class ProspectivePipelineError(RuntimeError):
    """Base error for prospective validation pipeline safety gates."""


class PreKickoffViolationError(ProspectivePipelineError):
    """Raised when prediction timestamp is not strictly before kickoff timestamp."""


class DuplicatePredictionError(ProspectivePipelineError):
    """Raised when attempting to lock a duplicate prediction or duplicate fixture."""


class LockedRecordMutationError(ProspectivePipelineError):
    """Raised when attempting to modify or tamper with a locked prediction record."""


class OutcomeJoinError(ProspectivePipelineError):
    """Raised when outcome arrival violates causal timing or lacks a locked prediction."""


class ProbabilityValidationError(ProspectivePipelineError):
    """Raised when probabilities fail simplex sum, non-negativity, or contain NaN/Inf."""


class ReusedFixtureRejectionError(ProspectivePipelineError):
    """Raised when a reused historical fixture attempts to enter the prospective cohort."""


class MethodologyMismatchError(ProspectivePipelineError):
    """Raised when model ID, version, or methodology hash does not match frozen contract."""


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def canonical_json_hash(payload: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 digest of canonical JSON payload."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProspectiveContract:
    """Verification contract for prospective validation."""
    model_id: str = EXPECTED_MODEL_ID
    model_version: str = EXPECTED_MODEL_VERSION
    methodology_hash: str = EXPECTED_METHODOLOGY_HASH
    protocol_hash: str = EXPECTED_PROTOCOL_HASH
    min_sample_size: int = MIN_PROSPECTIVE_COHORT
    preferred_sample_size: int = PREFERRED_PROSPECTIVE_COHORT
    reused_300_fixture_ids: frozenset[int] = frozenset()

    @classmethod
    def load_and_verify(
        cls,
        champion_path: Path = FROZEN_CHAMPION_PATH,
        protocol_path: Path = FROZEN_PROTOCOL_PATH,
        reused_path: Path = REUSED_300_IDS_PATH,
    ) -> ProspectiveContract:
        """Load and structurally verify all frozen protocol files and hashes."""
        if not champion_path.exists():
            raise FileNotFoundError(f"Missing frozen champion specification: {champion_path}")
        if not protocol_path.exists():
            raise FileNotFoundError(f"Missing frozen prospective protocol: {protocol_path}")

        champ_md5 = md5(champion_path)
        proto_md5 = md5(protocol_path)

        if champ_md5 != EXPECTED_METHODOLOGY_HASH:
            raise MethodologyMismatchError(
                f"Champion methodology hash mismatch: got {champ_md5}, expected {EXPECTED_METHODOLOGY_HASH}"
            )
        if proto_md5 != EXPECTED_PROTOCOL_HASH:
            raise MethodologyMismatchError(
                f"Prospective protocol hash mismatch: got {proto_md5}, expected {EXPECTED_PROTOCOL_HASH}"
            )

        reused_ids: set[int] = set()
        if reused_path.exists():
            with open(reused_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                if isinstance(raw_data, dict) and "fixture_ids" in raw_data:
                    reused_ids = {int(x) for x in raw_data["fixture_ids"]}
                elif isinstance(raw_data, list):
                    reused_ids = {int(x) for x in raw_data}

        return cls(
            model_id=EXPECTED_MODEL_ID,
            model_version=EXPECTED_MODEL_VERSION,
            methodology_hash=champ_md5,
            protocol_hash=proto_md5,
            min_sample_size=MIN_PROSPECTIVE_COHORT,
            preferred_sample_size=PREFERRED_PROSPECTIVE_COHORT,
            reused_300_fixture_ids=frozenset(reused_ids),
        )


@dataclass(frozen=True)
class LockedPredictionRecord:
    """Immutable pre-match prediction record locked cryptographically before kickoff."""
    fixture_id: int
    prediction_id: str
    prediction_timestamp: str
    kickoff_timestamp: str
    model_id: str
    model_version: str
    methodology_hash: str

    p_home_v4: float
    p_draw_v4: float
    p_away_v4: float

    p_home_champion: float
    p_draw_champion: float
    p_away_champion: float

    p_draw_dc: float
    p_draw_elo: float
    expected_goals_home: float
    expected_goals_away: float
    modal_scoreline: str

    league: str
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    elo_diff: float
    abs_elo_diff: float
    lambda_home: float
    lambda_away: float

    prediction_hash: str
    status: str = "LOCKED"

    @classmethod
    def create_and_lock(
        cls,
        fixture_id: int,
        prediction_timestamp: str,
        kickoff_timestamp: str,
        p_v4: dict[str, float] | np.ndarray,
        p_champion: dict[str, float] | np.ndarray,
        p_draw_dc: float,
        p_draw_elo: float,
        expected_goals_home: float,
        expected_goals_away: float,
        modal_scoreline: str,
        league: str,
        home_team: str,
        away_team: str,
        home_elo: float,
        away_elo: float,
        lambda_home: float,
        lambda_away: float,
        contract: ProspectiveContract | None = None,
    ) -> LockedPredictionRecord:
        """Create, validate, hash, and lock a pre-match prediction record."""
        if contract is None:
            contract = ProspectiveContract.load_and_verify()

        # 1. Historical 300 OOS Exclusion
        if int(fixture_id) in contract.reused_300_fixture_ids:
            raise ReusedFixtureRejectionError(
                f"Fixture {fixture_id} is in REUSED_HISTORICAL_RESEARCH_OOS and cannot enter prospective validation."
            )

        # 2. Timing Verification (prediction strictly before kickoff)
        # Parse ISO timestamps or UNIX integer strings
        try:
            t_pred = datetime.fromisoformat(prediction_timestamp.replace("Z", "+00:00"))
            t_kick = datetime.fromisoformat(kickoff_timestamp.replace("Z", "+00:00"))
        except Exception:
            # Fallback for unix timestamp string comparison if integers passed
            t_pred = float(prediction_timestamp)
            t_kick = float(kickoff_timestamp)

        if t_pred >= t_kick:
            raise PreKickoffViolationError(
                f"Prediction timestamp ({prediction_timestamp}) must be strictly before kickoff ({kickoff_timestamp})"
            )

        # 3. Probability Vector Extraction & Simplex Validation
        if isinstance(p_v4, dict):
            pv4_vec = np.array([p_v4["H"], p_v4["D"], p_v4["A"]], dtype=float)
        else:
            pv4_vec = np.asarray(p_v4, dtype=float)

        if isinstance(p_champion, dict):
            pchamp_vec = np.array([p_champion["H"], p_champion["D"], p_champion["A"]], dtype=float)
        else:
            pchamp_vec = np.asarray(p_champion, dtype=float)

        for name, p_vec in [("V4", pv4_vec), ("Champion", pchamp_vec)]:
            if not np.all(np.isfinite(p_vec)):
                raise ProbabilityValidationError(f"{name} probabilities contain NaN or Inf: {p_vec}")
            if np.any(p_vec < 0.0):
                raise ProbabilityValidationError(f"{name} probabilities contain negative values: {p_vec}")
            if abs(p_vec.sum() - 1.0) > PROB_TOL:
                raise ProbabilityValidationError(f"{name} probabilities do not sum to 1.0: {p_vec.sum():.8f}")

        # 4. Causal Feature Validation
        for val_name, val in [
            ("lambda_home", lambda_home), ("lambda_away", lambda_away),
            ("home_elo", home_elo), ("away_elo", away_elo),
        ]:
            if not math.isfinite(float(val)):
                raise ProspectivePipelineError(f"Causal feature {val_name} is non-finite: {val}")

        elo_diff = float((home_elo + 100.0) - away_elo)
        abs_elo = float(abs(elo_diff))

        # 5. Deterministic Prediction ID
        pid_payload = {
            "fixture_id": int(fixture_id),
            "kickoff_timestamp": kickoff_timestamp,
            "prediction_timestamp": prediction_timestamp,
            "model_version": contract.model_version,
        }
        pred_id = f"pred_{canonical_json_hash(pid_payload)[:16]}"

        # 6. Canonical Hash Payload (Everything required for immutable prediction)
        hash_payload = {
            "fixture_id": int(fixture_id),
            "prediction_id": pred_id,
            "prediction_timestamp": prediction_timestamp,
            "kickoff_timestamp": kickoff_timestamp,
            "model_id": contract.model_id,
            "model_version": contract.model_version,
            "methodology_hash": contract.methodology_hash,
            "p_home_v4": round(float(pv4_vec[0]), 8),
            "p_draw_v4": round(float(pv4_vec[1]), 8),
            "p_away_v4": round(float(pv4_vec[2]), 8),
            "p_home_champion": round(float(pchamp_vec[0]), 8),
            "p_draw_champion": round(float(pchamp_vec[1]), 8),
            "p_away_champion": round(float(pchamp_vec[2]), 8),
            "p_draw_dc": round(float(p_draw_dc), 8),
            "p_draw_elo": round(float(p_draw_elo), 8),
            "expected_goals_home": round(float(expected_goals_home), 6),
            "expected_goals_away": round(float(expected_goals_away), 6),
            "modal_scoreline": str(modal_scoreline),
            "league": str(league),
            "home_team": str(home_team),
            "away_team": str(away_team),
            "home_elo": round(float(home_elo), 4),
            "away_elo": round(float(away_elo), 4),
            "elo_diff": round(elo_diff, 4),
            "abs_elo_diff": round(abs_elo, 4),
            "lambda_home": round(float(lambda_home), 6),
            "lambda_away": round(float(lambda_away), 6),
        }
        pred_hash = canonical_json_hash(hash_payload)

        return cls(
            fixture_id=int(fixture_id),
            prediction_id=pred_id,
            prediction_timestamp=prediction_timestamp,
            kickoff_timestamp=kickoff_timestamp,
            model_id=contract.model_id,
            model_version=contract.model_version,
            methodology_hash=contract.methodology_hash,
            p_home_v4=float(pv4_vec[0]),
            p_draw_v4=float(pv4_vec[1]),
            p_away_v4=float(pv4_vec[2]),
            p_home_champion=float(pchamp_vec[0]),
            p_draw_champion=float(pchamp_vec[1]),
            p_away_champion=float(pchamp_vec[2]),
            p_draw_dc=float(p_draw_dc),
            p_draw_elo=float(p_draw_elo),
            expected_goals_home=float(expected_goals_home),
            expected_goals_away=float(expected_goals_away),
            modal_scoreline=str(modal_scoreline),
            league=str(league),
            home_team=str(home_team),
            away_team=str(away_team),
            home_elo=float(home_elo),
            away_elo=float(away_elo),
            elo_diff=elo_diff,
            abs_elo_diff=abs_elo,
            lambda_home=float(lambda_home),
            lambda_away=float(lambda_away),
            prediction_hash=pred_hash,
            status="LOCKED",
        )

    def verify_hash(self) -> bool:
        """Verify that the record's contents strictly match its prediction hash."""
        payload = {
            "fixture_id": self.fixture_id,
            "prediction_id": self.prediction_id,
            "prediction_timestamp": self.prediction_timestamp,
            "kickoff_timestamp": self.kickoff_timestamp,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "methodology_hash": self.methodology_hash,
            "p_home_v4": round(float(self.p_home_v4), 8),
            "p_draw_v4": round(float(self.p_draw_v4), 8),
            "p_away_v4": round(float(self.p_away_v4), 8),
            "p_home_champion": round(float(self.p_home_champion), 8),
            "p_draw_champion": round(float(self.p_draw_champion), 8),
            "p_away_champion": round(float(self.p_away_champion), 8),
            "p_draw_dc": round(float(self.p_draw_dc), 8),
            "p_draw_elo": round(float(self.p_draw_elo), 8),
            "expected_goals_home": round(float(self.expected_goals_home), 6),
            "expected_goals_away": round(float(self.expected_goals_away), 6),
            "modal_scoreline": str(self.modal_scoreline),
            "league": str(self.league),
            "home_team": str(self.home_team),
            "away_team": str(self.away_team),
            "home_elo": round(float(self.home_elo), 4),
            "away_elo": round(float(self.away_elo), 4),
            "elo_diff": round(self.elo_diff, 4),
            "abs_elo_diff": round(self.abs_elo_diff, 4),
            "lambda_home": round(float(self.lambda_home), 6),
            "lambda_away": round(float(self.lambda_away), 6),
        }
        return canonical_json_hash(payload) == self.prediction_hash


@dataclass(frozen=True)
class OutcomeRecord:
    """Post-match outcome record joined via fixture_id."""
    fixture_id: int
    outcome_timestamp: str
    home_goals: int
    away_goals: int
    actual_class: str
    result_source: str = "OFFICIAL_FT"
    status: str = "JOINED"

    @classmethod
    def create(
        cls,
        fixture_id: int,
        outcome_timestamp: str,
        home_goals: int,
        away_goals: int,
        kickoff_timestamp: str | None = None,
    ) -> OutcomeRecord:
        """Create and validate a post-match outcome record."""
        hg = int(home_goals)
        ag = int(away_goals)
        if hg < 0 or ag < 0:
            raise OutcomeJoinError(f"Negative goals in outcome: {hg}-{ag}")

        if kickoff_timestamp:
            try:
                t_out = datetime.fromisoformat(outcome_timestamp.replace("Z", "+00:00"))
                t_kick = datetime.fromisoformat(kickoff_timestamp.replace("Z", "+00:00"))
            except Exception:
                t_out = float(outcome_timestamp)
                t_kick = float(kickoff_timestamp)
            if t_out <= t_kick:
                raise OutcomeJoinError(
                    f"Outcome timestamp ({outcome_timestamp}) must be strictly after kickoff ({kickoff_timestamp})"
                )

        act = "H" if hg > ag else ("D" if hg == ag else "A")
        return cls(
            fixture_id=int(fixture_id),
            outcome_timestamp=outcome_timestamp,
            home_goals=hg,
            away_goals=ag,
            actual_class=act,
            result_source="OFFICIAL_FT",
            status="JOINED",
        )


class ProspectiveValidationStore:
    """Isolated SQLite / In-Memory storage for prospective validation records."""

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path)
        self._shared_conn = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self):
        if self._shared_conn is not None:
            self._shared_conn.close()

    def _release_connection(self, conn: sqlite3.Connection):
        if self._shared_conn is None:
            conn.close()

    def _init_db(self):
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS locked_predictions (
                fixture_id INTEGER PRIMARY KEY,
                prediction_id TEXT UNIQUE NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                kickoff_timestamp TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                methodology_hash TEXT NOT NULL,
                p_home_v4 REAL NOT NULL,
                p_draw_v4 REAL NOT NULL,
                p_away_v4 REAL NOT NULL,
                p_home_champion REAL NOT NULL,
                p_draw_champion REAL NOT NULL,
                p_away_champion REAL NOT NULL,
                p_draw_dc REAL NOT NULL,
                p_draw_elo REAL NOT NULL,
                expected_goals_home REAL NOT NULL,
                expected_goals_away REAL NOT NULL,
                modal_scoreline TEXT NOT NULL,
                league TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_elo REAL NOT NULL,
                away_elo REAL NOT NULL,
                elo_diff REAL NOT NULL,
                abs_elo_diff REAL NOT NULL,
                lambda_home REAL NOT NULL,
                lambda_away REAL NOT NULL,
                prediction_hash TEXT NOT NULL,
                status TEXT NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS match_outcomes (
                fixture_id INTEGER PRIMARY KEY,
                outcome_timestamp TEXT NOT NULL,
                home_goals INTEGER NOT NULL,
                away_goals INTEGER NOT NULL,
                actual_class TEXT NOT NULL,
                result_source TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (fixture_id) REFERENCES locked_predictions(fixture_id)
            );
        """)
        conn.commit()
        self._release_connection(conn)

    def lock_prediction(self, record: LockedPredictionRecord):
        """Save a locked prediction. Rejects duplicates or tampering."""
        if not record.verify_hash():
            raise LockedRecordMutationError(f"Prediction {record.fixture_id} failed hash verification (tampered).")

        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO locked_predictions (
                    fixture_id, prediction_id, prediction_timestamp, kickoff_timestamp,
                    model_id, model_version, methodology_hash,
                    p_home_v4, p_draw_v4, p_away_v4,
                    p_home_champion, p_draw_champion, p_away_champion,
                    p_draw_dc, p_draw_elo, expected_goals_home, expected_goals_away,
                    modal_scoreline, league, home_team, away_team,
                    home_elo, away_elo, elo_diff, abs_elo_diff,
                    lambda_home, lambda_away, prediction_hash, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.fixture_id, record.prediction_id, record.prediction_timestamp, record.kickoff_timestamp,
                record.model_id, record.model_version, record.methodology_hash,
                record.p_home_v4, record.p_draw_v4, record.p_away_v4,
                record.p_home_champion, record.p_draw_champion, record.p_away_champion,
                record.p_draw_dc, record.p_draw_elo, record.expected_goals_home, record.expected_goals_away,
                record.modal_scoreline, record.league, record.home_team, record.away_team,
                record.home_elo, record.away_elo, record.elo_diff, record.abs_elo_diff,
                record.lambda_home, record.lambda_away, record.prediction_hash, record.status
            ))
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise DuplicatePredictionError(f"Duplicate prediction or fixture ID: {record.fixture_id} ({exc})") from exc
        finally:
            self._release_connection(conn)

    def get_prediction(self, fixture_id: int) -> LockedPredictionRecord | None:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM locked_predictions WHERE fixture_id = ?", (int(fixture_id),))
        row = cur.fetchone()
        self._release_connection(conn)
        if not row:
            return None
        d = dict(row)
        rec = LockedPredictionRecord(**d)
        if not rec.verify_hash():
            raise LockedRecordMutationError(f"Retrieved prediction {fixture_id} failed hash verification.")
        return rec

    def join_outcome(self, outcome: OutcomeRecord):
        """Join a post-match outcome record."""
        pred = self.get_prediction(outcome.fixture_id)
        if pred is None:
            raise OutcomeJoinError(f"Cannot join outcome for fixture {outcome.fixture_id}: no locked prediction found.")

        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO match_outcomes (
                    fixture_id, outcome_timestamp, home_goals, away_goals, actual_class, result_source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.fixture_id, outcome.outcome_timestamp, outcome.home_goals,
                outcome.away_goals, outcome.actual_class, outcome.result_source, outcome.status
            ))
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise OutcomeJoinError(f"Duplicate outcome for fixture {outcome.fixture_id} ({exc})") from exc
        finally:
            self._release_connection(conn)

    def get_joined_records(self) -> list[tuple[LockedPredictionRecord, OutcomeRecord]]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT p.*, o.outcome_timestamp, o.home_goals, o.away_goals, o.actual_class, o.result_source, o.status as outcome_status
            FROM locked_predictions p
            JOIN match_outcomes o ON p.fixture_id = o.fixture_id
            ORDER BY p.kickoff_timestamp ASC, p.fixture_id ASC
        """)
        rows = cur.fetchall()
        self._release_connection(conn)

        out = []
        for r in rows:
            d = dict(r)
            o_d = {
                "fixture_id": d["fixture_id"],
                "outcome_timestamp": d["outcome_timestamp"],
                "home_goals": d["home_goals"],
                "away_goals": d["away_goals"],
                "actual_class": d["actual_class"],
                "result_source": d["result_source"],
                "status": d["outcome_status"],
            }
            p_d = {k: d[k] for k in d if k not in ("outcome_timestamp", "home_goals", "away_goals", "actual_class", "result_source", "outcome_status")}
            p_d["status"] = d["status"]
            pred = LockedPredictionRecord(**p_d)
            if not pred.verify_hash():
                raise LockedRecordMutationError(f"Joined prediction {pred.fixture_id} corrupted.")
            outcome = OutcomeRecord(**o_d)
            out.append((pred, outcome))
        return out

    def get_progress_summary(self) -> dict[str, Any]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as total_locked FROM locked_predictions")
        total_locked = cur.fetchone()["total_locked"]
        cur.execute("SELECT COUNT(*) as total_joined FROM match_outcomes")
        total_joined = cur.fetchone()["total_joined"]
        self._release_connection(conn)

        status = CohortStatus.PROSPECTIVE_ACTIVE
        if total_joined >= MIN_PROSPECTIVE_COHORT:
            status = CohortStatus.PROSPECTIVE_COMPLETE

        return {
            "locked_predictions": total_locked,
            "completed_outcomes": total_joined,
            "pending_outcomes": total_locked - total_joined,
            "min_required": MIN_PROSPECTIVE_COHORT,
            "preferred_target": PREFERRED_PROSPECTIVE_COHORT,
            "progress_pct": round(100.0 * total_joined / MIN_PROSPECTIVE_COHORT, 2),
            "status": status.value,
        }


class ProspectiveMonitor:
    """Calculates continuous monitoring metrics and chronological bucket stability."""

    @staticmethod
    def compute_metrics(records: list[tuple[LockedPredictionRecord, OutcomeRecord]]) -> dict[str, Any]:
        if not records:
            return {"n": 0, "status": "NO_DATA"}

        n = len(records)
        y = np.array([o.actual_class for _, o in records])
        oh = np.zeros((n, 3), dtype=float)
        for i, c in enumerate(CLASS_ORDER):
            oh[:, i] = (y == c)

        P_v4 = np.array([[p.p_home_v4, p.p_draw_v4, p.p_away_v4] for p, _ in records])
        P_champ = np.array([[p.p_home_champion, p.p_draw_champion, p.p_away_champion] for p, _ in records])

        P_v4_c = np.clip(P_v4, 1e-15, 1.0); P_v4_c /= P_v4_c.sum(axis=1, keepdims=True)
        P_champ_c = np.clip(P_champ, 1e-15, 1.0); P_champ_c /= P_champ_c.sum(axis=1, keepdims=True)

        ll_v4 = float(-np.mean(np.sum(oh * np.log(P_v4_c), axis=1)))
        ll_champ = float(-np.mean(np.sum(oh * np.log(P_champ_c), axis=1)))
        delta_ll = float(ll_champ - ll_v4)

        brier_v4 = float(np.mean(np.sum((P_v4 - oh) ** 2, axis=1)))
        brier_champ = float(np.mean(np.sum((P_champ - oh) ** 2, axis=1)))

        cp_v4, co = np.cumsum(P_v4, axis=1), np.cumsum(oh, axis=1)
        cp_champ = np.cumsum(P_champ, axis=1)
        rps_v4 = float(np.mean(np.sum((cp_v4[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))
        rps_champ = float(np.mean(np.sum((cp_champ[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))

        preds_v4 = np.array([CLASS_ORDER[i] for i in P_v4.argmax(axis=1)])
        preds_champ = np.array([CLASS_ORDER[i] for i in P_champ.argmax(axis=1)])
        acc_v4 = float(np.mean(preds_v4 == y))
        acc_champ = float(np.mean(preds_champ == y))

        actual_draw_rate = float(np.mean(y == "D"))
        mean_pd_v4 = float(P_v4[:, 1].mean())
        mean_pd_champ = float(P_champ[:, 1].mean())

        # Chronological 50-match and 100-match buckets
        buckets_50 = []
        n_b50 = int(math.ceil(n / 50.0))
        for b in range(n_b50):
            s_idx, e_idx = b * 50, min(n, (b + 1) * 50)
            sub_oh = oh[s_idx:e_idx]
            sub_v4 = P_v4_c[s_idx:e_idx]
            sub_champ = P_champ_c[s_idx:e_idx]
            sub_ll_v4 = float(-np.mean(np.sum(sub_oh * np.log(sub_v4), axis=1)))
            sub_ll_champ = float(-np.mean(np.sum(sub_oh * np.log(sub_champ), axis=1)))
            buckets_50.append({
                "bucket": f"{s_idx + 1}-{e_idx}",
                "n": e_idx - s_idx,
                "v4_log_loss": round(sub_ll_v4, 6),
                "champion_log_loss": round(sub_ll_champ, 6),
                "delta_log_loss": round(sub_ll_champ - sub_ll_v4, 6),
                "actual_draw_rate": round(float(np.mean(y[s_idx:e_idx] == "D")), 4),
                "mean_pd_champ": round(float(P_champ[s_idx:e_idx, 1].mean()), 4),
            })

        buckets_100 = []
        n_b100 = int(math.ceil(n / 100.0))
        for b in range(n_b100):
            s_idx, e_idx = b * 100, min(n, (b + 1) * 100)
            sub_oh = oh[s_idx:e_idx]
            sub_v4 = P_v4_c[s_idx:e_idx]
            sub_champ = P_champ_c[s_idx:e_idx]
            sub_ll_v4 = float(-np.mean(np.sum(sub_oh * np.log(sub_v4), axis=1)))
            sub_ll_champ = float(-np.mean(np.sum(sub_oh * np.log(sub_champ), axis=1)))
            buckets_100.append({
                "bucket": f"{s_idx + 1}-{e_idx}",
                "n": e_idx - s_idx,
                "v4_log_loss": round(sub_ll_v4, 6),
                "champion_log_loss": round(sub_ll_champ, 6),
                "delta_log_loss": round(sub_ll_champ - sub_ll_v4, 6),
                "actual_draw_rate": round(float(np.mean(y[s_idx:e_idx] == "D")), 4),
                "mean_pd_champ": round(float(P_champ[s_idx:e_idx, 1].mean()), 4),
            })

        # Descriptive drift metrics
        elo_diffs = np.array([p.elo_diff for p, _ in records])
        lam_homes = np.array([p.lambda_home for p, _ in records])
        lam_aways = np.array([p.lambda_away for p, _ in records])
        entropy_champ = float(-np.mean(np.sum(P_champ_c * np.log(P_champ_c), axis=1)))

        drift_metrics = {
            "mean_elo_diff": round(float(np.mean(elo_diffs)), 2),
            "std_elo_diff": round(float(np.std(elo_diffs)), 2),
            "mean_lambda_home": round(float(np.mean(lam_homes)), 4),
            "mean_lambda_away": round(float(np.mean(lam_aways)), 4),
            "mean_entropy_champion": round(entropy_champ, 4),
            "draw_rate_drift_vs_v4": round(mean_pd_champ - mean_pd_v4, 4),
            "draw_rate_actual": round(actual_draw_rate, 4),
        }

        return {
            "n": n,
            "v4": {
                "log_loss": round(ll_v4, 6),
                "brier": round(brier_v4, 6),
                "rps": round(rps_v4, 6),
                "accuracy": round(acc_v4, 4),
                "mean_p_draw": round(mean_pd_v4, 4),
                "draw_bias": round(mean_pd_v4 - actual_draw_rate, 4),
            },
            "champion": {
                "log_loss": round(ll_champ, 6),
                "brier": round(brier_champ, 6),
                "rps": round(rps_champ, 6),
                "accuracy": round(acc_champ, 4),
                "mean_p_draw": round(mean_pd_champ, 4),
                "draw_bias": round(mean_pd_champ - actual_draw_rate, 4),
            },
            "comparison": {
                "delta_log_loss": round(delta_ll, 6),
                "delta_brier": round(brier_champ - brier_v4, 6),
                "delta_rps": round(rps_champ - rps_v4, 6),
                "delta_accuracy": round(acc_champ - acc_v4, 4),
                "actual_draw_rate": round(actual_draw_rate, 4),
            },
            "buckets_50": buckets_50,
            "buckets_100": buckets_100,
            "drift_diagnostics": drift_metrics,
        }


def is_fresh_prospective_fixture(
    fixture_id: int,
    kickoff_timestamp: str,
    prediction_timestamp: str | None = None,
    contract: ProspectiveContract | None = None,
    store: ProspectiveValidationStore | None = None,
) -> bool:
    """Strict eligibility check for fresh prospective fixtures."""
    if contract is None:
        contract = ProspectiveContract.load_and_verify()

    fid = int(fixture_id)

    # 1. Must not belong to reused 300 OOS dataset
    if fid in contract.reused_300_fixture_ids:
        return False

    # 2. Must not already exist in prospective storage
    if store is not None and store.get_prediction(fid) is not None:
        return False

    # 3. Kickoff must be strictly in the future relative to prediction timestamp
    if prediction_timestamp is not None:
        try:
            t_pred = datetime.fromisoformat(prediction_timestamp.replace("Z", "+00:00"))
            t_kick = datetime.fromisoformat(kickoff_timestamp.replace("Z", "+00:00"))
        except Exception:
            t_pred = float(prediction_timestamp)
            t_kick = float(kickoff_timestamp)
        if t_pred >= t_kick:
            return False

    return True


def lock_prospective_prediction(
    fixture_id: int,
    prediction_timestamp: str,
    kickoff_timestamp: str,
    p_v4: dict[str, float] | np.ndarray,
    p_champion: dict[str, float] | np.ndarray,
    p_draw_dc: float,
    p_draw_elo: float,
    expected_goals_home: float,
    expected_goals_away: float,
    modal_scoreline: str,
    league: str,
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    lambda_home: float,
    lambda_away: float,
    store: ProspectiveValidationStore,
    contract: ProspectiveContract | None = None,
) -> LockedPredictionRecord:
    """High-level atomic workflow to validate, hash, and lock a prospective prediction."""
    if contract is None:
        contract = ProspectiveContract.load_and_verify()

    if not is_fresh_prospective_fixture(fixture_id, kickoff_timestamp, prediction_timestamp, contract, store):
        if int(fixture_id) in contract.reused_300_fixture_ids:
            raise ReusedFixtureRejectionError(f"Fixture {fixture_id} is in REUSED_HISTORICAL_RESEARCH_OOS.")
        if store.get_prediction(int(fixture_id)) is not None:
            raise DuplicatePredictionError(f"Fixture {fixture_id} is already locked in prospective store.")
        raise PreKickoffViolationError(f"Fixture {fixture_id} failed timing verification ({prediction_timestamp} >= {kickoff_timestamp}).")

    record = LockedPredictionRecord.create_and_lock(
        fixture_id=fixture_id,
        prediction_timestamp=prediction_timestamp,
        kickoff_timestamp=kickoff_timestamp,
        p_v4=p_v4,
        p_champion=p_champion,
        p_draw_dc=p_draw_dc,
        p_draw_elo=p_draw_elo,
        expected_goals_home=expected_goals_home,
        expected_goals_away=expected_goals_away,
        modal_scoreline=modal_scoreline,
        league=league,
        home_team=home_team,
        away_team=away_team,
        home_elo=home_elo,
        away_elo=away_elo,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        contract=contract,
    )

    store.lock_prediction(record)
    return record


def join_prospective_outcome(
    fixture_id: int,
    outcome_timestamp: str,
    home_goals: int,
    away_goals: int,
    store: ProspectiveValidationStore,
    result_source: str = "OFFICIAL_FT",
) -> OutcomeRecord:
    """High-level workflow to validate and record a match outcome."""
    pred = store.get_prediction(fixture_id)
    if pred is None:
        raise OutcomeJoinError(f"Cannot join outcome for fixture {fixture_id}: no locked prediction found.")

    outcome = OutcomeRecord.create(
        fixture_id=fixture_id,
        outcome_timestamp=outcome_timestamp,
        home_goals=home_goals,
        away_goals=away_goals,
        kickoff_timestamp=pred.kickoff_timestamp,
    )
    store.join_outcome(outcome)
    return outcome


def run_final_statistical_validation(
    records: list[tuple[LockedPredictionRecord, OutcomeRecord]],
    n_resamples: int = 10000,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Execute the pre-registered prospective statistical decision rule."""
    n = len(records)
    if n < MIN_PROSPECTIVE_COHORT:
        return {
            "status": "VALIDATION_BLOCKED",
            "n_completed": n,
            "n_required": MIN_PROSPECTIVE_COHORT,
            "verdict": "NOT_YET_ELIGIBLE_FOR_STATISTICAL_DECISION",
            "message": f"Prospective cohort has {n} fixtures; minimum {MIN_PROSPECTIVE_COHORT} required.",
        }

    metrics = ProspectiveMonitor.compute_metrics(records)
    y = np.array([o.actual_class for _, o in records])
    oh = np.zeros((n, 3), dtype=float)
    for i, c in enumerate(CLASS_ORDER):
        oh[:, i] = (y == c)

    P_v4 = np.array([[p.p_home_v4, p.p_draw_v4, p.p_away_v4] for p, _ in records])
    P_champ = np.array([[p.p_home_champion, p.p_draw_champion, p.p_away_champion] for p, _ in records])

    P_v4_c = np.clip(P_v4, 1e-15, 1.0); P_v4_c /= P_v4_c.sum(axis=1, keepdims=True)
    P_champ_c = np.clip(P_champ, 1e-15, 1.0); P_champ_c /= P_champ_c.sum(axis=1, keepdims=True)

    ll_v4_indiv = -np.sum(oh * np.log(P_v4_c), axis=1)
    ll_champ_indiv = -np.sum(oh * np.log(P_champ_c), axis=1)
    diff = ll_champ_indiv - ll_v4_indiv

    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    ci_lower = float(np.percentile(boot, 2.5))
    ci_upper = float(np.percentile(boot, 97.5))
    mean_delta = float(diff.mean())

    ci_excludes_zero = bool(ci_upper < 0.0)
    practical_pass = bool(mean_delta <= -0.0010)

    # Mandatory Gates Check
    gates = {
        "sample_size_satisfied": bool(n >= MIN_PROSPECTIVE_COHORT),
        "primary_metric_improved": bool(mean_delta < 0.0),
        "statistical_significance_95pct_ci": ci_excludes_zero,
        "practical_significance_met": practical_pass,
        "secondary_metrics_no_degradation": bool(metrics["comparison"]["delta_brier"] <= 0.0005),
    }
    all_gates_passed = all(gates.values())

    return {
        "status": "VALIDATION_PASSED_PENDING_HUMAN_REVIEW" if all_gates_passed else "VALIDATION_FAILED",
        "n": n,
        "metrics": metrics,
        "statistical_inference": {
            "mean_delta_log_loss": round(mean_delta, 6),
            "ci_95_lower": round(ci_lower, 6),
            "ci_95_upper": round(ci_upper, 6),
            "ci_excludes_zero": ci_excludes_zero,
            "p_value_empirical": round(float(np.mean(boot >= 0)), 4),
            "n_resamples": n_resamples,
        },
        "gates": gates,
        "decision": "PROSPECTIVE_CONFIRMATION_ACCEPTED_PENDING_HUMAN_REVIEW" if all_gates_passed else "PROSPECTIVE_CONFIRMATION_REJECTED",
    }


__all__ = [
    "CohortStatus",
    "ProspectivePipelineError",
    "PreKickoffViolationError",
    "DuplicatePredictionError",
    "LockedRecordMutationError",
    "OutcomeJoinError",
    "ProbabilityValidationError",
    "ReusedFixtureRejectionError",
    "MethodologyMismatchError",
    "ProspectiveContract",
    "LockedPredictionRecord",
    "OutcomeRecord",
    "ProspectiveValidationStore",
    "ProspectiveMonitor",
    "is_fresh_prospective_fixture",
    "lock_prospective_prediction",
    "join_prospective_outcome",
    "run_final_statistical_validation",
    "canonical_json_hash",
]
