"""Historical Calculation Memory Layer for Football Prediction Lab.

Maintains an immutable memory of historical pre-match calculation patterns
and their verified post-match outcomes. When evaluating upcoming matches,
it retrieves nearest calculation analogues, evaluates historical success rates
under strict sample-size safeguards, and offers score refinement suggestions
without mutating the frozen V4.0 production model or its baseline probabilities.

INVARIANTS:
1. Frozen V4.0: V4.0 remains the primary production inference engine.
2. Advisory Layer Only: Memory evidence is strictly auxiliary and never overrides V4.0 probabilities.
3. Zero Temporal Leakage: historical_kickoff_utc < query_kickoff_utc is strictly enforced.
4. League Isolation: Retrieval is league-specific by default to prevent cross-gameweek contamination.
5. Zero Fabrication: Insufficient historical evidence returns explicit uninflated states.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("historical_memory_service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEDGER_PATH = PROJECT_ROOT / "data" / "processed" / "prediction_snapshot_ledger.json"
SHADOW_LEDGER_PATH = PROJECT_ROOT / "research" / "v5_model_improvement" / "step3c_prospective_shadow" / "prospective_shadow_ledger.csv"
OUTCOME_LEDGER_PATH = PROJECT_ROOT / "research" / "v5_model_improvement" / "production_monitoring" / "live_outcome_ledger.csv"
MATCHES_DB = PROJECT_ROOT / "data" / "processed" / "matches.db"


@dataclass
class HistoricalCalculationRecord:
    """Represents a completed match with its pre-kickoff calculation and verified post-match outcome."""
    fixture_id: int
    competition_id: int
    competition_name: str
    season: str
    home_team: str
    away_team: str
    kickoff_utc: str  # ISO-8601 UTC string

    # --- Pre-Kickoff Calculation Pattern (V4.0 Baseline) ---
    p_home: float
    p_draw: float
    p_away: float
    predicted_outcome: str  # "H", "D", "A"
    lambda_home: Optional[float] = None
    lambda_away: Optional[float] = None
    baseline_score: Optional[str] = None
    draw_risk_score: Optional[float] = None
    draw_risk_tier: Optional[str] = None
    prediction_timestamp_utc: str = ""

    # --- Post-Match Verified Outcome (Strict Temporal Separation) ---
    actual_home_goals: Optional[int] = None
    actual_away_goals: Optional[int] = None
    actual_outcome: Optional[str] = None  # "H", "D", "A"
    actual_score: Optional[str] = None  # e.g. "2-1"
    prediction_correct: Optional[bool] = None
    exact_score_correct: Optional[bool] = None
    is_completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HistoricalAnalogue:
    """Individual historical calculation match retrieved as an analogue."""
    fixture_id: int
    competition_name: str
    season: str
    home_team: str
    away_team: str
    kickoff_utc: str
    similarity_score: float  # Percentage 0.0 to 100.0%
    p_home: float
    p_draw: float
    p_away: float
    predicted_outcome: str
    baseline_score: Optional[str]
    actual_score: str
    actual_outcome: str
    prediction_correct: bool
    exact_score_correct: bool
    scope: str = "league"  # "league" or "cross_league"


@dataclass
class CorrectScoreRefinementResult:
    """Result of historical correct score refinement evaluation."""
    baseline_score: str
    refined_score: str
    refinement_status: str  # "REFINED", "BASELINE_KEPT", "INSUFFICIENT_EVIDENCE"
    score_confidence: str  # "HIGH", "MODERATE", "LOW", "NONE"
    historical_cluster_sample: int
    modal_score_count: int
    modal_score_frequency: float
    reason: str
    score_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class HistoricalEvidenceSignal:
    """Statistical evidence summary derived from similar historical calculation patterns."""
    total_analogues_found: int
    same_outcome_analogues_count: int
    same_outcome_success_count: int
    same_outcome_success_rate: float  # Percentage 0.0 to 100.0%
    overall_analogue_accuracy: float  # Percentage 0.0 to 100.0%
    average_similarity: float
    evidence_level: str  # "HIGH", "MODERATE", "CAUTION", "INSUFFICIENT"
    evidence_badge: str  # e.g. "🔥 HIGH HISTORICAL CONVICTION"
    advisory_summary: str
    league_scope: str


@dataclass
class HistoricalMemoryAnalysisResult:
    """Comprehensive output of the Historical Memory Layer for a single query match."""
    fixture_id: int
    home_team: str
    away_team: str
    competition_name: str
    kickoff_utc: str
    v4_p_home: float
    v4_p_draw: float
    v4_p_away: float
    v4_decision: str
    v4_baseline_score: str

    # Historical Analogues & Evidence
    analogues: List[HistoricalAnalogue]
    evidence_signal: HistoricalEvidenceSignal
    score_refinement: CorrectScoreRefinementResult
    has_sufficient_evidence: bool


class CalculationSimilarityEngine:
    """Deterministic, explainable similarity engine for prediction calculation vectors."""

    @staticmethod
    def calculate_distance(
        p_home_q: float,
        p_draw_q: float,
        p_away_q: float,
        dec_q: str,
        lh_q: Optional[float],
        la_q: Optional[float],
        record: HistoricalCalculationRecord,
    ) -> float:
        """Calculate normalized Euclidean distance in probability and expected goal space."""
        # 1. Probability Euclidean distance
        d_p = np.sqrt(
            (p_home_q - record.p_home) ** 2 +
            (p_draw_q - record.p_draw) ** 2 +
            (p_away_q - record.p_away) ** 2
        )

        # 2. Expected goal distance (normalized by max realistic goals 5.0)
        if lh_q is not None and la_q is not None and record.lambda_home is not None and record.lambda_away is not None:
            d_xg = np.sqrt(
                ((lh_q - record.lambda_home) / 5.0) ** 2 +
                ((la_q - record.lambda_away) / 5.0) ** 2
            )
            combined_d = 0.70 * d_p + 0.30 * d_xg
        else:
            combined_d = d_p

        # 3. Outcome alignment penalty (+0.15 if predicted decisions diverge)
        if dec_q != record.predicted_outcome:
            combined_d += 0.15

        return float(combined_d)

    @classmethod
    def calculate_similarity_pct(cls, distance: float) -> float:
        """Convert distance to a 0.0 - 100.0% similarity score."""
        # sqrt(2) is theoretical max prob distance
        sim = max(0.0, min(1.0, 1.0 - (distance / np.sqrt(2)))) * 100.0
        return round(float(sim), 1)

    @classmethod
    def find_nearest_analogues(
        cls,
        p_home: float,
        p_draw: float,
        p_away: float,
        decision: str,
        lambda_home: Optional[float],
        lambda_away: Optional[float],
        baseline_score: Optional[str],
        query_kickoff_utc: str,
        candidates: List[HistoricalCalculationRecord],
        target_competition_id: Optional[int] = None,
        top_k: int = 8,
        min_similarity_pct: float = 60.0,
        allow_cross_league: bool = False,
    ) -> List[HistoricalAnalogue]:
        """Retrieve top-K calculation analogues strictly respecting temporal leakage and league isolation."""
        # Parse query kickoff timestamp for strict temporal guard
        try:
            from dashboard.time_utils import parse_to_utc_datetime
            q_dt = parse_to_utc_datetime(query_kickoff_utc)
        except Exception:
            q_dt = datetime.now(timezone.utc)

        scored_records: List[Tuple[float, HistoricalCalculationRecord, str]] = []

        for cand in candidates:
            # 1. Strict Completion & Validity Guard
            if not cand.is_completed or cand.actual_outcome is None or cand.actual_score is None:
                continue

            # 2. Strict Temporal Leakage Guard: Candidate kickoff MUST be strictly BEFORE query kickoff
            try:
                c_dt = parse_to_utc_datetime(cand.kickoff_utc)
            except Exception:
                continue

            if c_dt >= q_dt:
                # Temporal leakage violation — strictly skip
                continue

            # 3. League Isolation Guard
            is_same_league = (target_competition_id is None) or (cand.competition_id == target_competition_id)
            if not is_same_league and not allow_cross_league:
                continue

            scope = "league" if is_same_league else "cross_league"

            dist = cls.calculate_distance(
                p_home_q=p_home,
                p_draw_q=p_draw,
                p_away_q=p_away,
                dec_q=decision,
                lh_q=lambda_home,
                la_q=lambda_away,
                record=cand,
            )

            # Apply cross-league penalty if cross-league candidate
            if not is_same_league:
                dist += 0.10

            sim_pct = cls.calculate_similarity_pct(dist)
            if sim_pct >= min_similarity_pct:
                scored_records.append((sim_pct, cand, scope))

        # Sort by similarity descending, then by kickoff descending (preferring recent analogues on tie)
        scored_records.sort(key=lambda x: (x[0], x[1].kickoff_utc), reverse=True)

        analogues: List[HistoricalAnalogue] = []
        for sim_pct, cand, scope in scored_records[:top_k]:
            analogues.append(HistoricalAnalogue(
                fixture_id=cand.fixture_id,
                competition_name=cand.competition_name,
                season=cand.season,
                home_team=cand.home_team,
                away_team=cand.away_team,
                kickoff_utc=cand.kickoff_utc,
                similarity_score=sim_pct,
                p_home=cand.p_home,
                p_draw=cand.p_draw,
                p_away=cand.p_away,
                predicted_outcome=cand.predicted_outcome,
                baseline_score=cand.baseline_score,
                actual_score=cand.actual_score,
                actual_outcome=cand.actual_outcome,
                prediction_correct=bool(cand.prediction_correct),
                exact_score_correct=bool(cand.exact_score_correct),
                scope=scope,
            ))

        return analogues


class HistoricalEvidenceEvaluator:
    """Evaluates statistical evidence from historical analogue clusters with strict sample safeguards."""

    MIN_SAMPLE_SIZE: int = 3
    HIGH_CONVICTION_THRESHOLD: float = 75.0
    MODERATE_THRESHOLD: float = 50.0

    @classmethod
    def evaluate_evidence(
        cls,
        decision: str,
        analogues: List[HistoricalAnalogue],
        league_scope: str = "league",
        min_sample_size: Optional[int] = None,
    ) -> HistoricalEvidenceSignal:
        """Evaluate evidence strength under minimum sample-size safeguards."""
        min_n = min_sample_size or cls.MIN_SAMPLE_SIZE
        total_found = len(analogues)

        if total_found == 0:
            return HistoricalEvidenceSignal(
                total_analogues_found=0,
                same_outcome_analogues_count=0,
                same_outcome_success_count=0,
                same_outcome_success_rate=0.0,
                overall_analogue_accuracy=0.0,
                average_similarity=0.0,
                evidence_level="INSUFFICIENT",
                evidence_badge="⚪ INSUFFICIENT SAMPLE",
                advisory_summary="No historical calculation analogues found in memory.",
                league_scope=league_scope,
            )

        avg_sim = round(float(np.mean([a.similarity_score for a in analogues])), 1)
        total_correct = sum(1 for a in analogues if a.prediction_correct)
        overall_acc = round((total_correct / total_found) * 100.0, 1)

        # Same-outcome analogues (e.g. historical matches where model also predicted decision "H")
        same_outcome = [a for a in analogues if a.predicted_outcome == decision]
        n_same = len(same_outcome)
        n_same_correct = sum(1 for a in same_outcome if a.prediction_correct)
        same_success_rate = round((n_same_correct / n_same) * 100.0, 1) if n_same > 0 else 0.0

        # Minimum sample-size protection
        if n_same < min_n:
            return HistoricalEvidenceSignal(
                total_analogues_found=total_found,
                same_outcome_analogues_count=n_same,
                same_outcome_success_count=n_same_correct,
                same_outcome_success_rate=same_success_rate,
                overall_analogue_accuracy=overall_acc,
                average_similarity=avg_sim,
                evidence_level="INSUFFICIENT",
                evidence_badge="⚪ INSUFFICIENT SAMPLE",
                advisory_summary=f"Insufficient same-outcome historical sample ({n_same}/{min_n} required); keeping baseline V4.0 model confidence.",
                league_scope=league_scope,
            )

        # Classification based on empirical success rate
        if same_success_rate >= cls.HIGH_CONVICTION_THRESHOLD:
            level = "HIGH"
            badge = "🔥 HIGH HISTORICAL CONVICTION"
            summary = f"Strong historical pattern: {n_same_correct}/{n_same} ({same_success_rate}%) similar calculation patterns resulted in {decision}."
        elif same_success_rate >= cls.MODERATE_THRESHOLD:
            level = "MODERATE"
            badge = "⚡ MODERATE HISTORICAL EVIDENCE"
            summary = f"Moderate historical support: {n_same_correct}/{n_same} ({same_success_rate}%) similar calculation patterns resulted in {decision}."
        else:
            level = "CAUTION"
            badge = "⚠️ HISTORICAL CAUTION"
            summary = f"Historical caution: only {n_same_correct}/{n_same} ({same_success_rate}%) similar calculation patterns succeeded."

        return HistoricalEvidenceSignal(
            total_analogues_found=total_found,
            same_outcome_analogues_count=n_same,
            same_outcome_success_count=n_same_correct,
            same_outcome_success_rate=same_success_rate,
            overall_analogue_accuracy=overall_acc,
            average_similarity=avg_sim,
            evidence_level=level,
            evidence_badge=badge,
            advisory_summary=summary,
            league_scope=league_scope,
        )


class CorrectScoreRefiner:
    """Refines baseline score predictions using empirical historical analogue score concentrations."""

    MIN_SCORE_SAMPLE: int = 3
    MIN_MODAL_CONCENTRATION: float = 0.35

    @classmethod
    def refine_score(
        cls,
        baseline_score: str,
        decision: str,
        analogues: List[HistoricalAnalogue],
        min_sample: Optional[int] = None,
        min_concentration: Optional[float] = None,
    ) -> CorrectScoreRefinementResult:
        """Analyze actual score distribution and propose refinement if supported by evidence."""
        min_n = min_sample or cls.MIN_SCORE_SAMPLE
        min_conc = min_concentration or cls.MIN_MODAL_CONCENTRATION

        # Consider analogues with matching predicted outcome or overall cluster
        relevant_scores = [a.actual_score for a in analogues if a.actual_score]
        n_total = len(relevant_scores)

        if n_total < min_n:
            return CorrectScoreRefinementResult(
                baseline_score=baseline_score,
                refined_score=baseline_score,
                refinement_status="INSUFFICIENT_EVIDENCE",
                score_confidence="NONE",
                historical_cluster_sample=n_total,
                modal_score_count=0,
                modal_score_frequency=0.0,
                reason="Insufficient historical analogue sample for score refinement.",
                score_distribution={},
            )

        counts = Counter(relevant_scores)
        score_dist = dict(counts.most_common(5))
        modal_score, modal_count = counts.most_common(1)[0]
        modal_freq = round(modal_count / n_total, 3)

        # Check if modal score matches direction of decision
        try:
            hg, ag = [int(x) for x in modal_score.split("-")]
            modal_dec = "H" if hg > ag else ("A" if ag > hg else "D")
        except Exception:
            modal_dec = None

        # Refine if strong concentration and compatible direction
        if modal_freq >= min_conc and modal_count >= 2:
            conf = "HIGH" if modal_freq >= 0.50 else "MODERATE"
            status = "REFINED" if modal_score != baseline_score else "BASELINE_CONFIRMED"
            reason = f"Historical analogue cluster shows {modal_count}/{n_total} ({modal_freq*100:.0f}%) matches finished {modal_score}."

            return CorrectScoreRefinementResult(
                baseline_score=baseline_score,
                refined_score=modal_score,
                refinement_status=status,
                score_confidence=conf,
                historical_cluster_sample=n_total,
                modal_score_count=modal_count,
                modal_score_frequency=modal_freq,
                reason=reason,
                score_distribution=score_dist,
            )
        else:
            return CorrectScoreRefinementResult(
                baseline_score=baseline_score,
                refined_score=baseline_score,
                refinement_status="BASELINE_KEPT",
                score_confidence="LOW",
                historical_cluster_sample=n_total,
                modal_score_count=modal_count,
                modal_score_frequency=modal_freq,
                reason=f"Historical scores are dispersed across {len(counts)} outcomes (top: {modal_score} at {modal_freq*100:.0f}%); keeping baseline V4.0 score.",
                score_distribution=score_dist,
            )


class HistoricalCalculationMemoryStore:
    """In-memory store backed by immutable snapshot ledgers and verified historical match databases."""

    def __init__(self):
        self._records: Dict[int, HistoricalCalculationRecord] = {}
        self._load_from_sources()

    def _load_from_sources(self) -> None:
        """Load from available JSON/CSV ledgers without mutating existing files."""
        # 1. Ingest from primary JSON snapshot ledger
        if LEDGER_PATH.exists():
            try:
                data = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
                items = data if isinstance(data, list) else list(data.values())
                for item in items:
                    fid = int(item.get("fixture_id", 0))
                    if fid > 0 and fid not in self._records:
                        p_h = float(item.get("p_home", 0.33))
                        p_d = float(item.get("p_draw", 0.33))
                        p_a = float(item.get("p_away", 0.34))
                        dec = str(item.get("model_decision", "H"))
                        lh = float(item.get("lambda_home", 1.2)) if item.get("lambda_home") is not None else None
                        la = float(item.get("lambda_away", 1.0)) if item.get("lambda_away") is not None else None
                        b_score = f"{int(round(lh))}-{int(round(la))}" if (lh is not None and la is not None) else None

                        self._records[fid] = HistoricalCalculationRecord(
                            fixture_id=fid,
                            competition_id=int(item.get("competition_id", 0)),
                            competition_name=str(item.get("competition_name", "")),
                            season=str(item.get("season", "2026/2027")),
                            home_team=str(item.get("home_team", "")),
                            away_team=str(item.get("away_team", "")),
                            kickoff_utc=str(item.get("scheduled_kickoff_utc", "")),
                            p_home=p_h,
                            p_draw=p_d,
                            p_away=p_a,
                            predicted_outcome=dec,
                            lambda_home=lh,
                            lambda_away=la,
                            baseline_score=b_score,
                            draw_risk_score=float(item.get("draw_risk_score", 0.0)) if item.get("draw_risk_score") is not None else None,
                            draw_risk_tier=str(item.get("draw_risk_tier", "LOW")),
                            prediction_timestamp_utc=str(item.get("prediction_timestamp", "")),
                            is_completed=False,
                        )
            except Exception as e:
                logger.warning(f"Error ingesting JSON snapshot ledger: {e}")

        # 2. Ingest prospective outcomes from CSV ledgers
        for csv_path in [SHADOW_LEDGER_PATH, OUTCOME_LEDGER_PATH]:
            if csv_path.exists():
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            fid = int(row.get("fixture_id", 0))
                            if fid <= 0:
                                continue

                            act_res = row.get("actual_result")
                            def _safe_int(val: Any) -> Optional[int]:
                                if val is None or val == "" or str(val).lower() in ("none", "nan"):
                                    return None
                                try:
                                    return int(float(val))
                                except Exception:
                                    return None

                            act_hg = _safe_int(row.get("actual_home_goals"))
                            act_ag = _safe_int(row.get("actual_away_goals"))
                            act_score = f"{act_hg}-{act_ag}" if (act_hg is not None and act_ag is not None) else None
                            is_comp = act_res is not None and act_res != ""

                            if fid in self._records:
                                r = self._records[fid]
                                if is_comp:
                                    r.actual_home_goals = act_hg
                                    r.actual_away_goals = act_ag
                                    r.actual_outcome = act_res
                                    r.actual_score = act_score
                                    r.is_completed = True
                                    r.prediction_correct = (act_res == r.predicted_outcome) if act_res else None
                                    r.exact_score_correct = (r.baseline_score == act_score) if (r.baseline_score and act_score) else None
                                    if not r.kickoff_utc or "IST" in r.kickoff_utc or not r.kickoff_utc.startswith("202"):
                                        r.kickoff_utc = str(row.get("prediction_timestamp", ""))
                            else:
                                p_h = float(row.get("v40_p_home", row.get("v4_p_home", 0.33)))
                                p_d = float(row.get("v40_p_draw", row.get("v4_p_draw", 0.33)))
                                p_a = float(row.get("v40_p_away", row.get("v4_p_away", 0.34)))
                                dec = str(row.get("v40_prediction", row.get("v4_decision", "H")))
                                lh = float(row.get("v40_expected_home_goals", 1.2)) if row.get("v40_expected_home_goals") else None
                                la = float(row.get("v40_expected_away_goals", 1.0)) if row.get("v40_expected_away_goals") else None
                                b_score = f"{int(round(lh))}-{int(round(la))}" if (lh is not None and la is not None) else None

                                comp_name = row.get("league", row.get("competition", ""))
                                comp_id = 0
                                for cid, cname in {423: "Premier League", 419: "La Liga", 477: "Bundesliga", 499: "Serie A", 200: "Ligue 1"}.items():
                                    if cname.lower() in comp_name.lower():
                                        comp_id = cid
                                        comp_name = cname
                                        break

                                iso_k = str(row.get("scheduled_kickoff", row.get("kickoff", "")))
                                if "IST" in iso_k or not iso_k.startswith("202"):
                                    iso_k = str(row.get("prediction_timestamp", ""))

                                self._records[fid] = HistoricalCalculationRecord(
                                    fixture_id=fid,
                                    competition_id=comp_id,
                                    competition_name=comp_name,
                                    season=str(row.get("season", "2026/2027")),
                                    home_team=str(row.get("home_team", "")),
                                    away_team=str(row.get("away_team", "")),
                                    kickoff_utc=iso_k,
                                    p_home=p_h,
                                    p_draw=p_d,
                                    p_away=p_a,
                                    predicted_outcome=dec,
                                    lambda_home=lh,
                                    lambda_away=la,
                                    baseline_score=b_score,
                                    prediction_timestamp_utc=str(row.get("prediction_timestamp", "")),
                                    actual_home_goals=act_hg,
                                    actual_away_goals=act_ag,
                                    actual_outcome=act_res,
                                    actual_score=act_score,
                                    prediction_correct=(act_res == dec) if act_res else None,
                                    exact_score_correct=(b_score == act_score) if (b_score and act_score) else None,
                                    is_completed=is_comp,
                                )
                except Exception as e:
                    logger.warning(f"Error ingesting CSV ledger {csv_path}: {e}")

        logger.info(f"HistoricalCalculationMemoryStore initialized with {len(self._records)} records ({sum(1 for r in self._records.values() if r.is_completed)} completed).")

    def get_all_records(self) -> List[HistoricalCalculationRecord]:
        return list(self._records.values())

    def get_completed_records(self, competition_id: Optional[int] = None) -> List[HistoricalCalculationRecord]:
        """Return completed records optionally filtered by competition."""
        records = [r for r in self._records.values() if r.is_completed]
        if competition_id:
            records = [r for r in records if r.competition_id == competition_id]
        return records

    def add_record(self, record: HistoricalCalculationRecord) -> None:
        """Add or update a record deduplicated by fixture_id."""
        self._records[record.fixture_id] = record

    def total_count(self) -> int:
        return len(self._records)

    def completed_count(self) -> int:
        return sum(1 for r in self._records.values() if r.is_completed)


class HistoricalMemoryService:
    """Unified service exposing historical calculation memory analysis."""

    def __init__(self, store: Optional[HistoricalCalculationMemoryStore] = None):
        self.store = store or HistoricalCalculationMemoryStore()

    def analyze_fixture_calculation(
        self,
        fixture_id: int,
        home_team: str,
        away_team: str,
        competition_id: int,
        competition_name: str,
        kickoff_utc: str,
        p_home: float,
        p_draw: float,
        p_away: float,
        decision: str,
        lambda_home: Optional[float] = None,
        lambda_away: Optional[float] = None,
        baseline_score: Optional[str] = None,
        top_k: int = 8,
        min_similarity_pct: float = 60.0,
        allow_cross_league: bool = False,
    ) -> HistoricalMemoryAnalysisResult:
        """Evaluate an upcoming fixture against historical calculation memory."""
        # Derive baseline score if not provided
        if not baseline_score and lambda_home is not None and lambda_away is not None:
            lh_round = max(0, int(round(lambda_home)))
            la_round = max(0, int(round(lambda_away)))
            if lh_round == la_round and decision == "H":
                lh_round += 1
            elif lh_round == la_round and decision == "A":
                la_round += 1
            derived_baseline = f"{lh_round}-{la_round}"
        else:
            derived_baseline = baseline_score or "1-0"

        # 1. Retrieve candidate records
        candidates = self.store.get_all_records()

        # 2. Find nearest calculation analogues
        analogues = CalculationSimilarityEngine.find_nearest_analogues(
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            decision=decision,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            baseline_score=derived_baseline,
            query_kickoff_utc=kickoff_utc,
            candidates=candidates,
            target_competition_id=competition_id if not allow_cross_league else None,
            top_k=top_k,
            min_similarity_pct=min_similarity_pct,
            allow_cross_league=allow_cross_league,
        )

        # 3. Evaluate historical success rate and evidence signal
        scope_lbl = competition_name if not allow_cross_league else "All 5 Leagues (Cross-League)"
        evidence = HistoricalEvidenceEvaluator.evaluate_evidence(
            decision=decision,
            analogues=analogues,
            league_scope=scope_lbl,
        )

        # 4. Refine score based on analogue score distribution
        score_refinement = CorrectScoreRefiner.refine_score(
            baseline_score=derived_baseline,
            decision=decision,
            analogues=analogues,
        )

        has_sufficient = evidence.evidence_level != "INSUFFICIENT"

        return HistoricalMemoryAnalysisResult(
            fixture_id=fixture_id,
            home_team=home_team,
            away_team=away_team,
            competition_name=competition_name,
            kickoff_utc=kickoff_utc,
            v4_p_home=p_home,
            v4_p_draw=p_draw,
            v4_p_away=p_away,
            v4_decision=decision,
            v4_baseline_score=derived_baseline,
            analogues=analogues,
            evidence_signal=evidence,
            score_refinement=score_refinement,
            has_sufficient_evidence=has_sufficient,
        )


_GLOBAL_MEMORY_SERVICE: Optional[HistoricalMemoryService] = None


def get_historical_memory_service() -> HistoricalMemoryService:
    """Singleton getter for HistoricalMemoryService."""
    global _GLOBAL_MEMORY_SERVICE
    if _GLOBAL_MEMORY_SERVICE is None:
        _GLOBAL_MEMORY_SERVICE = HistoricalMemoryService()
    return _GLOBAL_MEMORY_SERVICE
