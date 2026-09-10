"""Draw Probability Calibrator Component.

Production-Safe Shadow Layer for Continuous Draw Probability Refinement.
Operates strictly as a post-inference continuous calibrator without modifying
base model weights, feature generators, or classification rules.

Invariants Enforced:
1. P'(H) >= 0, P'(D) >= 0, P'(A) >= 0
2. P'(H) <= 1, P'(D) <= 1, P'(A) <= 1
3. P'(H) + P'(D) + P'(A) == 1.0 within numerical tolerance
4. P'(H) / P'(A) == P(H) / P(A) within numerical tolerance (Home/Away ratio preservation)
5. Zero mutation of input objects or upstream models
6. Zero hard decision gates (Argmax decision policy preserved)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class CalibratedPredictionResult:
    """Immutable result container holding both original and calibrated probabilities."""
    # Original V4.0 outputs
    original_p_home: float
    original_p_draw: float
    original_p_away: float
    original_decision: str

    # Calibrated outputs
    calibrated_p_home: float
    calibrated_p_draw: float
    calibrated_p_away: float
    calibrated_decision: str

    # Metadata & Deltas
    draw_delta: float
    home_delta: float
    away_delta: float
    decision_changed: bool
    calibrator_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": {
                "H": self.original_p_home,
                "D": self.original_p_draw,
                "A": self.original_p_away,
                "decision": self.original_decision,
            },
            "calibrated": {
                "H": self.calibrated_p_home,
                "D": self.calibrated_p_draw,
                "A": self.calibrated_p_away,
                "decision": self.calibrated_decision,
            },
            "deltas": {
                "H": self.home_delta,
                "D": self.draw_delta,
                "A": self.away_delta,
            },
            "decision_changed": self.decision_changed,
            "version": self.calibrator_version,
        }


class DrawProbabilityCalibrator:
    """Isolated shadow continuous probability calibrator."""

    def __init__(
        self,
        slope_a: float = 0.9421034293933221,
        intercept_b: float = 0.12836262923594244,
        version: str = "v1.0-step2e-platt-shadow",
        eps: float = 1e-12,
    ) -> None:
        self.slope_a = float(slope_a)
        self.intercept_b = float(intercept_b)
        self.version = str(version)
        self.eps = float(eps)

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "DrawProbabilityCalibrator":
        """Load calibrator from frozen JSON configuration."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data.get("parameters", {})
        return cls(
            slope_a=params.get("slope_a", 0.9421034293933221),
            intercept_b=params.get("intercept_b", 0.12836262923594244),
            version=data.get("version", "v1.0-step2e-platt-shadow"),
        )

    def _stable_logit(self, p: float | np.ndarray) -> float | np.ndarray:
        """Numerically stable logit function clipped to avoid singularity."""
        p_c = np.clip(p, self.eps, 1.0 - self.eps)
        return np.log(p_c / (1.0 - p_c))

    def _stable_sigmoid(self, z: float | np.ndarray) -> float | np.ndarray:
        """Numerically stable sigmoid function."""
        z_c = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z_c))

    def calibrate_draw_probability(self, p_draw: float | np.ndarray) -> float | np.ndarray:
        """Calculate calibrated continuous Draw probability P'(D)."""
        z_raw = self._stable_logit(p_draw)
        z_cal = self.slope_a * z_raw + self.intercept_b
        return self._stable_sigmoid(z_cal)

    def calibrate_single(
        self,
        p_home: float,
        p_draw: float,
        p_away: float,
    ) -> CalibratedPredictionResult:
        """Calibrate a single 3-class probability distribution with invariant guarantees."""
        # 1. Validation of inputs
        if any(np.isnan([p_home, p_draw, p_away])) or any(np.isinf([p_home, p_draw, p_away])):
            raise ValueError(f"Invalid non-finite input probabilities: H={p_home}, D={p_draw}, A={p_away}")

        # Normalize input to perfect sum 1.0 defensively
        raw_sum = p_home + p_draw + p_away
        if raw_sum <= 0:
            raise ValueError(f"Probability sum non-positive: {raw_sum}")
        ph = float(p_home / raw_sum)
        pd_ = float(p_draw / raw_sum)
        pa = float(p_away / raw_sum)

        # Original decision via argmax
        probs_orig = {"H": ph, "D": pd_, "A": pa}
        orig_dec = max(probs_orig, key=probs_orig.get)

        # 2. Calibrate Draw probability
        pd_cal = float(self.calibrate_draw_probability(pd_))

        # 3. Proportional redistribution of non-draw mass to preserve H/A ratio
        non_draw_old = max(self.eps, 1.0 - pd_)
        non_draw_new = max(0.0, 1.0 - pd_cal)
        scale = non_draw_new / non_draw_old

        ph_cal = float(ph * scale)
        pa_cal = float(pa * scale)

        # Defensive precision normalization
        s_cal = ph_cal + pd_cal + pa_cal
        if s_cal > 0:
            ph_cal /= s_cal
            pd_cal /= s_cal
            pa_cal /= s_cal

        # 4. Calibrated decision via standard argmax (ZERO hard overrides)
        probs_cal = {"H": ph_cal, "D": pd_cal, "A": pa_cal}
        cal_dec = max(probs_cal, key=probs_cal.get)

        # 5. Invariant assertions
        assert 0.0 <= ph_cal <= 1.0, f"Calibrated P(H) out of bounds: {ph_cal}"
        assert 0.0 <= pd_cal <= 1.0, f"Calibrated P(D) out of bounds: {pd_cal}"
        assert 0.0 <= pa_cal <= 1.0, f"Calibrated P(A) out of bounds: {pa_cal}"
        assert abs((ph_cal + pd_cal + pa_cal) - 1.0) < 1e-7, "Probability sum invariant violated"

        # Check Home/Away ratio preservation if both are non-zero
        if ph > 1e-6 and pa > 1e-6 and ph_cal > 1e-6 and pa_cal > 1e-6:
            ratio_orig = ph / pa
            ratio_cal = ph_cal / pa_cal
            assert abs(ratio_orig - ratio_cal) < 1e-5, f"Home/Away ratio altered: {ratio_orig} vs {ratio_cal}"

        return CalibratedPredictionResult(
            original_p_home=float(ph),
            original_p_draw=float(pd_),
            original_p_away=float(pa),
            original_decision=orig_dec,
            calibrated_p_home=float(ph_cal),
            calibrated_p_draw=float(pd_cal),
            calibrated_p_away=float(pa_cal),
            calibrated_decision=cal_dec,
            draw_delta=float(pd_cal - pd_),
            home_delta=float(ph_cal - ph),
            away_delta=float(pa_cal - pa),
            decision_changed=(orig_dec != cal_dec),
            calibrator_version=self.version,
        )

    def calibrate_array(self, probs_3c: np.ndarray) -> np.ndarray:
        """Batch calibrate an (N, 3) matrix of [P(H), P(D), P(A)] probabilities."""
        if probs_3c.ndim != 2 or probs_3c.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) probability array, got shape {probs_3c.shape}")

        p_h = probs_3c[:, 0]
        p_d = probs_3c[:, 1]
        p_a = probs_3c[:, 2]

        p_d_cal = self.calibrate_draw_probability(p_d)
        denom = np.clip(1.0 - p_d, self.eps, 1.0)
        scale = np.clip(1.0 - p_d_cal, 0.0, 1.0) / denom

        p_h_cal = p_h * scale
        p_a_cal = p_a * scale

        res = np.column_stack([p_h_cal, p_d_cal, p_a_cal])
        s = np.sum(res, axis=1, keepdims=True)
        return res / s
