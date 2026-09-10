"""Phase 40 — Experiment E15: Selective Forecasting & Risk-Based Routing Policies.

Implements:
1. Risk-Coverage Curves (Retained forecast accuracy across 100% to 50% coverage).
2. Routing Policies (Mode A Pure E10, Mode B Flagging, Mode C Abstention, Mode D E13 Routing, Mode E E14 Specialist Routing).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_risk_coverage_curve(
    predicted_risk_scores: np.ndarray,
    probs_e10: np.ndarray,
    y_true: np.ndarray,
    coverage_levels: List[float] = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50],
) -> pd.DataFrame:
    """Evaluates retained forecast metrics under selective prediction (abstaining on highest risk)."""
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] if isinstance(y, str) else int(y) for y in y_true])
    n_total = len(y_true)

    records = []
    for cov in coverage_levels:
        if cov == 1.0:
            mask = np.ones(n_total, dtype=bool)
        else:
            pct_keep = cov * 100.0
            cutoff = float(np.percentile(predicted_risk_scores, pct_keep))
            mask = (predicted_risk_scores <= cutoff)

        n_retained = int(np.sum(mask))
        if n_retained == 0:
            continue

        y_ret = y_idx[mask]
        p_ret = probs_e10[mask]

        # RPS
        e1 = (p_ret[:, 0] - (y_ret == 0).astype(float)) ** 2
        e2 = ((p_ret[:, 0] + p_ret[:, 1]) - (y_ret <= 1).astype(float)) ** 2
        ret_rps = float(0.5 * np.mean(e1 + e2))

        # Log Loss
        clipped_p = np.clip(p_ret, 1e-15, 1.0 - 1e-15)
        clipped_p /= clipped_p.sum(axis=1, keepdims=True)
        ret_ll = float(-np.mean(np.log(clipped_p[np.arange(n_retained), y_ret])))

        # Draw ECE
        bins = np.linspace(0.0, 1.0, 11)
        d_probs = p_ret[:, 1]
        y_d = (y_ret == 1).astype(float)
        bin_indices = np.digitize(d_probs, bins) - 1
        d_ece = 0.0
        for b in range(10):
            b_mask = (bin_indices == b)
            cnt = int(np.sum(b_mask))
            if cnt > 0:
                bin_acc = np.mean(y_d[b_mask])
                bin_conf = np.mean(d_probs[b_mask])
                d_ece += (cnt / n_retained) * abs(bin_acc - bin_conf)

        records.append({
            "coverage_pct": int(round(cov * 100)),
            "retained_matches": n_retained,
            "abstain_count": n_total - n_retained,
            "retained_rps": round(ret_rps, 6),
            "retained_log_loss": round(ret_ll, 6),
            "draw_ece": round(d_ece, 4),
        })

    return pd.DataFrame(records)


def evaluate_routing_policies(
    predicted_risk_scores: np.ndarray,
    probs_e10: np.ndarray,
    probs_e13: np.ndarray,
    is_early_3: np.ndarray,
    y_true: np.ndarray,
    risk_threshold_top10: float,
) -> Dict[str, Any]:
    """Evaluates multi-model routing policies against Pure E10 baseline."""
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] if isinstance(y, str) else int(y) for y in y_true])
    n = len(y_true)

    # 1. Mode A: Pure E10
    e1_a = (probs_e10[:, 0] - (y_idx == 0).astype(float)) ** 2
    e2_a = ((probs_e10[:, 0] + probs_e10[:, 1]) - (y_idx <= 1).astype(float)) ** 2
    rps_a = float(0.5 * np.mean(e1_a + e2_a))

    # 2. Mode D: Route top 10% risk matches to E13
    is_high_risk = (predicted_risk_scores >= risk_threshold_top10)
    probs_d = np.where(is_high_risk[:, None], probs_e13, probs_e10)
    e1_d = (probs_d[:, 0] - (y_idx == 0).astype(float)) ** 2
    e2_d = ((probs_d[:, 0] + probs_d[:, 1]) - (y_idx <= 1).astype(float)) ** 2
    rps_d = float(0.5 * np.mean(e1_d + e2_d))

    # 3. Mode E: Route early-season top-risk matches to E14 blend (70% E13, 30% E10)
    is_early_high_risk = (is_early_3 == 1) & is_high_risk
    probs_e = np.where(is_early_high_risk[:, None], 0.70 * probs_e13 + 0.30 * probs_e10, probs_e10)
    e1_e = (probs_e[:, 0] - (y_idx == 0).astype(float)) ** 2
    e2_e = ((probs_e[:, 0] + probs_e[:, 1]) - (y_idx <= 1).astype(float)) ** 2
    rps_e = float(0.5 * np.mean(e1_e + e2_e))

    return {
        "mode_a_pure_e10_rps": round(rps_a, 6),
        "mode_d_route_high_risk_e13_rps": round(rps_d, 6),
        "mode_d_delta_rps": round(rps_d - rps_a, 6),
        "mode_e_route_early_risk_e14_rps": round(rps_e, 6),
        "mode_e_delta_rps": round(rps_e - rps_a, 6),
        "high_risk_match_count": int(np.sum(is_high_risk)),
        "early_high_risk_match_count": int(np.sum(is_early_high_risk)),
    }
