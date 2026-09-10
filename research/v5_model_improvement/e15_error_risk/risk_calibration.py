"""Phase 40 — Experiment E15: Risk Model Calibration & Decile Breakdown.

Computes:
- ROC-AUC, PR-AUC, Brier score, and calibration metrics for risk scores
- Decile stratification tables (Deciles 1 to 10 sorted by predicted risk)
- Monotonicity checks between predicted risk and realized match error
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def evaluate_risk_model_calibration(
    y_true_binary: np.ndarray,
    predicted_risk_scores: np.ndarray,
) -> Dict[str, Any]:
    """Computes discrimination and calibration metrics for risk scores."""
    n = len(y_true_binary)
    if n == 0 or len(np.unique(y_true_binary)) < 2:
        return {}

    auc_roc = float(roc_auc_score(y_true_binary, predicted_risk_scores))
    auc_pr = float(average_precision_score(y_true_binary, predicted_risk_scores))
    brier = float(brier_score_loss(y_true_binary, predicted_risk_scores))

    # 10-bin Expected Calibration Error
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(predicted_risk_scores, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = (bin_indices == b)
        cnt = int(np.sum(mask))
        if cnt > 0:
            bin_acc = np.mean(y_true_binary[mask])
            bin_conf = np.mean(predicted_risk_scores[mask])
            ece += (cnt / n) * abs(bin_acc - bin_conf)

    # Top-10% and Top-20% Recall
    thresh_top10 = float(np.percentile(predicted_risk_scores, 90.0))
    thresh_top20 = float(np.percentile(predicted_risk_scores, 80.0))

    pred_top10 = (predicted_risk_scores >= thresh_top10).astype(int)
    pred_top20 = (predicted_risk_scores >= thresh_top20).astype(int)

    act_positives = np.sum(y_true_binary == 1)
    recall_top10 = float(np.sum((y_true_binary == 1) & (pred_top10 == 1)) / act_positives) if act_positives > 0 else 0.0
    recall_top20 = float(np.sum((y_true_binary == 1) & (pred_top20 == 1)) / act_positives) if act_positives > 0 else 0.0

    return {
        "roc_auc": round(auc_roc, 4),
        "pr_auc": round(auc_pr, 4),
        "brier_score": round(brier, 4),
        "risk_ece": round(ece, 4),
        "recall_at_top10": round(recall_top10 * 100.0, 2),
        "recall_at_top20": round(recall_top20 * 100.0, 2),
    }


def compute_risk_deciles_breakdown(
    predicted_risk_scores: np.ndarray,
    actual_match_rps: np.ndarray,
    actual_match_ll: np.ndarray,
    match_entropy: np.ndarray,
    max_prob: np.ndarray,
    actual_results: np.ndarray,
) -> pd.DataFrame:
    """Stratifies out-of-sample fixtures into 10 risk deciles to check monotonicity."""
    df = pd.DataFrame({
        "risk_score": predicted_risk_scores,
        "rps": actual_match_rps,
        "log_loss": actual_match_ll,
        "entropy": match_entropy,
        "max_prob": max_prob,
        "is_draw": (actual_results == "D").astype(int),
    })

    # Rank into 10 deciles (1 = lowest predicted risk, 10 = highest predicted risk)
    df["decile"] = pd.qcut(df["risk_score"], q=10, labels=range(1, 11), duplicates="drop")

    decile_records = []
    for d, group in df.groupby("decile", observed=False):
        decile_records.append({
            "decile": int(d),
            "n_matches": len(group),
            "mean_predicted_risk": round(float(group["risk_score"].mean()), 4),
            "actual_mean_rps": round(float(group["rps"].mean()), 6),
            "actual_mean_log_loss": round(float(group["log_loss"].mean()), 6),
            "mean_entropy": round(float(group["entropy"].mean()), 4),
            "mean_max_prob": round(float(group["max_prob"].mean()), 4),
            "draw_rate_pct": round(float(group["is_draw"].mean() * 100.0), 2),
        })

    return pd.DataFrame(decile_records)
