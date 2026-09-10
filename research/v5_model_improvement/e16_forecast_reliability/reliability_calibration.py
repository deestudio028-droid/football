"""Phase 41 — Experiment E16: Reliability Calibration & Evaluation Suite.

Evaluates:
- Discrimination: ROC-AUC, PR-AUC
- Ranking: Spearman rank correlation with realized error
- Calibration: Calibration Slope, Calibration Intercept, ECE, Brier Score
- Error Accuracy: MAE, RMSE between predicted expected loss and realized loss
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


def evaluate_reliability_calibration(
    predicted_expected_rps: np.ndarray,
    predicted_reliability_score: np.ndarray,
    actual_match_rps: np.ndarray,
    top1_correct_binary: np.ndarray,
) -> Dict[str, Any]:
    """Computes comprehensive discrimination, ranking, calibration, and error estimation metrics."""
    n = len(actual_match_rps)
    if n == 0:
        return {}

    # 1. Ranking & Linear Correlation with Actual Realized RPS
    spearman_res = spearmanr(predicted_expected_rps, actual_match_rps)
    spearman_corr = float(spearman_res.statistic) if hasattr(spearman_res, "statistic") else float(spearman_res[0])
    pearson_corr = float(np.corrcoef(predicted_expected_rps, actual_match_rps)[0, 1])

    # 2. Prediction Accuracy of Expected Loss
    mae = float(np.mean(np.abs(predicted_expected_rps - actual_match_rps)))
    rmse = float(np.sqrt(np.mean((predicted_expected_rps - actual_match_rps) ** 2)))

    # 3. Calibration Slope & Intercept (OLS: actual_rps ~ alpha + beta * pred_rps)
    cov_matrix = np.cov(predicted_expected_rps, actual_match_rps)
    var_pred = float(cov_matrix[0, 0])
    cov_pred_act = float(cov_matrix[0, 1])
    slope = cov_pred_act / var_pred if var_pred > 1e-12 else 1.0
    intercept = float(np.mean(actual_match_rps) - slope * np.mean(predicted_expected_rps))

    # 4. Discrimination on Top-1 Outcome Correctness
    # Higher reliability -> higher chance top-1 is correct
    if len(np.unique(top1_correct_binary)) > 1:
        roc_auc = float(roc_auc_score(top1_correct_binary, predicted_reliability_score))
        pr_auc = float(average_precision_score(top1_correct_binary, predicted_reliability_score))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(top1_correct_binary))

    # 5. Expected Calibration Error (ECE) for Top-1 Correctness
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(predicted_reliability_score, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = (bin_indices == b)
        cnt = int(np.sum(mask))
        if cnt > 0:
            bin_acc = float(np.mean(top1_correct_binary[mask]))
            bin_conf = float(np.mean(predicted_reliability_score[mask]))
            ece += (cnt / n) * abs(bin_acc - bin_conf)

    # 6. Brier Score of Reliability Score vs Realized Top-1 Correctness
    brier = float(brier_score_loss(top1_correct_binary, np.clip(predicted_reliability_score, 0.0, 1.0)))

    return {
        "spearman_corr_rps": round(spearman_corr, 4),
        "pearson_corr_rps": round(pearson_corr, 4),
        "mae_expected_rps": round(mae, 6),
        "rmse_expected_rps": round(rmse, 6),
        "calibration_slope": round(slope, 4),
        "calibration_intercept": round(intercept, 6),
        "roc_auc_correctness": round(roc_auc, 4),
        "pr_auc_correctness": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "reliability_ece": round(ece, 4),
    }


def compute_reliability_decile_table(
    predicted_expected_rps: np.ndarray,
    predicted_reliability_score: np.ndarray,
    actual_match_rps: np.ndarray,
    actual_match_ll: np.ndarray,
    match_entropy: np.ndarray,
    max_prob: np.ndarray,
    top1_correct_binary: np.ndarray,
) -> pd.DataFrame:
    """Stratifies fixtures into 10 reliability deciles (Decile 1 = lowest reliability / highest error, Decile 10 = highest reliability)."""
    df = pd.DataFrame({
        "reliability_score": predicted_reliability_score,
        "pred_expected_rps": predicted_expected_rps,
        "actual_rps": actual_match_rps,
        "actual_log_loss": actual_match_ll,
        "entropy": match_entropy,
        "max_prob": max_prob,
        "top1_correct": top1_correct_binary,
    })

    # Sort into 10 deciles (1 = Lowest predicted reliability, 10 = Highest predicted reliability)
    df["decile"] = pd.qcut(df["reliability_score"], q=10, labels=range(1, 11), duplicates="drop")

    records = []
    for d, group in df.groupby("decile", observed=False):
        records.append({
            "decile": int(d),
            "n_matches": len(group),
            "mean_reliability_score": round(float(group["reliability_score"].mean()), 4),
            "mean_predicted_rps": round(float(group["pred_expected_rps"].mean()), 6),
            "actual_mean_rps": round(float(group["actual_rps"].mean()), 6),
            "actual_mean_log_loss": round(float(group["actual_log_loss"].mean()), 6),
            "top1_accuracy_pct": round(float(group["top1_correct"].mean() * 100.0), 2),
            "mean_entropy": round(float(group["entropy"].mean()), 4),
            "mean_max_prob": round(float(group["max_prob"].mean()), 4),
        })

    return pd.DataFrame(records)
