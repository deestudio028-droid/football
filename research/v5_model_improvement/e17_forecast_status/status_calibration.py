"""Phase 42 — Experiment E17: Status Taxonomy Calibration & Statistical Separation.

Computes:
- Status category breakdowns (STRONG, LEAN, CAUTION, AVOID)
- Monotonicity checks across realized RPS and Log-Loss
- Cluster bootstrap hypothesis testing for pairwise status separation
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_status_breakdown_table(
    status_assignments: np.ndarray,
    actual_match_rps: np.ndarray,
    actual_match_ll: np.ndarray,
    actual_match_brier: np.ndarray,
    top1_correct_binary: np.ndarray,
    entropy_arr: np.ndarray,
    max_prob_arr: np.ndarray,
    margin_arr: np.ndarray,
    probs_e10: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Computes comprehensive metrics per status category."""
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] if isinstance(y, str) else int(y) for y in y_true])
    n_total = len(actual_match_rps)

    status_order = ["STRONG", "LEAN", "CAUTION", "AVOID"]
    records = []

    for st in status_order:
        mask = (status_assignments == st)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue

        p_subset = probs_e10[mask]
        y_subset = y_idx[mask]

        # Draw ECE
        bins = np.linspace(0.0, 1.0, 11)
        d_probs = p_subset[:, 1]
        y_d = (y_subset == 1).astype(float)
        bin_indices = np.digitize(d_probs, bins) - 1
        d_ece = 0.0
        for b in range(10):
            b_mask = (bin_indices == b)
            c_bin = int(np.sum(b_mask))
            if c_bin > 0:
                bin_acc = np.mean(y_d[b_mask])
                bin_conf = np.mean(d_probs[b_mask])
                d_ece += (c_bin / cnt) * abs(bin_acc - bin_conf)

        records.append({
            "status": st,
            "n_matches": cnt,
            "coverage_pct": round(float(cnt / n_total * 100.0), 2),
            "actual_mean_rps": round(float(np.mean(actual_match_rps[mask])), 6),
            "actual_mean_log_loss": round(float(np.mean(actual_match_ll[mask])), 6),
            "actual_mean_brier": round(float(np.mean(actual_match_brier[mask])), 6),
            "top1_accuracy_pct": round(float(np.mean(top1_correct_binary[mask]) * 100.0), 2),
            "draw_ece": round(d_ece, 4),
            "mean_entropy": round(float(np.mean(entropy_arr[mask])), 4),
            "mean_max_prob": round(float(np.mean(max_prob_arr[mask])), 4),
            "mean_margin": round(float(np.mean(margin_arr[mask])), 4),
        })

    return pd.DataFrame(records)


def compute_status_separation_bootstrap(
    status_assignments: np.ndarray,
    actual_match_rps: np.ndarray,
    matchweek_clusters: np.ndarray,
    n_replicates: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Computes cluster bootstrap tests for pairwise status separation."""
    rng = np.random.default_rng(random_seed)
    unique_clusters = np.unique(matchweek_clusters)
    n_clusters = len(unique_clusters)

    cluster_to_indices = {c: np.where(matchweek_clusters == c)[0] for c in unique_clusters}

    comparisons = [
        ("STRONG", "AVOID"),
        ("STRONG", "CAUTION"),
        ("LEAN", "AVOID"),
    ]

    boot_diffs = {f"{c1}_vs_{c2}": [] for c1, c2 in comparisons}

    for _ in range(n_replicates):
        samp_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        samp_idx = np.concatenate([cluster_to_indices[c] for c in samp_clusters])

        samp_status = status_assignments[samp_idx]
        samp_rps = actual_match_rps[samp_idx]

        for c1, c2 in comparisons:
            m1 = samp_status == c1
            m2 = samp_status == c2
            if np.sum(m1) > 0 and np.sum(m2) > 0:
                diff = float(np.mean(samp_rps[m1]) - np.mean(samp_rps[m2]))
                boot_diffs[f"{c1}_vs_{c2}"].append(diff)

    results = {}
    for comp, diffs in boot_diffs.items():
        arr = np.array(diffs)
        mean_diff = float(np.mean(arr))
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        p_val = float(np.mean(arr >= 0.0))  # P(diff >= 0 when expecting diff < 0)
        results[comp] = {
            "delta_rps": round(mean_diff, 6),
            "ci_95": [round(ci_lower, 6), round(ci_upper, 6)],
            "p_value": round(p_val, 4),
            "is_significant": bool(ci_upper < 0.0),
        }

    return results
