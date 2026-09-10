"""Metrics & Analysis Service for Football Prediction Lab Dashboard.

Calculates multi-model scorecards, error economics, league breakdowns, temporal slices,
model agreement metrics, and 10,000 paired bootstrap statistical confidence intervals.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MetricsService:
    """Statistical and metric calculation engine for dashboard reporting."""

    @staticmethod
    def calculate_scorecard(y_true: List[str], y_prob: np.ndarray, y_pred: List[str]) -> Dict[str, Any]:
        n = len(y_true)
        if n == 0:
            return {
                "accuracy_pct": 0.0, "correct": 0, "wrong": 0, "n": 0,
                "draw_predictions": 0, "correct_draws": 0, "draw_precision_pct": 0.0,
                "draw_recall_pct": 0.0, "draw_f1": 0.0, "macro_f1": 0.0,
                "log_loss": 0.0, "brier_score": 0.0, "rps": 0.0, "draw_ece": 0.0,
            }

        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        acc = correct / n

        d_preds = sum(1 for p in y_pred if p == "D")
        c_draws = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "D" and yp == "D")
        act_draws = sum(1 for yt in y_true if yt == "D")

        draw_prec = (c_draws / d_preds * 100.0) if d_preds > 0 else 0.0
        draw_rec = (c_draws / act_draws * 100.0) if act_draws > 0 else 0.0
        draw_f1 = (2 * (draw_prec / 100.0) * (draw_rec / 100.0)) / ((draw_prec / 100.0) + (draw_rec / 100.0)) if (draw_prec + draw_rec) > 0 else 0.0

        f1_list = []
        for c in ["H", "D", "A"]:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
            pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0
            f1_list.append(f)
        macro_f1 = float(np.mean(f1_list))

        # Log Loss & Brier & RPS
        mapping = {"H": 0, "D": 1, "A": 2}
        eps = 1e-15
        probs = np.clip(y_prob, eps, 1.0 - eps)
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_idx = np.array([mapping[y] for y in y_true])
        ll = float(-np.mean(np.log(probs[np.arange(n), y_idx])))

        y_onehot = np.zeros_like(y_prob)
        for i, y in enumerate(y_true):
            y_onehot[i, mapping[y]] = 1.0
        bs = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

        rps_scores = []
        for i, y in enumerate(y_true):
            act_idx = mapping[y]
            e1 = (y_prob[i, 0] - (1.0 if act_idx == 0 else 0.0)) ** 2
            e2 = ((y_prob[i, 0] + y_prob[i, 1]) - (1.0 if act_idx <= 1 else 0.0)) ** 2
            rps_scores.append(0.5 * (e1 + e2))
        rps = float(np.mean(rps_scores))

        # Draw ECE
        bins = np.linspace(0.0, 1.0, 11)
        d_probs = y_prob[:, 1]
        y_d = np.array([1.0 if yt == "D" else 0.0 for yt in y_true])
        bin_indices = np.digitize(d_probs, bins) - 1
        ece = 0.0
        for b in range(10):
            mask = (bin_indices == b)
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_d[mask])
                bin_conf = np.mean(d_probs[mask])
                ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

        return {
            "accuracy_pct": round(acc * 100.0, 2),
            "correct": correct,
            "wrong": n - correct,
            "n": n,
            "draw_predictions": d_preds,
            "correct_draws": c_draws,
            "draw_precision_pct": round(draw_prec, 1),
            "draw_recall_pct": round(draw_rec, 1),
            "draw_f1": round(draw_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "log_loss": round(ll, 6),
            "brier_score": round(bs, 6),
            "rps": round(rps, 6),
            "draw_ece": round(ece, 4),
            "mean_p_draw": round(float(np.mean(d_probs)), 4),
            "actual_draw_rate": round(float(act_draws / n), 4),
        }

    @staticmethod
    def calculate_error_economics(y_true: List[str], p_base: List[str], p_override: List[str]) -> Dict[str, Any]:
        gained = sum(1 for yt, pb, po in zip(y_true, p_base, p_override) if po == "D" and pb != yt and yt == "D")
        lost = sum(1 for yt, pb, po in zip(y_true, p_base, p_override) if po == "D" and pb == yt and yt != "D")
        neutral = sum(1 for yt, pb, po in zip(y_true, p_base, p_override) if po == "D" and pb != yt and yt != "D")
        tot = gained + lost + neutral
        return {
            "good_draw_overrides_GAINED": gained,
            "bad_draw_overrides_SACRIFICED": lost,
            "neutral_overrides": neutral,
            "total_overrides": tot,
            "net_gain": gained - lost,
            "override_precision_pct": round((gained / tot * 100.0), 2) if tot > 0 else 0.0,
        }

    @staticmethod
    def calculate_model_agreement(
        dec_v4: List[str],
        dec_v46: List[str],
        dec_hist_h: List[str],
    ) -> Dict[str, Any]:
        n = len(dec_v4)
        if n == 0:
            return {"n": 0}

        v4_v46_agree = sum(1 for p1, p2 in zip(dec_v4, dec_v46) if p1 == p2)
        v4_h_agree = sum(1 for p1, p2 in zip(dec_v4, dec_hist_h) if p1 == p2)
        v46_h_agree = sum(1 for p1, p2 in zip(dec_v46, dec_hist_h) if p1 == p2)

        return {
            "n_fixtures": n,
            "v4_vs_v46_agreement_pct": round(v4_v46_agree / n * 100.0, 2),
            "v4_vs_v46_divergence_count": n - v4_v46_agree,
            "v4_vs_h_agreement_pct": round(v4_h_agree / n * 100.0, 2),
            "v4_vs_h_divergence_count": n - v4_h_agree,
            "v46_vs_h_agreement_pct": round(v46_h_agree / n * 100.0, 2),
            "v46_vs_h_divergence_count": n - v46_h_agree,
        }

    @staticmethod
    def run_paired_bootstrap(
        y_true: List[str],
        dec_base: List[str],
        dec_cand: List[str],
        n_resamples: int = 10000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        n = len(y_true)
        if n < 10:
            return {
                "n": n,
                "status": "SAMPLE_TOO_SMALL",
                "mean_delta_acc": 0.0,
                "ci_95": [0.0, 0.0],
                "p_cand_ge_base": 50.0,
            }

        np.random.seed(seed)
        y_arr = np.array(y_true)
        db_arr = np.array(dec_base)
        dc_arr = np.array(dec_cand)

        deltas = []
        for _ in range(n_resamples):
            idx = np.random.randint(0, n, size=n)
            acc_base = np.mean(db_arr[idx] == y_arr[idx]) * 100.0
            acc_cand = np.mean(dc_arr[idx] == y_arr[idx]) * 100.0
            deltas.append(acc_cand - acc_base)

        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
        p_ge = float(np.mean(np.array(deltas) >= 0.0) * 100.0)
        p_gt = float(np.mean(np.array(deltas) > 0.0) * 100.0)

        is_conclusive = bool(ci_low > 0.0 or ci_high < 0.0)

        return {
            "n": n,
            "n_resamples": n_resamples,
            "mean_delta_accuracy_pct": round(float(np.mean(deltas)), 4),
            "ci_95_accuracy_pct": [round(float(ci_low), 4), round(float(ci_high), 4)],
            "p_cand_ge_base_pct": round(p_ge, 1),
            "p_cand_gt_base_pct": round(p_gt, 1),
            "is_statistically_conclusive": is_conclusive,
            "conclusion_text": "STATISTICALLY SIGNIFICANT" if is_conclusive else "NOT STATISTICALLY CONCLUSIVE (CI crosses zero)",
        }
