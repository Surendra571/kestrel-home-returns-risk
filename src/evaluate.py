"""Comprehensive model evaluation module for Kestrel Home Returns Risk.

Computes statistical classification metrics, probability calibration metrics,
and operational business cost metrics for model evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.business_metrics import calculate_business_metrics


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculates all statistical, calibration, and business metrics for a model.

    Args:
        y_true: Ground truth binary labels (1 = returned, 0 = retained).
        y_pred: Binary predicted decisions (0 or 1).
        y_prob: Continuous predicted return risk probabilities in [0.0, 1.0].
        threshold: Decision threshold used to generate binary decisions from probabilities.

    Returns:
        Structured dictionary containing statistical, calibration, and business metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    # 1. Classification Performance Metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    results: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "return_class_recall": round(rec, 4),
        "return_class_precision": round(prec, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "threshold": round(float(threshold), 4) if threshold is not None else None,
    }

    # 2. Ranking & Calibration Metrics (if probabilities are available)
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        # Clip to avoid log(0) in log_loss
        y_prob_clipped = np.clip(y_prob_arr, 1e-15, 1.0 - 1e-15)

        # ROC-AUC requires both classes in y_true
        if len(np.unique(y_true_arr)) > 1:
            try:
                results["roc_auc"] = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
            except ValueError:
                results["roc_auc"] = None
            try:
                results["pr_auc"] = round(float(average_precision_score(y_true_arr, y_prob_arr)), 4)
            except ValueError:
                results["pr_auc"] = None
        else:
            results["roc_auc"] = None
            results["pr_auc"] = None

        # Calibration metrics
        results["calibration"] = {
            "brier_score": round(float(brier_score_loss(y_true_arr, y_prob_arr)), 4),
            "log_loss": round(float(log_loss(y_true_arr, y_prob_clipped)), 4),
            "mean_predicted_prob": round(float(np.mean(y_prob_arr)), 4),
            "actual_positive_rate": round(float(np.mean(y_true_arr)), 4),
            "calibration_gap": round(float(abs(np.mean(y_prob_arr) - np.mean(y_true_arr))), 4),
        }
    else:
        results["roc_auc"] = None
        results["pr_auc"] = None
        results["calibration"] = None

    # 3. Unit Economics & Business Financial Metrics
    biz = calculate_business_metrics(y_true_arr, y_pred_arr, y_prob)
    results["business_metrics"] = biz

    return results


def format_evaluation_summary(metrics_dict: Dict[str, Any], model_name: str) -> str:
    """Formats model evaluation metrics as a clean, readable text table."""
    cm = metrics_dict["confusion_matrix"]
    biz = metrics_dict["business_metrics"]
    cal = metrics_dict.get("calibration")

    lines = [
        f"============================================================",
        f" Model: {model_name} (Threshold: {metrics_dict.get('threshold')})",
        f"------------------------------------------------------------",
        f" Statistical Metrics:",
        f"   Accuracy:            {metrics_dict['accuracy'] * 100:.2f}%",
        f"   PR-AUC:              {metrics_dict.get('pr_auc')}",
        f"   ROC-AUC:             {metrics_dict.get('roc_auc')}",
        f"   Precision (Return):  {metrics_dict['precision'] * 100:.2f}%",
        f"   Recall (Return):     {metrics_dict['recall'] * 100:.2f}%",
        f"   F1-Score:            {metrics_dict['f1']:.4f}",
        f" Confusion Matrix:",
        f"   [TN: {cm['tn']:<5}  FP: {cm['fp']:<5}]",
        f"   [FN: {cm['fn']:<5}  TP: {cm['tp']:<5}]",
    ]

    if cal:
        lines.extend([
            f" Calibration:",
            f"   Brier Score:         {cal['brier_score']:.4f}",
            f"   Log Loss:            {cal['log_loss']:.4f}",
            f"   Mean Pred vs Actual: {cal['mean_predicted_prob'] * 100:.2f}% vs {cal['actual_positive_rate'] * 100:.2f}%",
        ])

    lines.extend([
        f" Business & Financial Impact:",
        f"   Calls Triggered:     {biz['calls_triggered']} ({biz['call_rate_pct']:.1f}% of volume)",
        f"   Returns Prevented:   {biz['returns_prevented']:.1f} out of {biz['actual_returns']} actual",
        f"   Call Cost Incurred:  INR {biz['total_call_cost_inr']:,.2f}",
        f"   Gross Savings:       INR {biz['gross_prevented_savings_inr']:,.2f}",
        f"   Net Savings:         INR {biz['net_financial_savings_inr']:,.2f}",
        f"   Net Savings/Order:   INR {biz['net_savings_per_order_inr']:.2f}",
        f"   Cost Reduction:      {biz['cost_reduction_pct']:.2f}%",
        f"============================================================",
    ])

    return "\n".join(lines)
