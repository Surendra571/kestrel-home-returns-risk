"""Business and financial metrics computation for Kestrel Home Returns Risk.

Translates machine learning classification outcomes into direct financial impact
based on Operations Policy v4.1 (§4, §6, §7).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

# Operational parameters from Operations Policy v4.1
COST_OF_RETURN_INR = 1150.0            # Average reverse transit, QC, repacking, depreciation
COST_OF_CALL_INR = 45.0                # Outbound pre-dispatch verification call cost
PREVENTION_EFFECTIVENESS = 0.35        # Proportion of potential returns prevented by call
OPTIMAL_ECONOMIC_THRESHOLD = 0.1118    # 45 / (0.35 * 1150) = 11.18%


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    call_cost: float = COST_OF_CALL_INR,
    return_cost: float = COST_OF_RETURN_INR,
    prevention_rate: float = PREVENTION_EFFECTIVENESS,
) -> Dict[str, Any]:
    """Calculates comprehensive business unit economics and cost savings.

    Financial logic:
    - Do-Nothing Gross Cost: actual_returns * return_cost
    - True Positive (TP): Customer would have returned; call prevents 35% of returns.
      Savings per TP = (0.35 * return_cost) - call_cost = ₹402.50 - ₹45.00 = ₹357.50
    - False Positive (FP): Customer would not have returned; unnecessary call.
      Loss per FP = call_cost = ₹45.00
    - False Negative (FN): Missed return; zero savings, full return cost incurred.
    - True Negative (TN): Correctly dispatched with zero friction; ₹0 cost.

    Args:
        y_true: Ground truth binary outcomes (1 = returned, 0 = retained).
        y_pred: Binary decisions (1 = initiate call / intervention, 0 = dispatch).
        y_prob: Optional predicted probability scores.
        call_cost: Cost per intervention call in INR.
        return_cost: Cost per returned order in INR.
        prevention_rate: Operational return prevention efficiency.

    Returns:
        Dictionary of business financial and operational metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    total_orders = len(y_true)
    actual_returns = int(np.sum(y_true))
    actual_non_returns = total_orders - actual_returns
    base_return_rate = actual_returns / max(total_orders, 1)

    # Operational triage counts
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    calls_triggered = tp + fp
    call_rate_pct = (calls_triggered / total_orders * 100.0) if total_orders > 0 else 0.0

    # Prevented returns and financial figures
    returns_prevented = tp * prevention_rate
    gross_baseline_cost = actual_returns * return_cost
    total_call_cost = calls_triggered * call_cost
    gross_savings = returns_prevented * return_cost
    net_financial_savings = gross_savings - total_call_cost

    final_operational_cost = gross_baseline_cost - net_financial_savings
    cost_reduction_pct = (
        (net_financial_savings / gross_baseline_cost * 100.0) if gross_baseline_cost > 0 else 0.0
    )
    net_savings_per_order = net_financial_savings / max(total_orders, 1)

    # Cost per prevented return
    cost_per_prevented_return = (
        (total_call_cost / returns_prevented) if returns_prevented > 0 else float("inf")
    )

    return {
        "total_orders": total_orders,
        "actual_returns": actual_returns,
        "actual_non_returns": actual_non_returns,
        "base_return_rate": round(base_return_rate, 4),
        "calls_triggered": calls_triggered,
        "call_rate_pct": round(call_rate_pct, 2),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "returns_prevented": round(returns_prevented, 2),
        "gross_baseline_cost_inr": round(gross_baseline_cost, 2),
        "total_call_cost_inr": round(total_call_cost, 2),
        "gross_prevented_savings_inr": round(gross_savings, 2),
        "net_financial_savings_inr": round(net_financial_savings, 2),
        "net_savings_per_order_inr": round(net_savings_per_order, 2),
        "final_operational_cost_inr": round(final_operational_cost, 2),
        "cost_reduction_pct": round(cost_reduction_pct, 2),
        "cost_per_prevented_return_inr": round(cost_per_prevented_return, 2),
    }


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    call_cost: float = COST_OF_CALL_INR,
    return_cost: float = COST_OF_RETURN_INR,
    prevention_rate: float = PREVENTION_EFFECTIVENESS,
) -> Dict[str, Any]:
    """Finds the empirical probability threshold maximizing net financial savings."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_savings = -float("inf")
    best_metrics: Dict[str, Any] = {}

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        m = calculate_business_metrics(
            y_true, pred, y_prob, call_cost, return_cost, prevention_rate
        )
        if m["net_financial_savings_inr"] > best_savings:
            best_savings = m["net_financial_savings_inr"]
            best_threshold = float(t)
            best_metrics = m

    return {
        "optimal_threshold": round(best_threshold, 4),
        "max_net_savings_inr": round(best_savings, 2),
        "metrics_at_optimal": best_metrics,
    }

