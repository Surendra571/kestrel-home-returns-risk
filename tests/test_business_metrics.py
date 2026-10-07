"""Unit tests for business metrics calculations and economic thresholds."""

import numpy as np
import pytest

from src.business_metrics import (
    COST_OF_CALL_INR,
    COST_OF_RETURN_INR,
    OPTIMAL_ECONOMIC_THRESHOLD,
    PREVENTION_EFFECTIVENESS,
    calculate_business_metrics,
    find_cost_optimal_threshold,
)


def test_calculate_business_metrics_zero_calls():
    """Verify metrics when no calls are triggered (majority class do-nothing baseline)."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0, 0])

    biz = calculate_business_metrics(y_true, y_pred)
    assert biz["total_orders"] == 5
    assert biz["actual_returns"] == 2
    assert biz["calls_triggered"] == 0
    assert biz["true_positives"] == 0
    assert biz["returns_prevented"] == 0.0
    assert biz["total_call_cost_inr"] == 0.0
    assert biz["net_financial_savings_inr"] == 0.0
    assert biz["gross_baseline_cost_inr"] == 2 * COST_OF_RETURN_INR


def test_calculate_business_metrics_perfect_predictions():
    """Verify metrics when all positive returns are perfectly identified and called."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])

    biz = calculate_business_metrics(y_true, y_pred)
    assert biz["true_positives"] == 2
    assert biz["false_positives"] == 0
    assert biz["calls_triggered"] == 2

    # Prevented returns = 2 * 0.35 = 0.70
    expected_returns_prevented = 2 * PREVENTION_EFFECTIVENESS
    assert biz["returns_prevented"] == expected_returns_prevented

    # Gross savings = 0.70 * 1150 = 805.0
    expected_gross_savings = expected_returns_prevented * COST_OF_RETURN_INR
    assert biz["gross_prevented_savings_inr"] == expected_gross_savings

    # Total call cost = 2 * 45 = 90.0
    expected_call_cost = 2 * COST_OF_CALL_INR
    assert biz["total_call_cost_inr"] == expected_call_cost

    # Net savings = 805 - 90 = 715.0
    assert biz["net_financial_savings_inr"] == expected_gross_savings - expected_call_cost


def test_calculate_business_metrics_all_false_positives():
    """Verify metrics when calls are made on negative class (wasted calls)."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])

    biz = calculate_business_metrics(y_true, y_pred)
    assert biz["true_positives"] == 0
    assert biz["false_positives"] == 3
    assert biz["returns_prevented"] == 0.0
    assert biz["total_call_cost_inr"] == 3 * COST_OF_CALL_INR
    assert biz["net_financial_savings_inr"] == -3 * COST_OF_CALL_INR


def test_find_cost_optimal_threshold():
    """Verify threshold optimization finds positive net savings on separable data."""
    y_true = np.array([0] * 80 + [1] * 20)
    # High probability for positives, low for negatives
    y_prob = np.array([0.05] * 80 + [0.35] * 20)

    opt = find_cost_optimal_threshold(y_true, y_prob)
    assert opt["max_net_savings_inr"] > 0
    assert 0.05 < opt["optimal_threshold"] <= 0.35


def test_threshold_analysis_file_exists_and_valid():
    """Verify outputs/threshold_analysis.csv exists, has expected columns, and finds positive savings."""
    import pandas as pd
    from src.config import PROJECT_ROOT

    csv_path = PROJECT_ROOT / "outputs" / "threshold_analysis.csv"
    assert csv_path.exists(), "threshold_analysis.csv was not generated!"
    df = pd.read_csv(csv_path)

    expected_cols = [
        "threshold",
        "orders_held",
        "orders_held_pct",
        "true_returns_caught",
        "returns_missed",
        "false_positives",
        "precision",
        "recall",
        "expected_business_cost_inr",
        "expected_avoided_return_cost_inr",
        "expected_net_savings_inr",
        "cost_per_prevented_return_inr",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    assert len(df) > 50
    assert df["expected_net_savings_inr"].max() > 30000.0


def test_break_even_probability_calculation():
    """Verify break-even confirmation call probability calculation: 45 / (1150 * 0.35) ~ 0.1118."""
    calculated_p = COST_OF_CALL_INR / (COST_OF_RETURN_INR * PREVENTION_EFFECTIVENESS)
    assert calculated_p == pytest.approx(0.1118, abs=1e-4)


def test_call_capacity_constraint_at_threshold():
    """Verify that at operational threshold t=0.1500, the call volume remains within the 25% capacity limit."""
    import pandas as pd
    from src.config import PROJECT_ROOT

    csv_path = PROJECT_ROOT / "outputs" / "threshold_analysis.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)

    # Locate row for threshold 0.1500
    row_15 = df.loc[df["threshold"].round(4) == 0.1500]
    assert len(row_15) == 1
    call_rate_pct = float(row_15["orders_held_pct"].iloc[0])
    assert call_rate_pct <= 25.0, f"Call rate {call_rate_pct}% exceeds 25% capacity limit!"
    assert call_rate_pct == pytest.approx(22.15, abs=0.1)


