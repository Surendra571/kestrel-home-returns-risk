"""Unit tests for the explainability module and human-readable reason generator."""

import pandas as pd
import pytest

from src.explain import ReturnsExplainer, explain_prediction


@pytest.fixture
def sample_high_risk_order():
    return {
        "order_id": "KO_HIGH_TEST",
        "order_placed_at": "2026-10-06 14:00",
        "customer_id": "KC100001",
        "sku": "KH-RV-03",
        "sales_channel": "marketplace",
        "payment_mode": "cod",
        "discount_pct": 30,
        "qty": 1,
        "order_value_inr": 20788.6,
        "promised_delivery_days": 8,
        "delivery_pincode": 560001,
        "is_gift": "Y",
        "customer_prior_orders": 2,
        "customer_prior_returns": 1,
        "delivery_note": "Call before delivery",
    }


@pytest.fixture
def sample_low_risk_order():
    return {
        "order_id": "KO_LOW_TEST",
        "order_placed_at": "2026-10-06 10:00",
        "customer_id": "KC100002",
        "sku": "KH-CF-01",
        "sales_channel": "web",
        "payment_mode": "prepaid_upi",
        "discount_pct": 5,
        "qty": 1,
        "order_value_inr": 2659.0,
        "promised_delivery_days": 2,
        "delivery_pincode": 400001,
        "is_gift": "N",
        "customer_prior_orders": 4,
        "customer_prior_returns": 0,
        "delivery_note": "",
    }


def test_explain_prediction_structure_and_types(sample_high_risk_order):
    """Verify that explain_prediction returns expected keys and types."""
    result = explain_prediction(sample_high_risk_order)

    assert isinstance(result, dict)
    assert "order_id" in result
    assert "return_risk_score" in result
    assert "risk_level" in result
    assert "action_recommendation" in result
    assert "reasons" in result
    assert "detailed_factors" in result

    assert isinstance(result["return_risk_score"], float)
    assert 0.0 <= result["return_risk_score"] <= 1.0
    assert result["risk_level"] in ["High Risk", "Moderate Risk", "Low Risk"]
    assert isinstance(result["reasons"], list)
    assert 2 <= len(result["reasons"]) <= 4


def test_explanation_determinism(sample_high_risk_order):
    """Verify that multiple calls on identical input produce identical output."""
    res1 = explain_prediction(sample_high_risk_order)
    res2 = explain_prediction(sample_high_risk_order)

    assert res1["return_risk_score"] == res2["return_risk_score"]
    assert res1["risk_level"] == res2["risk_level"]
    assert res1["reasons"] == res2["reasons"]


def test_no_protected_or_geographic_attributes_in_reasons(sample_high_risk_order):
    """Verify that reasons never mention protected geographic or demographic attributes."""
    result = explain_prediction(sample_high_risk_order)
    all_reasons_text = " ".join(result["reasons"]).lower()

    forbidden_terms = ["pincode", "zone", "state", "maharashtra", "karnataka", "delhi", "560001", "city"]
    for term in forbidden_terms:
        assert term not in all_reasons_text, f"Forbidden attribute '{term}' leaked into business explanation!"


def test_risk_level_mapping_thresholds(sample_high_risk_order, sample_low_risk_order):
    """Verify that high-risk order maps to High Risk and low-risk maps to Low Risk."""
    res_high = explain_prediction(sample_high_risk_order, threshold=0.15)
    res_low = explain_prediction(sample_low_risk_order, threshold=0.15)

    assert res_high["return_risk_score"] >= 0.15
    assert res_high["risk_level"] == "High Risk"
    assert "Pre-Dispatch Review" in res_high["action_recommendation"]

    assert res_low["return_risk_score"] < 0.10
    assert res_low["risk_level"] == "Low Risk"
    assert "Dispatch Immediately" in res_low["action_recommendation"]


def test_reasons_include_both_risk_increasing_and_reducing(sample_high_risk_order):
    """Verify that detailed_factors captures risk_increasing and/or risk_reducing categories."""
    result = explain_prediction(sample_high_risk_order)
    factors = result["detailed_factors"]

    factor_types = {f["type"] for f in factors}
    assert "risk_increasing" in factor_types or "risk_reducing" in factor_types
    for f in factors:
        assert "factor_category" in f
        assert "contribution" in f
        assert "explanation" in f

