"""Integration and API unit tests for the Kestrel Returns Risk FastAPI service."""

from fastapi.testclient import TestClient
import pytest

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient with triggered FastAPI lifespan startup/shutdown."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_high_risk_payload():
    return {
        "order_id": "KO2610504",
        "order_placed_at": "2026-10-06 14:00",
        "customer_id": "KC100001",
        "sku": "KH-RV-03",
        "sales_channel": "marketplace",
        "payment_mode": "cod",
        "discount_pct": 30.0,
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
def valid_low_risk_payload():
    return {
        "order_id": "KO2610505",
        "order_placed_at": "2026-10-06 10:00",
        "customer_id": "KC100002",
        "sku": "KH-CF-01",
        "sales_channel": "web",
        "payment_mode": "prepaid_upi",
        "discount_pct": 5.0,
        "qty": 1,
        "order_value_inr": 2659.0,
        "promised_delivery_days": 2,
        "delivery_pincode": 400001,
        "is_gift": "N",
        "customer_prior_orders": 4,
        "customer_prior_returns": 0,
        "delivery_note": "",
    }


def test_get_health_endpoint(client):
    """Verify /health returns status ok, model version, and active threshold."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert data["threshold"] == pytest.approx(0.1500, rel=1e-3)


def test_predict_high_risk_order(client, valid_high_risk_payload):
    """Verify high-risk order returns HOLD_FOR_REVIEW, HIGH risk level, and reasons."""
    response = client.post("/predict", json=valid_high_risk_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "KO2610504"
    assert isinstance(data["return_risk"], float)
    assert data["return_risk"] >= 0.15
    assert data["risk_level"] == "HIGH"
    assert data["decision"] == "PRE_DISPATCH_REVIEW"
    assert "score" in data
    assert isinstance(data["reasons"], list)
    assert 2 <= len(data["reasons"]) <= 4

    # Ensure no technical internal names leak
    for reason in data["reasons"]:
        assert "smoothed_prior_return_rate" not in reason
        assert "is_cod" not in reason
        assert "pincode" not in reason.lower()


def test_predict_low_risk_order(client, valid_low_risk_payload):
    """Verify low-risk order returns DISPATCH, LOW/MEDIUM risk level, and reasons."""
    response = client.post("/predict", json=valid_low_risk_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "KO2610505"
    assert isinstance(data["return_risk"], float)
    assert data["return_risk"] < 0.15
    assert data["decision"] == "DISPATCH"
    assert data["risk_level"] in ["LOW", "MEDIUM"]
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) >= 1


def test_predict_validation_error_prior_returns_greater_than_orders(client, valid_high_risk_payload):
    """Verify Pydantic validation catches impossible customer history (returns > orders)."""
    invalid_payload = dict(valid_high_risk_payload)
    invalid_payload["customer_prior_orders"] = 1
    invalid_payload["customer_prior_returns"] = 3  # Impossible

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    error_msg = str(data["detail"])
    assert "cannot exceed customer_prior_orders" in error_msg


def test_predict_validation_error_missing_required_fields(client):
    """Verify Pydantic validation rejects incomplete request bodies."""
    response = client.post("/predict", json={"order_id": "TEST_ONLY"})
    assert response.status_code == 422


def test_openapi_documentation_accessible(client):
    """Verify interactive Swagger UI and OpenAPI schema are accessible."""
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200

    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    schema = openapi_resp.json()
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]


def test_get_root_serves_web_ui(client):
    """Verify GET / returns the operations web portal UI with correct title and elements."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html_text = response.text
    assert "Kestrel Home Appliances — Returns Risk Portal" in html_text
    assert "Check Return Risk" in html_text
    assert "PRE-DISPATCH REVIEW" in html_text


def test_predict_unknown_customer_and_product_ids(client, valid_high_risk_payload):
    """Verify service gracefully handles unknown customer or product IDs without failing."""
    payload = dict(valid_high_risk_payload)
    payload["order_id"] = "KO_UNKNOWN_01"
    payload["customer_id"] = "CUST_UNSEEN_9999"
    payload["sku"] = "SKU_UNSEEN_99"

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "KO_UNKNOWN_01"
    assert "score" in data
    assert "return_risk" in data
    assert "decision" in data
    assert "reasons" in data
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) >= 1

