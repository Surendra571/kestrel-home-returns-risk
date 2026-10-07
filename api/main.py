"""Production-quality FastAPI service for Kestrel Home Returns Risk scoring.

Provides:
- POST /predict: Scores a single dispatch order, returning risk score, risk level, decision, and business reasons.
- GET /health: Health check returning service status, model version, and active threshold.
- GET /docs & /redoc: Interactive OpenAPI documentation.

Features:
- 100% offline (no external APIs or paid keys required).
- Model and preprocessing pipeline loaded once at startup via lifespan context.
- Pydantic v2 request validation with explicit domain error messages.
- Environment-based configuration with sensible defaults.
- Safe CORS configuration.
- Structured JSON-compatible logging.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

from src.config import PROJECT_ROOT
from src.data import load_raw_data
from src.explain import ReturnsExplainer
from src.features import predict_order_features

# -----------------------------------------------------------------------------
# Structured Logging Setup
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("KESTREL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("kestrel.api")

# -----------------------------------------------------------------------------
# Environment & Service Settings
# -----------------------------------------------------------------------------
DEFAULT_THRESHOLD = float(os.getenv("KESTREL_RISK_THRESHOLD", "0.1500"))
MODEL_PATH = Path(os.getenv("KESTREL_MODEL_PATH", str(PROJECT_ROOT / "models" / "selected_model.joblib")))
PIPELINE_PATH = Path(
    os.getenv("KESTREL_PIPELINE_PATH", str(PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"))
)
MODEL_VERSION = os.getenv("KESTREL_MODEL_VERSION", "1.0.0")
UI_DIR = Path(os.getenv("KESTREL_UI_DIR", str(PROJECT_ROOT / "ui")))
CORS_ORIGINS_RAW = os.getenv("KESTREL_CORS_ORIGINS", "*")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_RAW.split(",")] if "," in CORS_ORIGINS_RAW else [CORS_ORIGINS_RAW]


# -----------------------------------------------------------------------------
# Lifespan Context: Load Models Once At Startup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads model, preprocessing pipeline, and reference lookup tables once at startup."""
    logger.info("Initializing Kestrel Returns Risk API service...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Selected model artifact not found at: {MODEL_PATH}")
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Preprocessing pipeline artifact not found at: {PIPELINE_PATH}")

    logger.info("Loading model artifact from: %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    logger.info("Loading feature pipeline artifact from: %s", PIPELINE_PATH)
    pipeline = joblib.load(PIPELINE_PATH)

    logger.info("Pre-loading reference customer and product tables for join lookups...")
    _, _, customers_df, products_df = load_raw_data()

    logger.info("Initializing offline explainability engine with operating threshold = %.4f...", DEFAULT_THRESHOLD)
    explainer = ReturnsExplainer(model=model, pipeline=pipeline, operating_threshold=DEFAULT_THRESHOLD)

    # Store state on app instance
    app.state.model = model
    app.state.pipeline = pipeline
    app.state.customers_df = customers_df
    app.state.products_df = products_df
    app.state.explainer = explainer
    app.state.threshold = DEFAULT_THRESHOLD
    app.state.model_version = MODEL_VERSION

    logger.info("Kestrel Returns Risk API service initialized successfully.")
    yield
    logger.info("Shutting down Kestrel Returns Risk API service...")


# -----------------------------------------------------------------------------
# FastAPI App Instance
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Kestrel Home Appliances — Returns Risk Prediction API",
    description=(
        "Production-quality, lightweight inference service scoring order return risk at the warehouse "
        "dispatch stage. Implements pre-shipment feature extraction, threshold-optimized operational decisioning, "
        "and human-readable business explanations."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class OrderRequest(BaseModel):
    """Input payload representing an individual order at warehouse packaging."""

    order_id: str = Field(..., min_length=1, description="Unique order identifier (e.g. KO2610504)", examples=["KO2610504"])
    order_placed_at: str = Field(..., min_length=10, description="Order timestamp in IST (e.g. '2026-10-06 14:00')", examples=["2026-10-06 14:00"])
    customer_id: str = Field(..., min_length=1, description="Customer unique ID (e.g. KC100001)", examples=["KC100001"])
    sku: str = Field(..., min_length=1, description="Product catalog SKU (e.g. KH-RV-03)", examples=["KH-RV-03"])
    sales_channel: str = Field(..., description="Channel: app, web, marketplace, partner_outlet", examples=["marketplace"])
    payment_mode: str = Field(..., description="Payment mode: cod, prepaid_upi, prepaid_card, emi", examples=["cod"])
    discount_pct: float = Field(0.0, ge=0.0, le=100.0, description="Discount percentage (0-100)", examples=[25.0])
    qty: int = Field(1, ge=1, le=10, description="Item quantity ordered (1 to 10)", examples=[1])
    order_value_inr: float = Field(..., gt=0.0, description="Recorded transaction amount in INR", examples=[22273.5])
    promised_delivery_days: int = Field(5, ge=1, le=30, description="Promised transit SLA in days", examples=[8])
    delivery_pincode: int = Field(..., ge=0, le=999999, description="6-digit Indian PIN code (000000 for walk-ins)", examples=[560001])
    is_gift: str = Field("N", description="Gift indicator ('Y' or 'N')", examples=["N"])
    customer_prior_orders: int = Field(0, ge=0, description="Historical customer orders before this one", examples=[2])
    customer_prior_returns: int = Field(0, ge=0, description="Historical customer returns before this one", examples=[1])
    delivery_note: Optional[str] = Field("", description="Optional customer checkout instructions", examples=["Call before delivery"])

    @field_validator("is_gift")
    @classmethod
    def validate_is_gift(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in ["Y", "N"]:
            raise ValueError("is_gift must be either 'Y' or 'N'")
        return v_upper

    @field_validator("customer_prior_returns")
    @classmethod
    def validate_prior_returns(cls, v: int, info) -> int:
        prior_orders = info.data.get("customer_prior_orders")
        if prior_orders is not None and v > prior_orders:
            raise ValueError(
                f"customer_prior_returns ({v}) cannot exceed customer_prior_orders ({prior_orders})"
            )
        return v


class PredictResponse(BaseModel):
    """Structured response output for return risk scoring and dispatch triage."""

    order_id: str = Field(..., description="Original order identifier")
    return_risk: float = Field(..., description="Calibrated return probability score between 0.0 and 1.0")
    score: float = Field(..., description="Calibrated return probability score between 0.0 and 1.0")
    risk_level: str = Field(..., description="Operational risk tier: HIGH, MEDIUM, or LOW")
    decision: str = Field(..., description="Dispatch decision: PRE_DISPATCH_REVIEW or DISPATCH")
    reasons: List[str] = Field(..., description="Top 2 to 4 plain-English business reasons for the decision")


class HealthResponse(BaseModel):
    """System health check and configuration metadata."""

    status: str = Field("ok", description="Service health status")
    model_version: str = Field(..., description="Active model deployment version")
    threshold: float = Field(..., description="Active operational decision threshold")


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def serve_dashboard():
    """Serves the single-screen Kestrel operations web portal UI."""
    index_file = UI_DIR / "index.html"
    if not index_file.exists():
        index_file = PROJECT_ROOT / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return HTMLResponse(
        "<h1>Kestrel UI not found</h1><p>Please ensure ui/index.html or frontend/index.html exists in the repository root.</p>",
        status_code=404,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    tags=["System"],
)
async def health_check():
    """Returns the operational status, active model version, and decision threshold."""
    return HealthResponse(
        status="ok",
        model_version=app.state.model_version,
        threshold=app.state.threshold,
    )


@app.get(
    "/predict",
    summary="Predict Endpoint Guidance & UI Access",
    tags=["Inference"],
)
async def get_predict_info(request: Request):
    """Provides user-friendly guidance or serves the operations UI when /predict is accessed via GET."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return await serve_dashboard()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "The /predict endpoint requires an HTTP POST request with an order JSON payload.",
            "operations_ui": "/",
            "api_documentation": "/docs#/Inference/predict_order_return_risk_predict_post",
            "example_post_curl": "curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{...}'"
        }
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Score Order Return Risk",
    tags=["Inference"],
)
async def predict_order_return_risk(order: OrderRequest):
    """Scores an order at warehouse dispatch and outputs calibrated risk, decision, and reasons.

    Decision Rule:
    - If return_risk >= threshold: decision = "PRE_DISPATCH_REVIEW" (Initiate pre-dispatch verification call within review window).
    - If return_risk < threshold: decision = "DISPATCH" (Clear for immediate outbound shipping).
    """
    t0 = time.perf_counter()
    order_dict = order.model_dump()
    order_id = order.order_id

    try:
        explainer: ReturnsExplainer = app.state.explainer

        # Transform raw payload using pipeline and lookup tables
        features_df = predict_order_features(
            order_payload=order_dict,
            pipeline=app.state.pipeline,
            customers_df=app.state.customers_df,
            products_df=app.state.products_df,
        )

        # Generate deterministic explanation and risk score
        explanation_res = explainer.explain_order(
            order_features_df=features_df,
            raw_order_context=pd.Series(order_dict),
        )

        raw_score = explanation_res["return_risk_score"]
        # Round to 2 decimal places as specified in format
        rounded_risk = round(raw_score, 2)

        # Map to required schema fields
        active_threshold = app.state.threshold
        if raw_score >= active_threshold:
            decision = os.getenv("KESTREL_HIGH_RISK_DECISION", "PRE_DISPATCH_REVIEW")
            risk_level_str = "HIGH"
        elif raw_score >= 0.1000:
            decision = "DISPATCH"
            risk_level_str = "MEDIUM"
        else:
            decision = "DISPATCH"
            risk_level_str = "LOW"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Scored order_id=%s: return_risk=%.4f (rounded=%.2f), risk_level=%s, decision=%s, latency=%.2fms",
            order_id,
            raw_score,
            rounded_risk,
            risk_level_str,
            decision,
            latency_ms,
        )

        return PredictResponse(
            order_id=order_id,
            return_risk=rounded_risk,
            score=rounded_risk,
            risk_level=risk_level_str,
            decision=decision,
            reasons=explanation_res["reasons"],
        )

    except Exception as exc:
        logger.error("Inference failure for order_id=%s: %s", order_id, str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure processing order {order_id}: {str(exc)}",
        )
