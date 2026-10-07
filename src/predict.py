"""Prediction interface module for Kestrel Home Returns Risk scoring.

Provides:
- predict_single_order: Evaluates a single dispatch order payload and returns risk score, risk level, decision, and reasons.
- predict_batch_orders: Evaluates a DataFrame of orders and returns scored DataFrame.
- generate_submission_predictions: Generates predictions.csv matching sample_submission.csv format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SAMPLE_SUBMISSION_PATH,
    TEST_RAW_PATH,
)
from src.data import load_raw_data
from src.explain import ReturnsExplainer
from src.features import predict_order_features
from src.predict_submission import (
    generate_final_predictions,
    validate_predictions_format,
)

logger = logging.getLogger("kestrel.predict")

DEFAULT_MODEL_PATH = MODELS_DIR / "selected_model.joblib"
DEFAULT_PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
DEFAULT_THRESHOLD = 0.1500


class ReturnRiskPredictor:
    """Production scoring interface for Kestrel Returns Risk model."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        pipeline_path: Optional[Path] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.pipeline_path = pipeline_path or DEFAULT_PIPELINE_PATH
        self.threshold = threshold

        logger.info("Loading model from: %s", self.model_path)
        self.model = joblib.load(self.model_path)

        logger.info("Loading pipeline from: %s", self.pipeline_path)
        self.pipeline = joblib.load(self.pipeline_path)

        logger.info("Loading reference customer and product catalogs...")
        _, _, self.customers_df, self.products_df = load_raw_data()

        self.explainer = ReturnsExplainer(
            model=self.model,
            pipeline=self.pipeline,
            operating_threshold=self.threshold,
        )

    def predict_single(self, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Scores a single order dictionary and returns risk score, decision, and reasons."""
        order_id = str(order_dict.get("order_id", "UNKNOWN"))

        # Extract features
        feats_df = predict_order_features(
            order_payload=order_dict,
            pipeline=self.pipeline,
            customers_df=self.customers_df,
            products_df=self.products_df,
        )

        # Generate score and explanation
        exp_res = self.explainer.explain_order(
            order_features_df=feats_df,
            raw_order_context=pd.Series(order_dict),
        )

        raw_score = float(exp_res["return_risk_score"])
        rounded_risk = round(raw_score, 2)

        if raw_score >= self.threshold:
            decision = "PRE_DISPATCH_REVIEW"
            risk_level = "HIGH"
        elif raw_score >= 0.1000:
            decision = "DISPATCH"
            risk_level = "MEDIUM"
        else:
            decision = "DISPATCH"
            risk_level = "LOW"

        return {
            "order_id": order_id,
            "score": rounded_risk,
            "return_risk": rounded_risk,
            "raw_risk_score": round(raw_score, 4),
            "risk_level": risk_level,
            "decision": decision,
            "reasons": exp_res["reasons"],
        }


def predict_batch_orders(
    test_df: pd.DataFrame,
    model_path: Optional[Path] = None,
    pipeline_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Scores a batch of orders and returns DataFrame with scores."""
    model = joblib.load(model_path or DEFAULT_MODEL_PATH)
    pipeline = joblib.load(pipeline_path or DEFAULT_PIPELINE_PATH)

    features = pipeline.transform(test_df)
    feature_cols = [c for c in features.columns if c not in ["order_id", "returned"]]
    X = features[feature_cols]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    return pd.DataFrame({
        "order_id": features["order_id"],
        "score": probs,
    })


if __name__ == "__main__":
    generate_final_predictions()

