"""Explainability and human-readable operational reasoning module for Kestrel Home Returns Risk.

Translates mathematical model contributions into plain-English business rationales
for warehouse packaging station operators.
- Fully offline and deterministic (no LLMs or paid APIs required).
- Grounded strictly in validated input features and model weights.
- Completely excludes protected or irrelevant attributes (geographic biases, etc.).
- Returns top 2–4 concise reasons covering risk-increasing and risk-reducing factors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.config import ID_COLUMN, MODELS_DIR, PROJECT_ROOT
from src.features import ReturnsFeaturePipeline, predict_order_features

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "selected_model.joblib"
DEFAULT_PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
OPERATING_THRESHOLD = 0.1500  # Recommended operational threshold from reports/threshold_decision.md


# -----------------------------------------------------------------------------
# Business Factor Definition & Text Templates
# -----------------------------------------------------------------------------
# Protected and irrelevant features strictly excluded from explanations:
EXCLUDED_ATTRIBUTES = {
    "order_id",
    "returned",
    "pincode_zone",
    "is_walkin_pincode",
    "order_dayofweek",
    "order_day",
    "order_month",
    "state_DL",
    "state_KA",
    "state_MH",
    "state_MP",
    "state_RJ",
    "state_TN",
    "state_TS",
    "state_UP",
    "sales_channel_web",
    "sales_channel_app",
}


class ReturnsExplainer:
    """Generates deterministic, non-technical business explanations for return risk scores."""

    def __init__(
        self,
        model: Optional[Any] = None,
        pipeline: Optional[ReturnsFeaturePipeline] = None,
        operating_threshold: float = OPERATING_THRESHOLD,
    ) -> None:
        self.model = model or joblib.load(DEFAULT_MODEL_PATH)
        self.pipeline = pipeline or joblib.load(DEFAULT_PIPELINE_PATH)
        self.threshold = operating_threshold

        # Extract underlying classifier and scaler
        if hasattr(self.model, "named_steps"):
            self.scaler = self.model.named_steps.get("scaler")
            self.clf = self.model.named_steps.get("clf")
        else:
            self.scaler = None
            self.clf = self.model

        self.feature_names = self.pipeline.feature_names_

    def _get_feature_contributions(self, X_features: pd.DataFrame) -> pd.DataFrame:
        """Computes exact linear standardized log-odds contributions (w_i * z_i)."""
        feat_cols = [c for c in X_features.columns if c not in [ID_COLUMN, "returned"]]
        X_df = X_features[feat_cols]

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_df)
        else:
            X_scaled = X_df.values

        coefs = self.clf.coef_[0]
        contributions = X_scaled * coefs

        return pd.DataFrame(contributions, columns=feat_cols, index=X_features.index)

    def _generate_factor_narrative(
        self,
        factor_name: str,
        contribution: float,
        raw_row: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Maps an aggregated business factor contribution to plain business language."""
        abs_contrib = abs(contribution)
        if abs_contrib < 0.04:  # Disregard negligible contributions
            return None

        is_increasing = contribution > 0
        factor_type = "risk_increasing" if is_increasing else "risk_reducing"

        text = None

        if factor_name == "prior_returns":
            prior_ret = int(raw_row.get("customer_prior_returns", 0))
            if is_increasing and prior_ret >= 1:
                text = f"Customer has a history of prior returns ({prior_ret} previous return{'s' if prior_ret > 1 else ''})."
            elif not is_increasing and prior_ret == 0:
                text = "Customer has a clean track record with zero prior returns."

        elif factor_name == "delivery_sla":
            sla_days = int(raw_row.get("promised_delivery_days", 5))
            if is_increasing and sla_days >= 6:
                text = f"Promised delivery timeline is long ({sla_days} days), increasing buyer fatigue."
            elif not is_increasing and sla_days <= 3:
                text = f"Rapid delivery timeline ({sla_days} day{'s' if sla_days > 1 else ''}) significantly reduces cancellation risk."

        elif factor_name == "payment_cod":
            is_cod = int(raw_row.get("is_cod", 0)) == 1 or str(raw_row.get("payment_mode", "")).lower() == "cod"
            if is_increasing and is_cod:
                text = "Payment mode is Cash on Delivery, which carries high doorstep refusal risk."

        elif factor_name == "payment_prepaid":
            pay_mode = str(raw_row.get("payment_mode", "")).lower()
            if not is_increasing and ("upi" in pay_mode or "card" in pay_mode or "prepaid" in pay_mode):
                text = "Prepaid payment commitment indicates strong purchase intent."

        elif factor_name == "shield_member":
            is_shield = str(raw_row.get("shield_member", "")).upper() == "Y" or int(raw_row.get("is_shield_member", 0)) == 1
            if is_increasing and is_shield:
                text = "Kestrel Shield member with contractual 30-day free returns entitlement."
            elif not is_increasing and not is_shield:
                text = "Non-Shield customer with standard 10-day return policy."

        elif factor_name == "high_order_value":
            val_inr = float(raw_row.get("order_value_inr", 0))
            if is_increasing and val_inr >= 15000:
                text = f"High-ticket appliance purchase (INR {val_inr:,.0f}) subject to stricter buyer scrutiny."

        elif factor_name == "gift_purchase":
            is_gift = str(raw_row.get("is_gift", "")).upper() == "Y" or int(raw_row.get("is_gift_order", 0)) == 1
            if is_increasing and is_gift:
                text = "Order is designated as a gift, which carries higher recipient mismatch risk."

        elif factor_name == "deep_discount":
            disc = float(raw_row.get("discount_pct", 0))
            if is_increasing and disc >= 25:
                text = f"Heavy promotional discount ({disc:.0f}%) suggests possible impulse buying."

        elif factor_name == "complex_product":
            req_inst = int(raw_row.get("requires_installation", 0)) == 1
            fam = str(raw_row.get("family", "appliance"))
            if is_increasing and req_inst:
                text = f"Complex appliance ({fam}) with elevated setup and installation sensitivity."

        elif factor_name == "reliable_appliance":
            fam = str(raw_row.get("family", "")).lower()
            if not is_increasing and ("fan" in fam or "mixer" in fam):
                text = "Category (Ceiling Fan / Mixer Grinder) exhibits very low historical return rates."

        elif factor_name == "customer_history":
            p_orders = int(raw_row.get("customer_prior_orders", 0))
            if not is_increasing and p_orders >= 2:
                text = f"Established customer loyalty with {p_orders} previous completed orders."

        elif factor_name == "delivery_notes":
            has_note = int(raw_row.get("note_mentions_call", 0)) == 1
            if is_increasing and has_note:
                text = "Customer note explicitly requested calling before delivery, indicating delivery anxiety."

        if text is None:
            return None

        return {
            "factor_category": factor_name,
            "type": factor_type,
            "contribution": round(float(contribution), 4),
            "abs_impact": round(abs_contrib, 4),
            "explanation": text,
        }

    def explain_order(
        self,
        order_features_df: pd.DataFrame,
        raw_order_context: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Generates risk score, triage level, and top 2-4 plain-English reasons for a single order."""
        feat_cols = [c for c in order_features_df.columns if c not in [ID_COLUMN, "returned"]]
        X_vec = order_features_df[feat_cols]

        prob = float(self.model.predict_proba(X_vec)[0, 1])

        # Assign operational risk tier and recommendation
        if prob >= self.threshold:
            risk_level = "High Risk"
            recommendation = "Pre-Dispatch Review: Perform Confirmation Call Before Dispatch"
        elif prob >= 0.1000:
            risk_level = "Moderate Risk"
            recommendation = "Standard Dispatch with High-Priority Tracking"
        else:
            risk_level = "Low Risk"
            recommendation = "Standard Dispatch Immediately (No Hold Needed)"

        # Calculate exact linear contributions
        contrib_row = self._get_feature_contributions(order_features_df).iloc[0]

        # Extract raw values for templating
        raw_ctx = raw_order_context if raw_order_context is not None else order_features_df.iloc[0]

        # Aggregate collinear feature groups
        aggregated_factors = {
            "prior_returns": (
                contrib_row.get("customer_prior_returns", 0)
                + contrib_row.get("has_prior_returns", 0)
                + contrib_row.get("has_multiple_prior_returns", 0)
                + contrib_row.get("smoothed_prior_return_rate", 0)
                + contrib_row.get("prior_return_rate", 0)
            ),
            "delivery_sla": (
                contrib_row.get("promised_delivery_days", 0)
                + contrib_row.get("is_long_sla", 0)
                + contrib_row.get("is_short_sla", 0)
                + contrib_row.get("sla_per_thousand_value", 0)
            ),
            "payment_cod": (
                contrib_row.get("payment_mode_cod", 0)
                + contrib_row.get("is_cod", 0)
                + contrib_row.get("is_first_time_cod", 0)
                + contrib_row.get("is_cod_and_high_value", 0)
            ),
            "payment_prepaid": (
                contrib_row.get("payment_mode_prepaid_upi", 0)
                + contrib_row.get("payment_mode_prepaid_card", 0)
                + contrib_row.get("payment_mode_emi", 0)
            ),
            "shield_member": (
                contrib_row.get("is_shield_member", 0)
                + contrib_row.get("is_shield_and_high_value", 0)
            ),
            "high_order_value": (
                contrib_row.get("list_price_inr", 0)
                + contrib_row.get("order_value_inr", 0)
                + contrib_row.get("total_list_price_inr", 0)
                + contrib_row.get("is_high_order_value", 0)
            ),
            "gift_purchase": contrib_row.get("is_gift_order", 0),
            "deep_discount": (
                contrib_row.get("discount_pct", 0)
                + contrib_row.get("discount_amount", 0)
                + contrib_row.get("is_deep_discount", 0)
            ),
            "complex_product": (
                contrib_row.get("requires_installation", 0)
                + contrib_row.get("family_Robot Vacuum", 0)
                + contrib_row.get("family_Water Purifier", 0)
            ),
            "reliable_appliance": (
                contrib_row.get("family_Ceiling Fan", 0)
                + contrib_row.get("family_Mixer Grinder", 0)
                + contrib_row.get("warranty_months", 0)
            ),
            "customer_history": (
                contrib_row.get("customer_prior_orders", 0)
                + contrib_row.get("is_repeat_customer", 0)
                + contrib_row.get("customer_tenure_days", 0)
            ),
            "delivery_notes": (
                contrib_row.get("note_mentions_call", 0)
                + contrib_row.get("note_mentions_friction", 0)
            ),
        }

        # Build candidate narratives
        narratives: List[Dict[str, Any]] = []
        for factor_name, total_contrib in aggregated_factors.items():
            narr = self._generate_factor_narrative(factor_name, total_contrib, raw_ctx)
            if narr is not None:
                narratives.append(narr)

        # Sort by absolute impact magnitude
        narratives.sort(key=lambda x: x["abs_impact"], reverse=True)

        # Select top 2 to 4 reasons
        # If order is High Risk, prioritize top risk-increasing factors, but include a mitigating factor if significant
        selected_reasons: List[str] = []
        detailed_factors: List[Dict[str, Any]] = []

        if prob >= self.threshold:
            # High risk: select top positive factors first
            pos_factors = [n for n in narratives if n["type"] == "risk_increasing"]
            neg_factors = [n for n in narratives if n["type"] == "risk_reducing"]

            top_pos = pos_factors[:3]
            top_neg = neg_factors[:1] if len(neg_factors) > 0 and neg_factors[0]["abs_impact"] >= 0.15 else []

            combined = top_pos + top_neg
            combined = combined[:4]
            if len(combined) < 2 and len(narratives) >= 2:
                combined = narratives[:2]

            for item in combined:
                selected_reasons.append(item["explanation"])
                detailed_factors.append(item)
        else:
            # Low/Moderate risk: select top reasons by impact
            top_items = narratives[:3]
            if len(top_items) < 2 and len(narratives) >= 2:
                top_items = narratives[:2]

            for item in top_items:
                selected_reasons.append(item["explanation"])
                detailed_factors.append(item)

        # Fallback if no specific factors passed threshold
        if len(selected_reasons) == 0:
            if prob >= self.threshold:
                selected_reasons.append("Overall transaction risk indicators exceed the pre-dispatch threshold.")
            else:
                selected_reasons.append("Standard low-risk profile across transaction, customer, and catalog dimensions.")

        order_id_val = order_features_df[ID_COLUMN].iloc[0] if ID_COLUMN in order_features_df.columns else "N/A"

        return {
            "order_id": str(order_id_val),
            "return_risk_score": round(prob, 4),
            "risk_level": risk_level,
            "action_recommendation": recommendation,
            "reasons": selected_reasons,
            "detailed_factors": detailed_factors,
        }


def explain_prediction(
    order_payload: Union[Dict[str, Any], pd.DataFrame],
    model: Optional[Any] = None,
    pipeline: Optional[ReturnsFeaturePipeline] = None,
    threshold: float = OPERATING_THRESHOLD,
    customers_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Prediction and explainability endpoint for a single order payload.

    Args:
        order_payload: Raw order dict or DataFrame at warehouse dispatch.
        model: Fitted classification model (defaults to models/selected_model.joblib).
        pipeline: Fitted ReturnsFeaturePipeline (defaults to models/preprocessing_pipeline.joblib).
        threshold: Decision threshold for risk classification.
        customers_df: Optional customers lookup table.
        products_df: Optional products catalog lookup table.

    Returns:
        Structured prediction dict with risk score, level, action, and human-readable reasons.
    """
    explainer = ReturnsExplainer(model=model, pipeline=pipeline, operating_threshold=threshold)

    # Transform payload to feature vector
    features_df = predict_order_features(
        order_payload=order_payload,
        pipeline=explainer.pipeline,
        customers_df=customers_df,
        products_df=products_df,
    )

    # Capture raw context
    if isinstance(order_payload, dict):
        raw_series = pd.Series(order_payload)
    else:
        raw_series = order_payload.iloc[0]

    return explainer.explain_order(features_df, raw_order_context=raw_series)
