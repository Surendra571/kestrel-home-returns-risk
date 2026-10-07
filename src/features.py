"""Feature engineering pipeline module for Kestrel Home Returns Risk.

Constructs pre-shipment operational features strictly available prior to warehouse
dispatch. Implements a scikit-learn compatible transformer that fits exclusively
on historical training data and transforms test or live API prediction payloads.
Guarantees zero target leakage.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    FEATURE_PIPELINE_PATH,
    ID_COLUMN,
    INSTALLATION_FAMILIES,
    KNOWN_CUSTOMER_STATES,
    KNOWN_PAYMENT_MODES,
    KNOWN_PRODUCT_FAMILIES,
    KNOWN_PRODUCT_TIERS,
    KNOWN_SALES_CHANNELS,
    LAUNCH_DATE_COL,
    ORDER_DATE_COL,
    PROCESSED_DATA_DIR,
    SIGNUP_DATE_COL,
    TARGET_COLUMN,
    TEST_PROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
)
from src.data import load_and_merge_test, load_and_merge_train, merge_order_data

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def extract_product_tier(series: pd.Series) -> pd.Series:
    """Extracts product tier ('Lite', 'Pro', 'Max') from model name or SKU."""
    tier = series.astype(str).str.extract(r"(Lite|Pro|Max)", expand=False)
    # Default fallback if unparseable
    return tier.fillna("Pro")


class ReturnsFeaturePipeline(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for engineering pre-shipment return risk features.

    All feature calculations use only facts known at the packaging station before
    outbound dispatch. Target variables and post-delivery lifecycle events are strictly
    excluded. All statistical state (medians, categorical encodings, Bayes priors)
    is learned during `fit()` on training data only.
    """

    def __init__(
        self,
        include_id: bool = True,
        include_target: bool = True,
        categorical_columns: Optional[List[str]] = None,
    ) -> None:
        self.include_id = include_id
        self.include_target = include_target
        self.categorical_columns = categorical_columns or [
            "sales_channel",
            "payment_mode",
            "family",
            "product_tier",
            "state",
        ]

        # Learned state attributes
        self.encoder_: Optional[OneHotEncoder] = None
        self.feature_names_: List[str] = []
        self.numeric_medians_: Dict[str, float] = {}
        self.prior_alpha_: float = 0.5
        self.prior_beta_: float = 4.0
        self.is_fitted_: bool = False

    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes raw pre-shipment business features prior to encoding."""
        feats = pd.DataFrame(index=df.index)

        # ---------------------------------------------------------
        # 1. Order Transaction & Pricing Features
        # ---------------------------------------------------------
        feats["qty"] = df["qty"].fillna(1).astype(int)
        feats["discount_pct"] = df["discount_pct"].fillna(0).astype(float)
        feats["order_value_inr"] = df["order_value_inr"].fillna(0.0).astype(float)
        feats["list_price_inr"] = df["list_price_inr"].fillna(0.0).astype(float)

        total_list = feats["list_price_inr"] * feats["qty"]
        feats["total_list_price_inr"] = total_list
        feats["effective_unit_price"] = feats["order_value_inr"] / np.maximum(feats["qty"], 1)
        feats["discount_amount"] = np.maximum(0.0, total_list - feats["order_value_inr"])
        feats["discount_ratio"] = feats["discount_amount"] / np.maximum(total_list, 1.0)
        feats["price_ratio"] = feats["order_value_inr"] / np.maximum(total_list, 1.0)

        feats["is_deep_discount"] = (feats["discount_pct"] >= 30.0).astype(int)
        feats["is_high_order_value"] = (feats["order_value_inr"] >= 15000.0).astype(int)

        # ---------------------------------------------------------
        # 2. Fulfillment & SLA Features
        # ---------------------------------------------------------
        feats["promised_delivery_days"] = df["promised_delivery_days"].fillna(5).astype(int)
        feats["is_long_sla"] = (feats["promised_delivery_days"] >= 7).astype(int)
        feats["is_short_sla"] = (feats["promised_delivery_days"] <= 2).astype(int)
        feats["sla_per_thousand_value"] = feats["promised_delivery_days"] / (
            feats["order_value_inr"] / 1000.0 + 1.0
        )

        # ---------------------------------------------------------
        # 3. Channel, Payment, and Gift Flags
        # ---------------------------------------------------------
        feats["is_gift_order"] = (df["is_gift"].astype(str).str.upper() == "Y").astype(int)
        feats["is_cod"] = (df["payment_mode"].astype(str).str.lower() == "cod").astype(int)
        feats["is_marketplace"] = (
            df["sales_channel"].astype(str).str.lower() == "marketplace"
        ).astype(int)

        # ---------------------------------------------------------
        # 4. Customer Behavioral History & Loyalty
        # ---------------------------------------------------------
        p_orders = df["customer_prior_orders"].fillna(0).astype(float)
        p_returns = df["customer_prior_returns"].fillna(0).astype(float)

        feats["customer_prior_orders"] = p_orders
        feats["customer_prior_returns"] = p_returns
        feats["is_repeat_customer"] = (p_orders > 0).astype(int)
        feats["has_prior_returns"] = (p_returns > 0).astype(int)
        feats["has_multiple_prior_returns"] = (p_returns >= 2).astype(int)

        # Raw prior return rate
        feats["prior_return_rate"] = np.where(p_orders > 0, p_returns / np.maximum(p_orders, 1.0), 0.0)

        # Smoothed Empirical Bayes prior return rate using learned training priors
        feats["smoothed_prior_return_rate"] = (p_returns + self.prior_alpha_) / (
            p_orders + self.prior_beta_
        )

        feats["is_shield_member"] = (
            df["shield_member"].astype(str).str.upper() == "Y"
        ).astype(int)

        # Customer tenure in days relative to order placement
        order_dt = pd.to_datetime(df[ORDER_DATE_COL])
        signup_dt = pd.to_datetime(df[SIGNUP_DATE_COL])
        tenure_days = (order_dt - signup_dt).dt.total_seconds() / 86400.0

        feats["is_guest_or_backfilled"] = (tenure_days < 0).astype(int)
        feats["customer_tenure_days"] = np.maximum(0.0, tenure_days.fillna(0.0))

        # ---------------------------------------------------------
        # 5. Product Catalog & Complexity
        # ---------------------------------------------------------
        feats["warranty_months"] = df["warranty_months"].fillna(12).astype(int)

        launch_dt = pd.to_datetime(df[LAUNCH_DATE_COL])
        prod_age_days = (order_dt - launch_dt).dt.total_seconds() / 86400.0
        feats["product_age_days"] = np.maximum(0.0, prod_age_days.fillna(0.0))

        feats["requires_installation"] = (
            df["family"].astype(str).isin(INSTALLATION_FAMILIES)
        ).astype(int)

        # Extract tier: Lite, Pro, Max
        tier_source = df["model_name"] if "model_name" in df.columns else df["sku"]
        feats["product_tier"] = extract_product_tier(tier_source)
        feats["family"] = df["family"].fillna("Air Fryer")

        # ---------------------------------------------------------
        # 6. Delivery Address & Geographic Hierarchy
        # ---------------------------------------------------------
        pincode_numeric = pd.to_numeric(df["delivery_pincode"], errors="coerce").fillna(0).astype(int)
        feats["is_walkin_pincode"] = (pincode_numeric == 0).astype(int)
        feats["pincode_zone"] = np.where(pincode_numeric == 0, 0, pincode_numeric // 100000)
        feats["state"] = df["state"].fillna("MH")

        # ---------------------------------------------------------
        # 7. Delivery Note NLP & Intent Indicators
        # ---------------------------------------------------------
        note_series = df["delivery_note"].fillna("").astype(str)
        note_lower = note_series.str.lower()
        feats["has_delivery_note"] = (note_series.str.strip() != "").astype(int)
        feats["note_char_length"] = note_series.str.len()

        feats["note_mentions_call"] = note_lower.str.contains(
            r"call|contact|phone|ring", regex=True
        ).astype(int)
        feats["note_mentions_no_call"] = note_lower.str.contains(
            r"do not call|don't call|whatsapp", regex=True
        ).astype(int)
        feats["note_mentions_time_slot"] = note_lower.str.contains(
            r"slot|morning|after 6|evening|weekdays|office", regex=True
        ).astype(int)
        feats["note_mentions_friction"] = note_lower.str.contains(
            r"floor|lift|stair|gate|security|fragile", regex=True
        ).astype(int)
        feats["note_mentions_landmark"] = note_lower.str.contains(r"landmark", regex=True).astype(int)

        # ---------------------------------------------------------
        # 8. Temporal & Calendar Dynamics
        # ---------------------------------------------------------
        feats["order_hour"] = order_dt.dt.hour.fillna(12).astype(int)
        feats["order_dayofweek"] = order_dt.dt.dayofweek.fillna(0).astype(int)
        feats["order_day"] = order_dt.dt.day.fillna(15).astype(int)
        feats["order_month"] = order_dt.dt.month.fillna(6).astype(int)
        feats["is_weekend"] = (feats["order_dayofweek"] >= 5).astype(int)
        feats["is_night_order"] = (feats["order_hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])).astype(int)

        # ---------------------------------------------------------
        # 9. High-Risk Compound Interactions
        # ---------------------------------------------------------
        feats["is_cod_and_high_value"] = feats["is_cod"] * feats["is_high_order_value"]
        feats["is_cod_and_long_sla"] = feats["is_cod"] * feats["is_long_sla"]
        feats["is_first_time_cod"] = (feats["customer_prior_orders"] == 0).astype(int) * feats["is_cod"]
        feats["is_shield_and_high_value"] = feats["is_shield_member"] * feats["is_high_order_value"]
        feats["is_complex_no_installation"] = (
            feats["requires_installation"]
            * (df["sales_channel"].astype(str).str.lower() != "partner_outlet").astype(int)
        )

        # Base categoricals preserved for encoding step
        feats["sales_channel"] = df["sales_channel"].fillna("web")
        feats["payment_mode"] = df["payment_mode"].fillna("prepaid_upi")

        return feats

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "ReturnsFeaturePipeline":
        """Fits the feature engineering pipeline exclusively on training data.

        Learns:
        - Prior return rate Bayes parameters from historical orders
        - Numerical imputation medians
        - One-hot encoder categories for categorical dimensions
        """
        logger.info("Fitting ReturnsFeaturePipeline on %d records...", len(X))
        base_df = self._extract_base_features(X)

        # 1. Fit Empirical Bayes smoothing parameters from customer history in train
        prior_orders_sum = base_df["customer_prior_orders"].sum()
        prior_returns_sum = base_df["customer_prior_returns"].sum()
        global_prior_rate = (
            (prior_returns_sum / max(prior_orders_sum, 1.0)) if prior_orders_sum > 0 else 0.114
        )
        smoothing_weight = 4.0
        self.prior_alpha_ = float(global_prior_rate * smoothing_weight)
        self.prior_beta_ = float((1.0 - global_prior_rate) * smoothing_weight)

        # Recompute smoothed prior rate with fitted priors
        base_df["smoothed_prior_return_rate"] = (
            base_df["customer_prior_returns"] + self.prior_alpha_
        ) / (base_df["customer_prior_orders"] + self.prior_beta_)

        # 2. Fit OneHotEncoder on categorical columns
        cat_data = base_df[self.categorical_columns].astype(str)
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder_.fit(cat_data)

        # 3. Compute medians for numerical columns for reliable inference imputation
        num_cols = [c for c in base_df.columns if c not in self.categorical_columns]
        for col in num_cols:
            self.numeric_medians_[col] = float(base_df[col].median())

        # Determine full transformed feature names
        encoded_cat_names = list(self.encoder_.get_feature_names_out(self.categorical_columns))
        self.feature_names_ = num_cols + encoded_cat_names

        self.is_fitted_ = True
        logger.info(
            "Feature pipeline fitted successfully. Features count=%d (Numeric=%d, Categorical Encoded=%d)",
            len(self.feature_names_),
            len(num_cols),
            len(encoded_cat_names),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw merged order data into clean, model-ready features."""
        if not self.is_fitted_:
            raise RuntimeError("ReturnsFeaturePipeline instance is not fitted yet. Call fit() first.")

        base_df = self._extract_base_features(X)

        # Recompute smoothed prior rate using strictly fitted train priors
        base_df["smoothed_prior_return_rate"] = (
            base_df["customer_prior_returns"] + self.prior_alpha_
        ) / (base_df["customer_prior_orders"] + self.prior_beta_)

        # Separate numeric and categorical features
        num_cols = [c for c in base_df.columns if c not in self.categorical_columns]
        num_df = base_df[num_cols].copy()

        # Impute numeric columns with fitted medians
        for col in num_cols:
            if col in self.numeric_medians_:
                num_df[col] = num_df[col].fillna(self.numeric_medians_[col])

        # Transform categoricals with fitted encoder
        cat_data = base_df[self.categorical_columns].astype(str)
        encoded_arr = self.encoder_.transform(cat_data)
        encoded_cat_names = list(self.encoder_.get_feature_names_out(self.categorical_columns))
        encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cat_names, index=X.index)

        # Combine all features
        features_df = pd.concat([num_df, encoded_df], axis=1)

        # Enforce exact column alignment
        features_df = features_df[self.feature_names_]

        # Preserve order_id at the front
        output_parts: List[pd.DataFrame] = []
        if self.include_id and ID_COLUMN in X.columns:
            id_df = pd.DataFrame({ID_COLUMN: X[ID_COLUMN]}, index=X.index)
            output_parts.append(id_df)

        output_parts.append(features_df)

        # Preserve returned target at the end (for train datasets)
        if self.include_target and TARGET_COLUMN in X.columns:
            target_df = pd.DataFrame({TARGET_COLUMN: X[TARGET_COLUMN].astype(int)}, index=X.index)
            output_parts.append(target_df)

        result_df = pd.concat(output_parts, axis=1)
        return result_df


def build_feature_pipeline(**kwargs: Any) -> ReturnsFeaturePipeline:
    """Factory helper to construct a configured ReturnsFeaturePipeline instance."""
    return ReturnsFeaturePipeline(**kwargs)


def generate_processed_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, ReturnsFeaturePipeline]:
    """Builds, fits, and exports the clean train and test feature datasets.

    Generates:
    - data/processed/train_features.parquet
    - data/processed/test_features.parquet
    - data/processed/feature_pipeline.joblib

    Returns:
        Tuple of (train_features, test_features, fitted_pipeline).
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cleaned and merged raw datasets...")
    train_merged = load_and_merge_train()
    test_merged = load_and_merge_test()

    pipeline = build_feature_pipeline(include_id=True, include_target=True)

    logger.info("Fitting pipeline on training data exclusively...")
    pipeline.fit(train_merged)

    logger.info("Transforming train dataset...")
    train_features = pipeline.transform(train_merged)

    logger.info("Transforming test dataset...")
    test_features = pipeline.transform(test_merged)

    logger.info("Saving processed datasets to Parquet...")
    train_features.to_parquet(TRAIN_PROCESSED_PATH, index=False)
    test_features.to_parquet(TEST_PROCESSED_PATH, index=False)

    logger.info("Persisting fitted feature pipeline with joblib...")
    joblib.dump(pipeline, FEATURE_PIPELINE_PATH)

    logger.info("Pipeline generation complete:")
    logger.info("  train_features saved: %s (shape: %s)", TRAIN_PROCESSED_PATH, train_features.shape)
    logger.info("  test_features saved:  %s (shape: %s)", TEST_PROCESSED_PATH, test_features.shape)
    logger.info("  pipeline saved:       %s", FEATURE_PIPELINE_PATH)

    return train_features, test_features, pipeline


def predict_order_features(
    order_payload: Union[Dict[str, Any], pd.DataFrame],
    pipeline: ReturnsFeaturePipeline,
    customers_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Transforms a single raw order or batch into model-ready features for API inference.

    Args:
        order_payload: Single dict or DataFrame representing a warehouse dispatch order.
        pipeline: Fitted ReturnsFeaturePipeline instance.
        customers_df: Optional customer table for join lookup.
        products_df: Optional products catalog for join lookup.

    Returns:
        DataFrame containing the transformed feature vector ready for model inference.
    """
    if isinstance(order_payload, dict):
        order_df = pd.DataFrame([order_payload])
    else:
        order_df = order_payload.copy()

    # Load lookup tables if not provided
    if customers_df is None or products_df is None:
        from src.data import load_raw_data

        _, _, cust_raw, prod_raw = load_raw_data()
        customers_df = customers_df if customers_df is not None else cust_raw
        products_df = products_df if products_df is not None else prod_raw

    # Perform validated join and clean
    merged = merge_order_data(
        orders_df=order_df,
        customers_df=customers_df,
        products_df=products_df,
        drop_leakage=True,
        validate_cardinality=False,  # Skip strict table uniqueness check on single record
    )

    # Transform using fitted pipeline
    features = pipeline.transform(merged)
    return features

