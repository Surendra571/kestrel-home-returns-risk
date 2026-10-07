"""Model training and evaluation pipeline for Kestrel Home Returns Risk baselines.

Implements an Out-of-Time (OOT) temporal validation strategy and trains four
leakage-safe baseline models:
1. Majority-Class baseline
2. Simple Rule / Empirical Segment Probability baseline
3. Regularized Logistic Regression baseline
4. Extreme Gradient Boosting (XGBoost) tabular baseline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.business_metrics import OPTIMAL_ECONOMIC_THRESHOLD
from src.config import (
    ID_COLUMN,
    ORDER_DATE_COL,
    PROJECT_ROOT,
    TARGET_COLUMN,
)
from src.data import load_and_merge_train
from src.evaluate import evaluate_predictions, format_evaluation_summary
from src.features import ReturnsFeaturePipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VAL_SPLIT_DATE = "2026-04-01"


# -----------------------------------------------------------------------------
# Baseline 1: Majority-Class Classifier
# -----------------------------------------------------------------------------
class MajorityClassBaseline:
    """Predicts the negative class (never return) for all orders.

    Output probability is constant and equal to the training set base return rate.
    """

    def __init__(self) -> None:
        self.base_rate_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityClassBaseline":
        self.base_rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.full((len(X), 2), [1.0 - self.base_rate_, self.base_rate_])
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


# -----------------------------------------------------------------------------
# Baseline 2: Simple Rule & Empirical Segment Probability Model
# -----------------------------------------------------------------------------
class RuleBasedBaseline:
    """Business rule and empirical Bayesian segment risk model.

    Segments orders by primary risk drivers (payment_mode, has_prior_returns,
    requires_installation, is_shield_member) and estimates segment return rates
    with Empirical Bayes smoothing.
    """

    def __init__(self, threshold: float = OPTIMAL_ECONOMIC_THRESHOLD) -> None:
        self.threshold = threshold
        self.segment_probs_: Dict[Tuple, float] = {}
        self.global_rate_: float = 0.114

    def _get_segment_keys(self, X: pd.DataFrame) -> pd.Series:
        # Extract interpretable segment key
        is_cod = (X["is_cod"] == 1).astype(int)
        has_returns = (X["has_prior_returns"] == 1).astype(int)
        req_install = (X["requires_installation"] == 1).astype(int)
        is_shield = (X["is_shield_member"] == 1).astype(int)
        return list(zip(is_cod, has_returns, req_install, is_shield))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RuleBasedBaseline":
        self.global_rate_ = float(np.mean(y))
        seg_keys = self._get_segment_keys(X)
        temp_df = pd.DataFrame({"seg": seg_keys, "y": y})

        smoothing_alpha = self.global_rate_ * 10.0
        smoothing_beta = (1.0 - self.global_rate_) * 10.0

        grouped = temp_df.groupby("seg")["y"].agg(["sum", "count"])
        for seg, row in grouped.iterrows():
            prob = (row["sum"] + smoothing_alpha) / (row["count"] + smoothing_alpha + smoothing_beta)
            self.segment_probs_[seg] = float(prob)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        seg_keys = self._get_segment_keys(X)
        pos_probs = np.array([self.segment_probs_.get(k, self.global_rate_) for k in seg_keys])
        probs = np.column_stack([1.0 - pos_probs, pos_probs])
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pos_probs = self.predict_proba(X)[:, 1]
        return (pos_probs >= self.threshold).astype(int)


# -----------------------------------------------------------------------------
# Temporal Splitting Helper
# -----------------------------------------------------------------------------
def get_temporal_split(
    split_date: str = VAL_SPLIT_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Partitions historical orders into train (12 months) and out-of-time val (3 months).

    Fits feature pipeline strictly on the train period to eliminate data leakage.

    Returns:
        X_train, X_val, y_train, y_val, train_order_ids, val_order_ids
    """
    logger.info("Loading merged training data for temporal split at %s...", split_date)
    raw_merged = load_and_merge_train()

    dates = pd.to_datetime(raw_merged[ORDER_DATE_COL])
    train_mask = dates < split_date
    val_mask = dates >= split_date

    train_raw = raw_merged[train_mask].copy().reset_index(drop=True)
    val_raw = raw_merged[val_mask].copy().reset_index(drop=True)

    logger.info(
        "Temporal Split created: Train=%d orders (%.2f%% returns), Val=%d orders (%.2f%% returns)",
        len(train_raw),
        train_raw[TARGET_COLUMN].mean() * 100,
        len(val_raw),
        val_raw[TARGET_COLUMN].mean() * 100,
    )

    # Fit pipeline strictly on train_raw
    pipeline = ReturnsFeaturePipeline(include_id=True, include_target=True)
    pipeline.fit(train_raw)

    train_transformed = pipeline.transform(train_raw)
    val_transformed = pipeline.transform(val_raw)

    train_order_ids = train_transformed[ID_COLUMN].tolist()
    val_order_ids = val_transformed[ID_COLUMN].tolist()

    y_train = train_transformed[TARGET_COLUMN].astype(int)
    y_val = val_transformed[TARGET_COLUMN].astype(int)

    feature_cols = [c for c in train_transformed.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
    X_train = train_transformed[feature_cols].copy()
    X_val = val_transformed[feature_cols].copy()

    return X_train, X_val, y_train, y_val, train_order_ids, val_order_ids


# -----------------------------------------------------------------------------
# Main Training & Evaluation Routine
# -----------------------------------------------------------------------------
def run_baseline_training() -> Dict[str, Any]:
    """Trains and benchmarks all 4 baseline models on out-of-time validation data."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val, _, val_order_ids = get_temporal_split()

    all_metrics: Dict[str, Any] = {}
    predictions_df = pd.DataFrame({
        "order_id": val_order_ids,
        "actual_returned": y_val.values,
    })

    # -------------------------------------------------------------------------
    # 1. Majority-Class Baseline
    # -------------------------------------------------------------------------
    logger.info("Evaluating Baseline 1: Majority-Class Classifier...")
    majority_model = MajorityClassBaseline()
    majority_model.fit(X_train, y_train)

    majority_prob = majority_model.predict_proba(X_val)[:, 1]
    majority_pred = majority_model.predict(X_val)
    predictions_df["majority_prob"] = majority_prob
    predictions_df["majority_pred"] = majority_pred

    metrics_majority = evaluate_predictions(y_val, majority_pred, majority_prob, threshold=0.5)
    all_metrics["majority_class"] = metrics_majority
    print(format_evaluation_summary(metrics_majority, "Majority-Class Baseline"))

    # -------------------------------------------------------------------------
    # 2. Simple Rule / Empirical Segment Probability Baseline
    # -------------------------------------------------------------------------
    logger.info("Evaluating Baseline 2: Rule-Based / Empirical Probability Model...")
    rule_model = RuleBasedBaseline(threshold=OPTIMAL_ECONOMIC_THRESHOLD)
    rule_model.fit(X_train, y_train)

    rule_prob = rule_model.predict_proba(X_val)[:, 1]
    rule_pred = rule_model.predict(X_val)
    predictions_df["rule_prob"] = rule_prob
    predictions_df["rule_pred"] = rule_pred

    metrics_rule = evaluate_predictions(y_val, rule_pred, rule_prob, threshold=OPTIMAL_ECONOMIC_THRESHOLD)
    all_metrics["rule_based"] = metrics_rule
    print(format_evaluation_summary(metrics_rule, "Rule-Based Segment Probability Baseline"))

    # -------------------------------------------------------------------------
    # 3. Logistic Regression Baseline
    # -------------------------------------------------------------------------
    logger.info("Evaluating Baseline 3: Regularized Logistic Regression...")
    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")),
    ])
    logreg_pipe.fit(X_train, y_train)

    logreg_prob = logreg_pipe.predict_proba(X_val)[:, 1]
    logreg_pred_05 = (logreg_prob >= 0.50).astype(int)
    logreg_pred_opt = (logreg_prob >= OPTIMAL_ECONOMIC_THRESHOLD).astype(int)

    predictions_df["logreg_prob"] = logreg_prob
    predictions_df["logreg_pred_0.5"] = logreg_pred_05
    predictions_df["logreg_pred_optimal"] = logreg_pred_opt

    metrics_logreg_05 = evaluate_predictions(y_val, logreg_pred_05, logreg_prob, threshold=0.50)
    metrics_logreg_opt = evaluate_predictions(
        y_val, logreg_pred_opt, logreg_prob, threshold=OPTIMAL_ECONOMIC_THRESHOLD
    )
    all_metrics["logistic_regression_default_thresh"] = metrics_logreg_05
    all_metrics["logistic_regression_economic_thresh"] = metrics_logreg_opt
    print(format_evaluation_summary(metrics_logreg_05, "Logistic Regression (Threshold=0.50)"))
    print(format_evaluation_summary(metrics_logreg_opt, "Logistic Regression (Economic Threshold=0.1118)"))

    # -------------------------------------------------------------------------
    # 4. Strong Tree-Based Model: XGBoost
    # -------------------------------------------------------------------------
    logger.info("Evaluating Baseline 4: XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_pred_05 = (xgb_prob >= 0.50).astype(int)
    xgb_pred_opt = (xgb_prob >= OPTIMAL_ECONOMIC_THRESHOLD).astype(int)

    predictions_df["xgboost_prob"] = xgb_prob
    predictions_df["xgboost_pred_0.5"] = xgb_pred_05
    predictions_df["xgboost_pred_optimal"] = xgb_pred_opt

    metrics_xgb_05 = evaluate_predictions(y_val, xgb_pred_05, xgb_prob, threshold=0.50)
    metrics_xgb_opt = evaluate_predictions(
        y_val, xgb_pred_opt, xgb_prob, threshold=OPTIMAL_ECONOMIC_THRESHOLD
    )
    all_metrics["xgboost_default_thresh"] = metrics_xgb_05
    all_metrics["xgboost_economic_thresh"] = metrics_xgb_opt
    print(format_evaluation_summary(metrics_xgb_05, "XGBoost Classifier (Threshold=0.50)"))
    print(format_evaluation_summary(metrics_xgb_opt, "XGBoost Classifier (Economic Threshold=0.1118)"))

    # -------------------------------------------------------------------------
    # Save Deliverables
    # -------------------------------------------------------------------------
    metrics_path = OUTPUTS_DIR / "baseline_metrics.json"
    predictions_path = OUTPUTS_DIR / "validation_predictions.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved baseline metrics to: %s", metrics_path)

    predictions_df.to_csv(predictions_path, index=False)
    logger.info("Saved validation predictions to: %s (shape=%s)", predictions_path, predictions_df.shape)

    return all_metrics


if __name__ == "__main__":
    run_baseline_training()

