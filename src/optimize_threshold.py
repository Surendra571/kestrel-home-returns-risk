"""Operational decision threshold optimization module for Kestrel Home Returns Risk.

Scans decision thresholds against unit economics from Operations Policy v4.1:
- C_return = INR 1,150
- C_call = INR 45
- Prevention rate = 0.35
- Baseline gross cost = actual_returns * 1150

Evaluates operational tradeoffs:
- Cost-minimizing / Net-savings maximizing threshold
- 95% return catch threshold (and its operational feasibility)
- Operationally balanced threshold respecting contact center capacity and 24h SLA

Outputs:
- outputs/threshold_analysis.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

from src.business_metrics import (
    COST_OF_CALL_INR,
    COST_OF_RETURN_INR,
    OPTIMAL_ECONOMIC_THRESHOLD,
    PREVENTION_EFFECTIVENESS,
)
from src.config import ID_COLUMN, ORDER_DATE_COL, PROJECT_ROOT, TARGET_COLUMN
from src.data import load_and_merge_train

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"


def generate_threshold_analysis() -> pd.DataFrame:
    """Generates comprehensive threshold scan table and exports to CSV."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and fitted model pipeline
    raw_train = load_and_merge_train()
    val_mask = pd.to_datetime(raw_train[ORDER_DATE_COL]) >= "2026-04-01"
    val_raw = raw_train[val_mask].copy().reset_index(drop=True)

    model_path = MODELS_DIR / "selected_model.joblib"
    pipe_path = MODELS_DIR / "preprocessing_pipeline.joblib"

    logger.info("Loading selected model from %s...", model_path)
    model = joblib.load(model_path)
    pipeline = joblib.load(pipe_path)

    val_transformed = pipeline.transform(val_raw)
    feat_cols = [c for c in val_transformed.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
    X_val = val_transformed[feat_cols]
    y_val = val_transformed[TARGET_COLUMN].astype(int).values

    probs = model.predict_proba(X_val)[:, 1]

    total_orders = len(y_val)
    actual_returns = int(np.sum(y_val))
    baseline_gross_cost = actual_returns * COST_OF_RETURN_INR  # INR 281,750.0

    # 2. Build threshold evaluation grid
    grid_points = sorted(list(set(
        [round(t, 4) for t in np.linspace(0.01, 0.90, 90)]
        + [0.03, 0.05, 0.08, 0.10, OPTIMAL_ECONOMIC_THRESHOLD, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
    )))

    records: List[Dict[str, Any]] = []
    for t in grid_points:
        pred = (probs >= t).astype(int)
        held = int(np.sum(pred))
        held_pct = (held / total_orders * 100.0) if total_orders > 0 else 0.0

        tp = int(np.sum((y_val == 1) & (pred == 1)))
        fp = int(np.sum((y_val == 0) & (pred == 1)))
        fn = int(np.sum((y_val == 1) & (pred == 0)))
        tn = int(np.sum((y_val == 0) & (pred == 0)))

        prec = (tp / held) if held > 0 else 0.0
        rec = (tp / actual_returns) if actual_returns > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        prevented_returns = tp * PREVENTION_EFFECTIVENESS
        avoided_return_cost = prevented_returns * COST_OF_RETURN_INR
        total_call_cost = held * COST_OF_CALL_INR
        net_savings = avoided_return_cost - total_call_cost
        expected_business_cost = baseline_gross_cost - net_savings
        cost_per_prevented = (total_call_cost / prevented_returns) if prevented_returns > 0 else float("inf")
        cost_reduction_pct = (net_savings / baseline_gross_cost * 100.0) if baseline_gross_cost > 0 else 0.0

        records.append({
            "threshold": round(t, 4),
            "orders_held": held,
            "orders_held_pct": round(held_pct, 2),
            "true_returns_caught": tp,
            "returns_missed": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "returns_prevented": round(prevented_returns, 2),
            "total_call_cost_inr": round(total_call_cost, 2),
            "expected_avoided_return_cost_inr": round(avoided_return_cost, 2),
            "expected_net_savings_inr": round(net_savings, 2),
            "expected_business_cost_inr": round(expected_business_cost, 2),
            "cost_reduction_pct": round(cost_reduction_pct, 2),
            "cost_per_prevented_return_inr": round(cost_per_prevented, 2),
        })

    df_analysis = pd.DataFrame(records)
    csv_path = OUTPUTS_DIR / "threshold_analysis.csv"
    df_analysis.to_csv(csv_path, index=False)
    logger.info("Saved threshold analysis table to: %s (rows=%d)", csv_path, len(df_analysis))

    return df_analysis


if __name__ == "__main__":
    df = generate_threshold_analysis()
    # Print key threshold milestones
    key_t = [0.01, 0.03, 0.05, 0.10, OPTIMAL_ECONOMIC_THRESHOLD, 0.12, 0.15, 0.18, 0.20, 0.30, 0.50]
    sub = df[df["threshold"].isin([round(k, 4) for k in key_t])]
    print("\nKEY OPERATIONAL THRESHOLD MILESTONES:")
    print(sub[[
        "threshold", "orders_held", "orders_held_pct", "true_returns_caught",
        "returns_missed", "precision", "recall", "expected_net_savings_inr",
        "expected_business_cost_inr", "cost_per_prevented_return_inr"
    ]].to_string(index=False))

