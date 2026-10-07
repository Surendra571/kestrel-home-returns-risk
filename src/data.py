"""Data ingestion, validation, and joining module for Kestrel Home Returns Risk.

Safely loads raw CSV files, enforces referential integrity, deduplicates records,
fixes historical CRM migration anomalies, and performs validated joins while
strictly dropping post-outcome leakage features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    CUSTOMER_ID_COLUMN,
    CUSTOMERS_RAW_PATH,
    ID_COLUMN,
    LAUNCH_DATE_COL,
    LEAKAGE_COLUMNS,
    METADATA_COLUMNS,
    OCT_2025_END,
    OCT_2025_START,
    ORDER_DATE_COL,
    PAISE_TO_RUPEE_FACTOR,
    PRODUCT_ID_COLUMN,
    PRODUCTS_RAW_PATH,
    RAW_DATA_DIR,
    SIGNUP_DATE_COL,
    TARGET_COLUMN,
    TEST_RAW_PATH,
    TRAIN_RAW_PATH,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_raw_data(
    data_dir: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Safely loads all raw data files without mutating raw disk files.

    Args:
        data_dir: Optional directory path override for raw files.

    Returns:
        Tuple of (train_raw, test_raw, customers_raw, products_raw).
    """
    base_dir = data_dir or RAW_DATA_DIR
    train_path = base_dir / "train.csv" if hasattr(base_dir, "__truediv__") else TRAIN_RAW_PATH
    test_path = base_dir / "test_unlabelled.csv" if hasattr(base_dir, "__truediv__") else TEST_RAW_PATH
    cust_path = base_dir / "customers.csv" if hasattr(base_dir, "__truediv__") else CUSTOMERS_RAW_PATH
    prod_path = base_dir / "products.csv" if hasattr(base_dir, "__truediv__") else PRODUCTS_RAW_PATH

    logger.info("Loading raw data from: %s", base_dir)
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    customers_raw = pd.read_csv(cust_path)
    products_raw = pd.read_csv(prod_path)

    logger.info(
        "Raw tables loaded successfully. Shapes: train=%s, test=%s, customers=%s, products=%s",
        train_raw.shape,
        test_raw.shape,
        customers_raw.shape,
        products_raw.shape,
    )
    return train_raw, test_raw, customers_raw, products_raw


def clean_train_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicates training records resulting from partner feed re-import.

    In train.csv, 651 order pairs (1,302 rows) exist where one row has
    source='crm' and the other has source='partner_feed'. Deduplication retains
    the primary 'crm' record and ensures 1 unique row per order_id.

    Args:
        train_df: Raw training DataFrame.

    Returns:
        Deduplicated DataFrame with 10,504 unique orders.
    """
    df = train_df.copy()
    initial_count = len(df)
    unique_orders = df[ID_COLUMN].nunique()

    # Prioritize 'crm' records over 'partner_feed' during deduplication
    if "source" in df.columns:
        df = df.sort_values(by="source", ascending=True)

    df = df.drop_duplicates(subset=[ID_COLUMN], keep="first").reset_index(drop=True)
    deduped_count = len(df)
    dropped_count = initial_count - deduped_count

    logger.info(
        "Deduplication complete: %d duplicate rows dropped (initial=%d, final=%d, unique_ids=%d)",
        dropped_count,
        initial_count,
        deduped_count,
        unique_orders,
    )
    return df


def fix_currency_bug(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects the October 2025 CRM migration paise bug in order_value_inr.

    Per Operations Policy v4.1 (§9), during the Zoho-to-Kestrel CRM migration in
    October 2025, payment amounts were recorded in paise rather than INR (100x).

    Args:
        df: DataFrame containing 'order_placed_at' and 'order_value_inr'.

    Returns:
        DataFrame with corrected 'order_value_inr'.
    """
    df_clean = df.copy()
    if "order_value_inr" not in df_clean.columns or ORDER_DATE_COL not in df_clean.columns:
        return df_clean

    dates = pd.to_datetime(df_clean[ORDER_DATE_COL])
    oct_mask = (dates >= OCT_2025_START) & (dates <= OCT_2025_END)

    # In October 2025, recorded values are in paise (100x normal rupee amounts)
    # Check both date range and value magnitude to be fully safe
    affected_mask = oct_mask & (df_clean["order_value_inr"] > 50000.0)
    # Also catch any lower-priced items in October (e.g. Room Heater Lite discounted)
    affected_all_oct = oct_mask & (df_clean["order_value_inr"] > 1000.0)

    num_affected = affected_all_oct.sum()
    if num_affected > 0:
        logger.info("Applying October 2025 currency fix to %d orders (dividing paise by 100.0)", num_affected)
        df_clean.loc[affected_all_oct, "order_value_inr"] = (
            df_clean.loc[affected_all_oct, "order_value_inr"] / PAISE_TO_RUPEE_FACTOR
        )

    return df_clean


def validate_join_cardinality(
    orders_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    join_key: str,
    parent_name: str,
) -> Dict[str, Any]:
    """Validates referential integrity and join cardinality before merging.

    Verifies that the parent table key is strictly unique (m:1 join) and reports
    any unmatched records in the child (orders) table.

    Args:
        orders_df: Orders DataFrame (child table).
        parent_df: Dimension DataFrame (parent table, e.g. customers or products).
        join_key: Primary/Foreign key column name.
        parent_name: Human-readable name of parent table for reporting.

    Returns:
        Dict containing validation statistics.

    Raises:
        ValueError: If the parent table contains duplicate keys.
    """
    total_orders = len(orders_df)
    unique_parent_keys = parent_df[join_key].nunique()
    total_parent_rows = len(parent_df)

    if unique_parent_keys != total_parent_rows:
        duplicate_count = total_parent_rows - unique_parent_keys
        raise ValueError(
            f"Cardinality violation in {parent_name}: {duplicate_count} duplicate keys found on '{join_key}'. "
            f"Join would cause undesirable row multiplication."
        )

    # Check for unmatched child records
    orders_keys = set(orders_df[join_key].dropna())
    parent_keys = set(parent_df[join_key].dropna())
    unmatched_keys = orders_keys - parent_keys
    unmatched_orders_mask = ~orders_df[join_key].isin(parent_keys)
    unmatched_order_count = int(unmatched_orders_mask.sum())
    unmatched_pct = (unmatched_order_count / total_orders * 100.0) if total_orders > 0 else 0.0

    report = {
        "parent_table": parent_name,
        "join_key": join_key,
        "total_orders": total_orders,
        "parent_row_count": total_parent_rows,
        "unmatched_orders_count": unmatched_order_count,
        "unmatched_orders_pct": unmatched_pct,
        "unmatched_unique_keys": len(unmatched_keys),
    }

    if unmatched_order_count > 0:
        logger.warning(
            "Unmatched records detected joining %s on '%s': %d orders (%.2f%%) have no matching parent record.",
            parent_name,
            join_key,
            unmatched_order_count,
            unmatched_pct,
        )
    else:
        logger.info(
            "Referential integrity verified: 100%% of orders match %s on '%s'.",
            parent_name,
            join_key,
        )

    return report


def merge_order_data(
    orders_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    drop_leakage: bool = True,
    validate_cardinality: bool = True,
) -> pd.DataFrame:
    """Performs validated left joins from orders to customers and products.

    Enforces that:
    1. Parent key cardinality is verified (no row duplication).
    2. Date columns are safely parsed to datetimes.
    3. October 2025 currency bug is patched.
    4. Post-outcome leakage columns (pickup_scheduled_at, last_service_event_type)
       are strictly dropped when drop_leakage=True.
    5. The resulting row count strictly matches the input orders row count.

    Args:
        orders_df: Orders DataFrame.
        customers_df: Customers dimension DataFrame.
        products_df: Products catalog dimension DataFrame.
        drop_leakage: Whether to strip target leakage columns.
        validate_cardinality: Whether to run referential validation checks.

    Returns:
        Merged and enriched DataFrame with preserved order_id.
    """
    initial_count = len(orders_df)
    orders = orders_df.copy()

    # Apply currency fix
    orders = fix_currency_bug(orders)

    # Validate cardinality
    if validate_cardinality:
        validate_join_cardinality(orders, customers_df, CUSTOMER_ID_COLUMN, "customers")
        validate_join_cardinality(orders, products_df, PRODUCT_ID_COLUMN, "products")

    # Left join to customers
    merged = orders.merge(customers_df, on=CUSTOMER_ID_COLUMN, how="left")
    if len(merged) != initial_count:
        raise ValueError(f"Customer join altered row count from {initial_count} to {len(merged)}!")

    # Left join to products
    merged = merged.merge(products_df, on=PRODUCT_ID_COLUMN, how="left")
    if len(merged) != initial_count:
        raise ValueError(f"Product join altered row count from {initial_count} to {len(merged)}!")

    # Parse dates safely
    for col in [ORDER_DATE_COL, SIGNUP_DATE_COL, LAUNCH_DATE_COL]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], errors="coerce")

    # Strictly drop leakage columns
    if drop_leakage:
        cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in merged.columns]
        if cols_to_drop:
            logger.info("Dropping post-outcome leakage columns: %s", cols_to_drop)
            merged = merged.drop(columns=cols_to_drop)

    # Drop non-feature metadata
    cols_meta = [c for c in METADATA_COLUMNS if c in merged.columns]
    if cols_meta:
        logger.info("Dropping metadata columns: %s", cols_meta)
        merged = merged.drop(columns=cols_meta)

    return merged


def load_and_merge_train(
    data_dir: Optional[Any] = None,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """Convenience function to load, clean, and merge training data."""
    train_raw, _, customers_raw, products_raw = load_raw_data(data_dir=data_dir)
    train_clean = clean_train_data(train_raw)
    return merge_order_data(
        orders_df=train_clean,
        customers_df=customers_raw,
        products_df=products_raw,
        drop_leakage=drop_leakage,
        validate_cardinality=True,
    )


def load_and_merge_test(
    data_dir: Optional[Any] = None,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """Convenience function to load and merge unlabelled test data."""
    _, test_raw, customers_raw, products_raw = load_raw_data(data_dir=data_dir)
    return merge_order_data(
        orders_df=test_raw,
        customers_df=customers_raw,
        products_df=products_raw,
        drop_leakage=drop_leakage,
        validate_cardinality=True,
    )

