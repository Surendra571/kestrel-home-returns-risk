"""Unit and integration tests for data ingestion, cleaning, validation, and merging."""

import pandas as pd
import pytest

from src.config import (
    CUSTOMERS_RAW_PATH,
    ID_COLUMN,
    LEAKAGE_COLUMNS,
    PRODUCTS_RAW_PATH,
    TARGET_COLUMN,
    TEST_RAW_PATH,
    TRAIN_RAW_PATH,
)
from src.data import (
    clean_train_data,
    fix_currency_bug,
    load_and_merge_test,
    load_and_merge_train,
    load_raw_data,
    merge_order_data,
    validate_join_cardinality,
)


def test_raw_files_exist_and_untouched():
    """Verify that raw files exist and have exact original dimensions."""
    assert TRAIN_RAW_PATH.exists()
    assert TEST_RAW_PATH.exists()
    assert CUSTOMERS_RAW_PATH.exists()
    assert PRODUCTS_RAW_PATH.exists()

    train_df = pd.read_csv(TRAIN_RAW_PATH)
    test_df = pd.read_csv(TEST_RAW_PATH)
    cust_df = pd.read_csv(CUSTOMERS_RAW_PATH)
    prod_df = pd.read_csv(PRODUCTS_RAW_PATH)

    assert len(train_df) == 11155, "Raw train.csv row count was modified!"
    assert len(test_df) == 2096, "Raw test_unlabelled.csv row count was modified!"
    assert len(cust_df) == 9000, "Raw customers.csv row count was modified!"
    assert len(prod_df) == 21, "Raw products.csv row count was modified!"


def test_load_raw_data():
    """Verify load_raw_data returns expected DataFrames."""
    train, test, cust, prod = load_raw_data()
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(cust, pd.DataFrame)
    assert isinstance(prod, pd.DataFrame)
    assert len(train) == 11155
    assert len(test) == 2096


def test_clean_train_data_deduplication():
    """Verify clean_train_data drops exactly 651 duplicate records and prioritizes CRM."""
    raw_train = pd.read_csv(TRAIN_RAW_PATH)
    cleaned = clean_train_data(raw_train)

    assert len(cleaned) == 10504
    assert cleaned[ID_COLUMN].nunique() == 10504
    # All duplicate partner_feed orders should have been dropped in favor of crm
    assert (cleaned["source"] == "crm").all()


def test_fix_currency_bug():
    """Verify that October 2025 paise bug is fixed and other dates remain untouched."""
    dummy_data = pd.DataFrame({
        ID_COLUMN: ["O1", "O2", "O3"],
        "order_placed_at": ["2025-10-15 10:00", "2025-09-15 10:00", "2025-10-20 12:00"],
        "order_value_inr": [500000.0, 5000.0, 87956.0],
    })
    fixed = fix_currency_bug(dummy_data)

    assert fixed.loc[0, "order_value_inr"] == 5000.0
    assert fixed.loc[1, "order_value_inr"] == 5000.0  # September unchanged
    assert fixed.loc[2, "order_value_inr"] == 879.56  # October small value corrected


def test_validate_join_cardinality_success_and_failure():
    """Verify cardinality checks pass on valid keys and raise on duplicate parent keys."""
    orders = pd.DataFrame({"order_id": ["O1", "O2"], "customer_id": ["C1", "C2"]})
    valid_customers = pd.DataFrame({"customer_id": ["C1", "C2"], "city": ["Delhi", "Mumbai"]})

    report = validate_join_cardinality(orders, valid_customers, "customer_id", "customers")
    assert report["unmatched_orders_count"] == 0
    assert report["unmatched_unique_keys"] == 0

    # Test failure on duplicate parent key
    invalid_customers = pd.DataFrame({
        "customer_id": ["C1", "C1"],
        "city": ["Delhi", "Mumbai"],
    })
    with pytest.raises(ValueError, match="Cardinality violation"):
        validate_join_cardinality(orders, invalid_customers, "customer_id", "customers")


def test_validate_join_reports_unmatched_records():
    """Verify cardinality validation reports unmatched child records."""
    orders = pd.DataFrame({"order_id": ["O1", "O2", "O3"], "customer_id": ["C1", "C2", "C999"]})
    customers = pd.DataFrame({"customer_id": ["C1", "C2"], "city": ["Delhi", "Mumbai"]})

    report = validate_join_cardinality(orders, customers, "customer_id", "customers")
    assert report["unmatched_orders_count"] == 1
    assert report["unmatched_orders_pct"] == pytest.approx(33.33, rel=1e-2)


def test_merge_order_data_drops_leakage_and_preserves_rows():
    """Verify merge_order_data drops leakage columns and preserves order count."""
    orders = pd.DataFrame({
        "order_id": ["O1"],
        "customer_id": ["C1"],
        "sku": ["P1"],
        "order_placed_at": ["2026-01-01 10:00"],
        "order_value_inr": [1000.0],
        "pickup_scheduled_at": ["2026-01-05 10:00"],
        "last_service_event_type": ["REVERSE_PICKUP"],
        "source": ["crm"],
        "returned": [1],
    })
    customers = pd.DataFrame({
        "customer_id": ["C1"],
        "city": ["Delhi"],
        "state": ["DL"],
        "signup_date": ["2025-01-01"],
        "shield_member": ["N"],
    })
    products = pd.DataFrame({
        "sku": ["P1"],
        "family": ["Air Fryer"],
        "model_name": ["AeroFry Pro"],
        "list_price_inr": [5000],
        "warranty_months": [12],
        "launch_date": ["2024-01-01"],
    })

    merged = merge_order_data(orders, customers, products, drop_leakage=True)
    assert len(merged) == 1
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in merged.columns
    assert "source" not in merged.columns
    assert "city" in merged.columns
    assert "family" in merged.columns
    assert pd.api.types.is_datetime64_any_dtype(merged["order_placed_at"])
    assert pd.api.types.is_datetime64_any_dtype(merged["signup_date"])
    assert pd.api.types.is_datetime64_any_dtype(merged["launch_date"])


def test_load_and_merge_train_and_test():
    """Verify end-to-end load and merge for both train and test splits."""
    train_merged = load_and_merge_train()
    test_merged = load_and_merge_test()

    assert len(train_merged) == 10504
    assert len(test_merged) == 2096
    assert ID_COLUMN in train_merged.columns
    assert ID_COLUMN in test_merged.columns
    assert TARGET_COLUMN in train_merged.columns
    assert TARGET_COLUMN not in test_merged.columns

    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in train_merged.columns
        assert leak_col not in test_merged.columns


def test_train_data_row_accounting_reconciliation():
    """Verify exact 651-row deduplication reconciliation from raw to fit and validation."""
    raw_train = pd.read_csv(TRAIN_RAW_PATH)
    original_train_rows = len(raw_train)
    assert original_train_rows == 11155

    cleaned = clean_train_data(raw_train)
    duplicate_rows_removed = original_train_rows - len(cleaned)
    assert duplicate_rows_removed == 651

    # Ingestion integrity assertions
    missing_target_rows_removed = int(cleaned[TARGET_COLUMN].isnull().sum())
    assert missing_target_rows_removed == 0

    final_modeling_rows = len(cleaned)
    assert final_modeling_rows == 10504

    fit_mask = cleaned["order_placed_at"] < "2026-04-01"
    val_mask = cleaned["order_placed_at"] >= "2026-04-01"
    fit_rows = int(fit_mask.sum())
    validation_rows = int(val_mask.sum())

    assert fit_rows == 8378
    assert validation_rows == 2126
    assert fit_rows + validation_rows == final_modeling_rows
    assert original_train_rows - duplicate_rows_removed == fit_rows + validation_rows


def test_target_column_binary_validity():
    """Verify target column contains only binary 0 and 1 with no missing values."""
    raw_train = pd.read_csv(TRAIN_RAW_PATH)
    assert set(raw_train[TARGET_COLUMN].unique()) == {0, 1}
    assert raw_train[TARGET_COLUMN].isnull().sum() == 0

