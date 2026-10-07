"""Unit and integration tests for pre-shipment feature engineering and pipeline reusability."""

import io
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import ID_COLUMN, LEAKAGE_COLUMNS, TARGET_COLUMN
from src.data import load_and_merge_test, load_and_merge_train
from src.features import (
    ReturnsFeaturePipeline,
    build_feature_pipeline,
    predict_order_features,
)


@pytest.fixture(scope="module")
def merged_train_df():
    """Cached merged training DataFrame."""
    return load_and_merge_train()


@pytest.fixture(scope="module")
def merged_test_df():
    """Cached merged test DataFrame."""
    return load_and_merge_test()


@pytest.fixture(scope="module")
def fitted_pipeline(merged_train_df):
    """Fitted ReturnsFeaturePipeline on training data only."""
    pipe = build_feature_pipeline(include_id=True, include_target=True)
    pipe.fit(merged_train_df)
    return pipe


def test_pipeline_fit_transform_train(merged_train_df, fitted_pipeline):
    """Verify feature pipeline transforms train set without NaNs and preserves order_id."""
    train_feats = fitted_pipeline.transform(merged_train_df)

    assert len(train_feats) == len(merged_train_df)
    assert train_feats.isna().sum().sum() == 0, "Transformed train contains NaN values!"
    assert ID_COLUMN in train_feats.columns
    assert TARGET_COLUMN in train_feats.columns
    assert (train_feats[ID_COLUMN] == merged_train_df[ID_COLUMN]).all()


def test_pipeline_transform_test(merged_test_df, fitted_pipeline):
    """Verify feature pipeline transforms test set without NaNs and without target column."""
    test_feats = fitted_pipeline.transform(merged_test_df)

    assert len(test_feats) == len(merged_test_df)
    assert test_feats.isna().sum().sum() == 0, "Transformed test contains NaN values!"
    assert ID_COLUMN in test_feats.columns
    assert TARGET_COLUMN not in test_feats.columns
    assert (test_feats[ID_COLUMN] == merged_test_df[ID_COLUMN]).all()


def test_no_target_leakage_in_features(fitted_pipeline):
    """Verify target and post-outcome leakage variables are strictly absent from features."""
    feature_names = fitted_pipeline.feature_names_

    assert TARGET_COLUMN not in feature_names
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in feature_names
    assert "source" not in feature_names


def test_train_test_feature_parity(merged_train_df, merged_test_df, fitted_pipeline):
    """Verify exact feature alignment between train and test outputs."""
    train_feats = fitted_pipeline.transform(merged_train_df)
    test_feats = fitted_pipeline.transform(merged_test_df)

    train_cols_no_target = [c for c in train_feats.columns if c != TARGET_COLUMN]
    test_cols = list(test_feats.columns)

    assert train_cols_no_target == test_cols, "Feature columns mismatch between train and test!"


def test_no_fitting_on_test_data(merged_test_df, fitted_pipeline):
    """Verify that transforming test data does not mutate any learned pipeline statistics."""
    alpha_before = fitted_pipeline.prior_alpha_
    beta_before = fitted_pipeline.prior_beta_
    medians_before = dict(fitted_pipeline.numeric_medians_)

    # Transform test
    _ = fitted_pipeline.transform(merged_test_df)

    assert fitted_pipeline.prior_alpha_ == alpha_before
    assert fitted_pipeline.prior_beta_ == beta_before
    assert fitted_pipeline.numeric_medians_ == medians_before


def test_api_single_record_prediction(fitted_pipeline):
    """Verify predict_order_features transforms a single payload into a valid feature row."""
    sample_order = {
        "order_id": "KO_TEST_001",
        "order_placed_at": "2026-10-06 14:00",
        "customer_id": "KC100001",
        "sku": "KH-RV-03",
        "sales_channel": "app",
        "payment_mode": "cod",
        "discount_pct": 20,
        "qty": 1,
        "order_value_inr": 23758.4,
        "promised_delivery_days": 4,
        "delivery_pincode": 560001,
        "is_gift": "N",
        "customer_prior_orders": 2,
        "customer_prior_returns": 1,
        "delivery_note": "Please call before delivery",
    }

    feat_row = predict_order_features(sample_order, fitted_pipeline)

    assert isinstance(feat_row, pd.DataFrame)
    assert len(feat_row) == 1
    assert feat_row.isna().sum().sum() == 0
    assert feat_row[ID_COLUMN].iloc[0] == "KO_TEST_001"
    assert feat_row["is_cod"].iloc[0] == 1
    assert feat_row["requires_installation"].iloc[0] == 1
    assert feat_row["note_mentions_call"].iloc[0] == 1


def test_pipeline_serialization_and_reusability(merged_test_df, fitted_pipeline):
    """Verify the fitted pipeline can be serialized and produces identical transforms."""
    buffer = io.BytesIO()
    joblib.dump(fitted_pipeline, buffer)
    buffer.seek(0)

    loaded_pipeline = joblib.load(buffer)
    out1 = fitted_pipeline.transform(merged_test_df)
    out2 = loaded_pipeline.transform(merged_test_df)

    pd.testing.assert_frame_equal(out1, out2)


def test_generate_submission(tmp_path):
    """Verify test submission generation creates valid format with matching rows and no NaNs."""
    from src.predict_submission import generate_final_predictions
    from src.config import PROJECT_ROOT, RAW_DATA_DIR

    sub_path = tmp_path / "predictions_test.csv"
    sub = generate_final_predictions(
        output_csv_path=sub_path,
    )
    assert sub_path.exists()
    assert len(sub) == 2096
    assert list(sub.columns) == ["order_id", "score"]
    assert sub["score"].isnull().sum() == 0
    assert (sub["score"] >= 0.0).all() and (sub["score"] <= 1.0).all()

    # Verify predictions.csv at PROJECT_ROOT
    root_preds = PROJECT_ROOT / "predictions.csv"
    assert root_preds.exists()
    root_df = pd.read_csv(root_preds)
    assert len(root_df) == 2096
    assert list(root_df.columns) == ["order_id", "score"]
    sample_sub = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
    assert (root_df["order_id"] == sample_sub["order_id"]).all()
    assert root_df["order_id"].duplicated().sum() == 0
    assert root_df["score"].isnull().sum() == 0


def test_bayes_smoothing_invariance_to_future_data(fitted_pipeline, merged_train_df):
    """Verify adding future orders/returns to the evaluation set cannot alter historical smoothed priors."""
    test_record = merged_train_df.iloc[[0]].copy()
    initial_transform = fitted_pipeline.transform(test_record)
    initial_smoothed_rate = initial_transform["smoothed_prior_return_rate"].iloc[0]

    # Create dummy dataset appending 100 hypothetical future orders with extreme returns
    future_orders = merged_train_df.iloc[:100].copy()
    future_orders["customer_prior_orders"] = 10
    future_orders["customer_prior_returns"] = 10
    future_orders["order_placed_at"] = "2027-01-01 12:00:00"

    augmented_set = pd.concat([test_record, future_orders], ignore_index=True)
    augmented_transform = fitted_pipeline.transform(augmented_set)

    # First row value must remain exactly invariant
    assert augmented_transform["smoothed_prior_return_rate"].iloc[0] == pytest.approx(initial_smoothed_rate, rel=1e-6)


def test_prior_return_rate_calculation(fitted_pipeline, merged_train_df):
    """Verify prior return rate strictly computes customer_prior_returns / customer_prior_orders."""
    feats = fitted_pipeline.transform(merged_train_df)
    orders = merged_train_df["customer_prior_orders"].fillna(0).values
    returns = merged_train_df["customer_prior_returns"].fillna(0).values

    expected_rates = np.where(orders > 0, returns / np.maximum(orders, 1.0), 0.0)
    np.testing.assert_allclose(feats["prior_return_rate"].values, expected_rates, rtol=1e-5)


def test_model_score_range_and_determinism(fitted_pipeline, merged_test_df):
    """Verify model predictions on test data are strictly bounded in [0, 1] and deterministic."""
    from src.config import MODELS_DIR
    model = joblib.load(MODELS_DIR / "selected_model.joblib")

    X = fitted_pipeline.transform(merged_test_df)
    feature_cols = [c for c in X.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
    probs1 = model.predict_proba(X[feature_cols])[:, 1]
    probs2 = model.predict_proba(X[feature_cols])[:, 1]

    assert (probs1 >= 0.0).all() and (probs1 <= 1.0).all()
    np.testing.assert_array_equal(probs1, probs2)

