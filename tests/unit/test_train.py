from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_mlflow.train import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    evaluate_predictions,
    prepare_training_features,
    train_models,
    train_random_forest_regressor,
    train_xgboost_regressor,
)


@pytest.fixture
def sample_taxi_dataframe() -> pd.DataFrame:
    """Provide a small synthetic DataFrame simulating taxi trip records."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "tpep_pickup_datetime": pd.date_range("2026-01-01 08:00:00", periods=n, freq="15min"),
        "tpep_dropoff_datetime": pd.date_range("2026-01-01 08:15:00", periods=n, freq="15min"),
        "passenger_count": np.random.choice([1, 2, 3], size=n),
        "trip_distance": np.random.uniform(0.5, 15.0, size=n),
        "RatecodeID": np.random.choice([1, 2], size=n),
        "PULocationID": np.random.choice([100, 101, 102], size=n),
        "DOLocationID": np.random.choice([200, 201, 202], size=n),
        "fare_amount": np.random.uniform(5.0, 50.0, size=n),
    })


def test_prepare_training_features_derives_time_columns(sample_taxi_dataframe):
    x, y = prepare_training_features(sample_taxi_dataframe, target_col="fare_amount")

    assert "pickup_hour" in x.columns
    assert "pickup_day_of_week" in x.columns
    assert "trip_distance" in x.columns
    assert len(x) == len(sample_taxi_dataframe)
    assert len(y) == len(sample_taxi_dataframe)


def test_prepare_training_features_filters_invalid_rows():
    bad_data = pd.DataFrame({
        "trip_distance": [1.0, -1.0, 2.0, 0.0],
        "passenger_count": [1, 1, 0, 2],
        "PULocationID": [1, 2, 3, 4],
        "DOLocationID": [10, 20, 30, 40],
        "RatecodeID": [1, 1, 1, 1],
        "pickup_hour": [10, 11, 12, 13],
        "pickup_day_of_week": [1, 2, 3, 4],
        "fare_amount": [15.0, 20.0, 25.0, -5.0],
    })
    x, y = prepare_training_features(bad_data, target_col="fare_amount")
    assert len(x) == 1
    assert len(y) == 1
    assert y.iloc[0] == 15.0


def test_prepare_training_features_raises_on_missing_features():
    df = pd.DataFrame({"col_a": [1, 2], "fare_amount": [10.0, 20.0]})
    with pytest.raises(ValueError, match="None of the requested feature columns"):
        prepare_training_features(df, feature_cols=["non_existent_col"])


def test_evaluate_predictions():
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 31.0])
    metrics = evaluate_predictions(y_true, y_pred)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]


def test_train_random_forest_regressor(sample_taxi_dataframe):
    x, y = prepare_training_features(sample_taxi_dataframe)
    split_idx = 40
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model, metrics = train_random_forest_regressor(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        params={"n_estimators": 5, "max_depth": 3, "random_state": 42},
        log_to_mlflow=False,
    )
    assert model is not None
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics


def test_train_xgboost_regressor(sample_taxi_dataframe):
    x, y = prepare_training_features(sample_taxi_dataframe)
    split_idx = 40
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model, metrics = train_xgboost_regressor(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        params={"n_estimators": 5, "max_depth": 3, "random_state": 42},
        log_to_mlflow=False,
    )
    assert model is not None
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics


@patch("nyc_taxi_mlflow.taxis.find_all_taxis")
def test_train_models_end_to_end(mock_find_all, sample_taxi_dataframe):
    mock_spark = MagicMock()
    mock_df = MagicMock()
    mock_df.limit.return_value.toPandas.return_value = sample_taxi_dataframe
    mock_df.toPandas.return_value = sample_taxi_dataframe
    mock_find_all.return_value = mock_df

    results = train_models(
        spark_session=mock_spark,
        catalog="dev",
        schema="test",
        table_name="raw_taxis",
        max_rows=100,
        rf_params={"n_estimators": 2, "max_depth": 2},
        xgb_params={"n_estimators": 2, "max_depth": 2},
        log_to_mlflow=False,
    )

    assert "random_forest" in results
    assert "xgboost" in results
    assert "rmse" in results["random_forest"]["metrics"]
    assert "rmse" in results["xgboost"]["metrics"]


@patch("mlflow.start_run")
@patch("mlflow.log_metrics")
@patch("mlflow.log_param")
@patch("mlflow.set_tag")
@patch("nyc_taxi_mlflow.train.train_xgboost_regressor")
@patch("nyc_taxi_mlflow.train.train_random_forest_regressor")
@patch("nyc_taxi_mlflow.taxis.find_all_taxis")
def test_train_models_with_mlflow(
    mock_find_all,
    mock_rf,
    mock_xgb,
    mock_tag,
    mock_param,
    mock_metrics,
    mock_start_run,
    sample_taxi_dataframe,
):
    mock_spark = MagicMock()
    mock_df = MagicMock()
    mock_df.limit.return_value.toPandas.return_value = sample_taxi_dataframe
    mock_df.toPandas.return_value = sample_taxi_dataframe
    mock_find_all.return_value = mock_df

    mock_rf.return_value = (MagicMock(), {"rmse": 1.5, "mae": 1.0, "r2": 0.85})
    mock_xgb.return_value = (MagicMock(), {"rmse": 1.2, "mae": 0.9, "r2": 0.90})

    results = train_models(
        spark_session=mock_spark,
        catalog="dev",
        schema="test",
        table_name="raw_taxis",
        max_rows=100,
        log_to_mlflow=True,
    )

    assert "random_forest" in results
    assert "xgboost" in results
    assert mock_start_run.called
    assert mock_rf.called
    assert mock_xgb.called


def test_default_constants():
    assert "trip_distance" in DEFAULT_FEATURE_COLS
    assert "passenger_count" in DEFAULT_FEATURE_COLS
    assert DEFAULT_TARGET_COL == "fare_amount"
