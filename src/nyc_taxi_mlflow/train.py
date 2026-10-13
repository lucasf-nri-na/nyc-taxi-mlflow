from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from nyc_taxi_mlflow import taxis

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLS = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
    "pickup_hour",
    "pickup_day_of_week",
]
DEFAULT_TARGET_COL = "fare_amount"


def _get_spark(spark_session: SparkSession | None = None) -> Any:
    """Resolve active SparkSession from parameter, databricks runtime, or active session."""
    if spark_session is not None:
        return spark_session
    try:
        from databricks.sdk.runtime import spark as runtime_spark

        return runtime_spark
    except Exception:
        pass
    try:
        from pyspark.sql import SparkSession

        active = SparkSession.getActiveSession()
        if active is not None:
            return active
    except Exception:
        pass
    raise RuntimeError("No active SparkSession available. Pass spark_session explicitly.")


def prepare_training_features(
    data: pd.DataFrame | DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = DEFAULT_TARGET_COL,
    max_rows: int | None = 100_000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, derive time features, and return feature matrix X and target y as pandas DataFrames."""
    if not isinstance(data, pd.DataFrame):
        # Convert Spark DataFrame to pandas with optional limit
        if max_rows is not None and max_rows > 0:
            pdf = data.limit(max_rows).toPandas()
        else:
            pdf = data.toPandas()
    else:
        pdf = data.copy()
        if max_rows is not None and max_rows > 0 and len(pdf) > max_rows:
            pdf = pdf.head(max_rows)

    # Derive pickup_hour and pickup_day_of_week if datetime column is present
    if "tpep_pickup_datetime" in pdf.columns:
        pickup_dt = pd.to_datetime(pdf["tpep_pickup_datetime"], errors="coerce")
        if "pickup_hour" not in pdf.columns:
            pdf["pickup_hour"] = pickup_dt.dt.hour
        if "pickup_day_of_week" not in pdf.columns:
            pdf["pickup_day_of_week"] = pickup_dt.dt.dayofweek

    cols_to_check = [target_col]
    features = list(feature_cols or DEFAULT_FEATURE_COLS)
    available_features = [col for col in features if col in pdf.columns]
    if not available_features:
        raise ValueError(f"None of the requested feature columns {features} are present in the dataset.")

    cols_to_check.extend(available_features)

    # Filter out missing or invalid target/features
    clean_df = pdf.dropna(subset=cols_to_check).copy()
    if target_col in clean_df.columns:
        clean_df = clean_df[clean_df[target_col] > 0.0]
    if "trip_distance" in clean_df.columns:
        clean_df = clean_df[clean_df["trip_distance"] > 0.0]
    if "passenger_count" in clean_df.columns:
        clean_df = clean_df[clean_df["passenger_count"] > 0]

    if clean_df.empty:
        raise ValueError("No valid rows remaining after filtering missing or invalid records.")

    x = clean_df[available_features].astype(float)
    y = clean_df[target_col].astype(float)
    return x, y


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Compute regression metrics: RMSE, MAE, and R-squared."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_random_forest_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any] | None = None,
    log_to_mlflow: bool = True,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    """Train a RandomForestRegressor model, evaluate metrics, and log to MLflow."""
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        rf_params.update(params)

    model = RandomForestRegressor(**rf_params)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = evaluate_predictions(y_test, predictions)
    logger.info("RandomForestRegressor evaluation metrics: %s", metrics)

    if log_to_mlflow:
        mlflow.log_params(rf_params)
        mlflow.log_metrics(metrics)
        try:
            mlflow.sklearn.log_model(
                model,
                name="random_forest_regressor",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        except TypeError:
            mlflow.sklearn.log_model(
                model,
                artifact_path="random_forest_regressor",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

    return model, metrics


def train_xgboost_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any] | None = None,
    log_to_mlflow: bool = True,
) -> tuple[XGBRegressor, dict[str, float]]:
    """Train an XGBoostRegressor model, evaluate metrics, and log to MLflow."""
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        xgb_params.update(params)

    model = XGBRegressor(**xgb_params)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = evaluate_predictions(y_test, predictions)
    logger.info("XGBoostRegressor evaluation metrics: %s", metrics)

    if log_to_mlflow:
        mlflow.log_params(xgb_params)
        mlflow.log_metrics(metrics)
        try:
            mlflow.xgboost.log_model(model, name="xgboost_regressor")
        except TypeError:
            mlflow.xgboost.log_model(model, artifact_path="xgboost_regressor")

    return model, metrics


def train_models(
    spark_session: SparkSession | None = None,
    catalog: str | None = None,
    schema: str | None = None,
    table_name: str = taxis.DEFAULT_TABLE_NAME,
    target_col: str = DEFAULT_TARGET_COL,
    feature_cols: list[str] | None = None,
    max_rows: int | None = 100_000,
    test_size: float = 0.2,
    random_state: int = 42,
    rf_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
    experiment_name: str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Train both RandomForestRegressor and XGBoostRegressor models with optional MLflow tracking."""
    spark = _get_spark(spark_session)
    full_table_name = taxis.get_full_table_name(table_name, catalog=catalog, schema=schema)

    logger.info("Loading taxi data from table: %s", full_table_name)
    df = taxis.find_all_taxis(
        spark_session=spark,
        catalog=catalog,
        schema=schema,
        table_name=table_name,
    )

    logger.info("Preparing feature dataset (target: %s, max_rows: %s)", target_col, max_rows)
    x, y = prepare_training_features(
        data=df,
        feature_cols=feature_cols,
        target_col=target_col,
        max_rows=max_rows,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    logger.info("Split dataset into %d training and %d test samples", len(x_train), len(x_test))

    results: dict[str, Any] = {}

    if not log_to_mlflow:
        rf_model, rf_metrics = train_random_forest_regressor(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            params=rf_params,
            log_to_mlflow=False,
        )
        xgb_model, xgb_metrics = train_xgboost_regressor(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            params=xgb_params,
            log_to_mlflow=False,
        )
        results["random_forest"] = {"model": rf_model, "metrics": rf_metrics}
        results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics}
        return results

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="nyc_taxi_models_comparison") as parent_run:
        mlflow.set_tag("dataset_table", full_table_name)
        mlflow.set_tag("target_column", target_col)
        mlflow.log_param("training_samples", len(x_train))
        mlflow.log_param("test_samples", len(x_test))
        mlflow.log_param("features", list(x.columns))

        # Train Random Forest Regressor in nested run
        with mlflow.start_run(run_name="RandomForestRegressor", nested=True):
            rf_model, rf_metrics = train_random_forest_regressor(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                params=rf_params,
                log_to_mlflow=True,
            )
            results["random_forest"] = {"model": rf_model, "metrics": rf_metrics}

        # Train XGBoost Regressor in nested run
        with mlflow.start_run(run_name="XGBoostRegressor", nested=True):
            xgb_model, xgb_metrics = train_xgboost_regressor(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                params=xgb_params,
                log_to_mlflow=True,
            )
            results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics}

        # Log comparison summary metrics to parent run
        mlflow.log_metrics({
            "rf_rmse": rf_metrics["rmse"],
            "rf_r2": rf_metrics["r2"],
            "xgb_rmse": xgb_metrics["rmse"],
            "xgb_r2": xgb_metrics["r2"],
        })
        results["parent_run_id"] = parent_run.info.run_id

    return results


def main() -> None:
    """Databricks job entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Databricks job to train RandomForestRegressor and XGBoostRegressor models",
    )
    parser.add_argument("--catalog", required=True, help="Databricks catalog name")
    parser.add_argument("--schema", required=True, help="Databricks schema name")
    parser.add_argument(
        "--table",
        default=taxis.DEFAULT_TABLE_NAME,
        help="Input table name (default: raw_taxis)",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help="Target column for regression (default: fare_amount)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Maximum rows to sample for training (default: 100000, 0 for all)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    spark = _get_spark()
    spark.sql(f"USE CATALOG `{args.catalog}`")
    spark.sql(f"USE SCHEMA `{args.schema}`")

    max_rows = None if args.max_rows == 0 else args.max_rows
    train_models(
        spark_session=spark,
        catalog=args.catalog,
        schema=args.schema,
        table_name=args.table,
        target_col=args.target_col,
        max_rows=max_rows,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
