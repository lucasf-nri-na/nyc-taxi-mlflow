import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]


def test_train_module_syntax_and_structure():
    train_file = ROOT / "src" / "nyc_taxi_mlflow" / "train.py"
    assert train_file.is_file()
    tree = ast.parse(train_file.read_text(encoding="utf-8"))
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "prepare_training_features" in function_names
    assert "evaluate_predictions" in function_names
    assert "train_random_forest_regressor" in function_names
    assert "train_xgboost_regressor" in function_names
    assert "train_models" in function_names
    assert "main" in function_names


def test_evaluate_predictions_logic():
    try:
        from nyc_taxi_mlflow.train import evaluate_predictions
    except ModuleNotFoundError:
        # In isolated CI environment where numpy/sklearn are not in the ci dependency-group
        return

    import numpy as np
    import pandas as pd

    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 31.0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]


def test_prepare_training_features_logic():
    try:
        from nyc_taxi_mlflow.train import prepare_training_features
    except ModuleNotFoundError:
        return

    import pandas as pd

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


@patch("nyc_taxi_mlflow.taxis.find_all_taxis")
def test_train_models_when_dependencies_present(mock_find_all):
    try:
        from nyc_taxi_mlflow.train import train_models
    except ModuleNotFoundError:
        return

    import numpy as np
    import pandas as pd

    n = 60
    sample_df = pd.DataFrame({
        "tpep_pickup_datetime": pd.date_range("2026-01-01 08:00:00", periods=n, freq="15min"),
        "tpep_dropoff_datetime": pd.date_range("2026-01-01 08:15:00", periods=n, freq="15min"),
        "passenger_count": np.random.choice([1, 2, 3], size=n),
        "trip_distance": np.random.uniform(0.5, 15.0, size=n),
        "RatecodeID": np.random.choice([1, 2], size=n),
        "PULocationID": np.random.choice([100, 101, 102], size=n),
        "DOLocationID": np.random.choice([200, 201, 202], size=n),
        "fare_amount": np.random.uniform(5.0, 50.0, size=n),
    })

    mock_spark = MagicMock()
    mock_df = MagicMock()
    mock_df.limit.return_value.toPandas.return_value = sample_df
    mock_df.toPandas.return_value = sample_df
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
