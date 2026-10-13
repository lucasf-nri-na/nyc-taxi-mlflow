from unittest.mock import MagicMock, patch

from nyc_taxi_mlflow import taxis
from nyc_taxi_mlflow.taxis import (
    DEFAULT_TABLE_NAME,
    TAXI_DATA_URL,
    get_full_table_name,
    is_table_populated,
    load_raw_taxis,
)


def test_taxi_data_url_and_table_constants():
    assert TAXI_DATA_URL == "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet"
    assert DEFAULT_TABLE_NAME == "raw_taxis"


def test_get_full_table_name():
    assert get_full_table_name("raw_taxis") == "`raw_taxis`"
    assert get_full_table_name("raw_taxis", catalog="cat", schema="sch") == "`cat`.`sch`.`raw_taxis`"
    assert get_full_table_name("raw_taxis", schema="sch") == "`sch`.`raw_taxis`"
    assert get_full_table_name("cat.sch.raw_taxis") == "`cat`.`sch`.`raw_taxis`"


def test_is_table_populated_when_table_has_records():
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = True
    mock_df = MagicMock()
    mock_df.limit.return_value.count.return_value = 100
    mock_spark.table.return_value = mock_df

    assert is_table_populated(mock_spark, "raw_taxis") is True
    mock_spark.catalog.tableExists.assert_called_once_with("raw_taxis")
    mock_spark.table.assert_called_once_with("raw_taxis")
    mock_df.limit.assert_called_once_with(1)


def test_is_table_populated_when_table_has_zero_records():
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = True
    mock_df = MagicMock()
    mock_df.limit.return_value.count.return_value = 0
    mock_spark.table.return_value = mock_df

    assert is_table_populated(mock_spark, "raw_taxis") is False


def test_is_table_populated_when_table_does_not_exist():
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = False
    mock_spark.table.side_effect = Exception("Table not found")

    assert is_table_populated(mock_spark, "raw_taxis") is False


@patch("nyc_taxi_mlflow.taxis.download_taxi_data")
@patch("nyc_taxi_mlflow.taxis._read_and_save_parquet")
def test_load_raw_taxis_skips_download_when_table_populated(mock_save, mock_download):
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = True
    mock_df = MagicMock()
    mock_df.limit.return_value.count.return_value = 50
    mock_spark.table.return_value = mock_df

    result = load_raw_taxis(spark_session=mock_spark, catalog="dev", schema="test", table_name="raw_taxis")

    mock_download.assert_not_called()
    mock_save.assert_not_called()
    mock_spark.read.table.assert_called_once_with("`dev`.`test`.`raw_taxis`")
    assert result == mock_spark.read.table.return_value


@patch("nyc_taxi_mlflow.taxis.download_taxi_data")
@patch("nyc_taxi_mlflow.taxis._read_and_save_parquet")
def test_load_raw_taxis_downloads_when_table_does_not_exist(mock_save, mock_download):
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = False
    mock_spark.table.side_effect = Exception("Table not found")

    result = load_raw_taxis(spark_session=mock_spark, catalog="dev", schema="test", table_name="raw_taxis")

    mock_download.assert_called_once()
    assert mock_download.call_args[0][0] == TAXI_DATA_URL
    mock_save.assert_called_once()
    assert mock_save.call_args[0][2] == "`dev`.`test`.`raw_taxis`"
    mock_spark.read.table.assert_called_once_with("`dev`.`test`.`raw_taxis`")
    assert result == mock_spark.read.table.return_value


@patch("nyc_taxi_mlflow.taxis.download_taxi_data")
@patch("nyc_taxi_mlflow.taxis._read_and_save_parquet")
def test_load_raw_taxis_downloads_when_table_has_zero_records(mock_save, mock_download):
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = True
    mock_df = MagicMock()
    mock_df.limit.return_value.count.return_value = 0
    mock_spark.table.return_value = mock_df

    result = load_raw_taxis(spark_session=mock_spark, catalog="dev", schema="test", table_name="raw_taxis")

    mock_download.assert_called_once()
    assert mock_download.call_args[0][0] == TAXI_DATA_URL
    mock_save.assert_called_once()
    mock_spark.read.table.assert_called_once_with("`dev`.`test`.`raw_taxis`")
    assert result == mock_spark.read.table.return_value


@patch("nyc_taxi_mlflow.taxis.load_raw_taxis")
def test_find_all_taxis_loads_if_unpopulated(mock_load):
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = False
    mock_spark.table.side_effect = Exception("Table not found")

    taxis.find_all_taxis(spark_session=mock_spark, catalog="dev", schema="test")

    mock_load.assert_called_once_with(
        spark_session=mock_spark,
        catalog="dev",
        schema="test",
        table_name="raw_taxis",
    )


def test_prep_raw_taxis_transformation_file_exists_and_defines_table():
    import ast
    from pathlib import Path

    prep_file = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "nyc_taxi_mlflow_etl"
        / "transformations"
        / "prep_raw_taxis.py"
    )
    assert prep_file.is_file(), f"Expected transformation file at {prep_file}"
    tree = ast.parse(prep_file.read_text(encoding="utf-8"))
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "prep_raw_taxis" in function_names
    assert "clean_and_prep_taxis" in function_names

