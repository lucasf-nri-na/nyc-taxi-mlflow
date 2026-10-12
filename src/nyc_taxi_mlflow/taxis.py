from __future__ import annotations

import logging
import os
import pathlib
import tempfile
import urllib.request
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

TAXI_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet"
DEFAULT_TABLE_NAME = "raw_taxis"


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


def get_full_table_name(
    table_name: str = DEFAULT_TABLE_NAME,
    catalog: str | None = None,
    schema: str | None = None,
) -> str:
    """Return fully-qualified table name with backtick escaping if catalog and schema are provided."""
    if "." in table_name:
        parts = [p.strip("`") for p in table_name.split(".")]
        return ".".join(f"`{p}`" for p in parts)
    if catalog and schema:
        return f"`{catalog.strip('`')}`.`{schema.strip('`')}`.`{table_name.strip('`')}`"
    if schema:
        return f"`{schema.strip('`')}`.`{table_name.strip('`')}`"
    return f"`{table_name.strip('`')}`"


def is_table_populated(spark: Any, table_name: str) -> bool:
    """Check if the table exists and contains at least one record.

    Returns True if the table exists and has > 0 rows.
    Returns False if the table does not exist or has 0 rows.
    """
    clean_name = table_name.replace("`", "")
    exists = False
    try:
        exists = bool(spark.catalog.tableExists(clean_name))
    except Exception:
        exists = False

    if not exists:
        try:
            spark.table(table_name)
            exists = True
        except Exception:
            return False

    try:
        return spark.table(table_name).limit(1).count() > 0
    except Exception:
        return False


def download_taxi_data(url: str, dest_path: str) -> None:
    """Download parquet file from url to dest_path."""
    logger.info("Downloading taxi data from %s to %s", url, dest_path)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "nyc-taxi-mlflow/1.0"},
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        chunk_size = 1024 * 1024  # 1 MB chunks
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)


def _read_and_save_parquet(
    spark: Any,
    parquet_path: str,
    full_table_name: str,
) -> None:
    """Read local parquet file and save as a Delta table."""
    file_uri = pathlib.Path(parquet_path).resolve().as_uri()
    try:
        df = spark.read.parquet(file_uri)
        df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)
    except Exception as e:
        logger.warning(
            "Direct Spark parquet read failed (%s). Falling back to PyArrow/Pandas createDataFrame.",
            e,
        )
        try:
            import pyarrow.parquet as pq

            arrow_table = pq.read_table(parquet_path)
            pdf = arrow_table.to_pandas()
            df = spark.createDataFrame(pdf)
            df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load parquet file into table {full_table_name}: {fallback_error}"
            ) from fallback_error


def load_raw_taxis(
    spark_session: SparkSession | None = None,
    catalog: str | None = None,
    schema: str | None = None,
    table_name: str = DEFAULT_TABLE_NAME,
    url: str = TAXI_DATA_URL,
) -> DataFrame:
    """Load raw taxi data table from URL.

    Only downloads if the table does not exist or has zero records.
    """
    spark = _get_spark(spark_session)
    full_table_name = get_full_table_name(table_name, catalog=catalog, schema=schema)

    if is_table_populated(spark, full_table_name):
        logger.info(
            "Table %s already exists and has records. Skipping download.",
            full_table_name,
        )
        return spark.read.table(full_table_name)

    logger.info(
        "Table %s does not exist or has zero records. Downloading from %s...",
        full_table_name,
        url,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_parquet = os.path.join(tmp_dir, "yellow_tripdata.parquet")
        download_taxi_data(url, temp_parquet)
        _read_and_save_parquet(spark, temp_parquet, full_table_name)

    return spark.read.table(full_table_name)


# Alias for load_raw_taxis
load_raw_table = load_raw_taxis


def find_all_taxis(
    spark_session: SparkSession | None = None,
    catalog: str | None = None,
    schema: str | None = None,
    table_name: str = DEFAULT_TABLE_NAME,
) -> DataFrame:
    """Find all taxi data. Ensures the raw table is loaded before querying."""
    spark = _get_spark(spark_session)
    full_table_name = get_full_table_name(table_name, catalog=catalog, schema=schema)
    if not is_table_populated(spark, full_table_name):
        load_raw_taxis(
            spark_session=spark,
            catalog=catalog,
            schema=schema,
            table_name=table_name,
        )
    return spark.read.table(full_table_name)

