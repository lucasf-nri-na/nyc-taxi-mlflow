import argparse
import logging

from databricks.sdk.runtime import spark

from nyc_taxi_mlflow import taxis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Process command-line arguments
    parser = argparse.ArgumentParser(
        description="Databricks job with catalog and schema parameters",
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--table",
        default=taxis.DEFAULT_TABLE_NAME,
        help="Name of the raw table to load",
    )
    parser.add_argument(
        "--url",
        default=taxis.TAXI_DATA_URL,
        help="URL to download raw taxi data from",
    )
    args = parser.parse_args()

    # Set the default catalog and schema
    spark.sql(f"USE CATALOG `{args.catalog}`")
    spark.sql(f"USE SCHEMA `{args.schema}`")

    # Load raw taxi table (only downloads if table does not exist or has zero records)
    taxis.load_raw_taxis(
        spark_session=spark,
        catalog=args.catalog,
        schema=args.schema,
        table_name=args.table,
        url=args.url,
    )

    # Example: display sample records from the loaded table
    taxis.find_all_taxis(
        spark_session=spark,
        catalog=args.catalog,
        schema=args.schema,
        table_name=args.table,
    ).show(5)


if __name__ == "__main__":
    main()

