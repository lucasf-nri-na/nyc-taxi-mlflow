from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    coalesce,
    col,
    date_format,
    dayofweek,
    hour,
    lit,
    to_date,
    trim,
    upper,
    when,
)

# This file defines a transformation to prep and clean data from raw_taxis.


def clean_and_prep_taxis(df: DataFrame) -> DataFrame:
    """Clean and prepare raw NYC taxi trip records.

    Transformations applied:
    - Filters out rows with null or non-positive pickup/dropoff datetimes where dropoff is not after pickup
    - Filters out invalid passenger counts (must be > 0), trip distances (must be > 0), and fare/total amounts (> 0)
    - Validates TLC taxi zone pickup and dropoff location IDs (must be non-null and > 0)
    - Calculates trip duration in minutes and filters to realistic trips (1 to 1440 minutes)
    - Extracts calendar and time features: pickup_date, pickup_hour, pickup_day_of_week, pickup_day_name
    - Derives dropoff_date and dropoff_hour
    - Maps PULocationID to pickup_zip and DOLocationID to dropoff_zip for zone compatibility
    - Fills null surcharges and fee amounts with 0.0
    - Normalizes store_and_fwd_flag to uppercase or defaults to 'N' if null
    """
    return (
        df.filter(
            col("tpep_pickup_datetime").isNotNull()
            & col("tpep_dropoff_datetime").isNotNull()
            & (col("tpep_dropoff_datetime") > col("tpep_pickup_datetime"))
            & col("passenger_count").isNotNull()
            & (col("passenger_count") > 0)
            & col("trip_distance").isNotNull()
            & (col("trip_distance") > 0.0)
            & col("fare_amount").isNotNull()
            & (col("fare_amount") > 0.0)
            & col("total_amount").isNotNull()
            & (col("total_amount") > 0.0)
            & col("PULocationID").isNotNull()
            & col("DOLocationID").isNotNull()
            & (col("PULocationID") > 0)
            & (col("DOLocationID") > 0)
        )
        .withColumn(
            "trip_duration_minutes",
            (col("tpep_dropoff_datetime").cast("long") - col("tpep_pickup_datetime").cast("long")) / 60.0,
        )
        .filter(
            (col("trip_duration_minutes") >= 1.0) & (col("trip_duration_minutes") <= 1440.0)
        )
        .withColumn("pickup_date", to_date(col("tpep_pickup_datetime")))
        .withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
        .withColumn("pickup_day_of_week", dayofweek(col("tpep_pickup_datetime")))
        .withColumn("pickup_day_name", date_format(col("tpep_pickup_datetime"), "EEEE"))
        .withColumn("dropoff_date", to_date(col("tpep_dropoff_datetime")))
        .withColumn("dropoff_hour", hour(col("tpep_dropoff_datetime")))
        .withColumn("pickup_zip", col("PULocationID"))
        .withColumn("dropoff_zip", col("DOLocationID"))
        .withColumn("extra", coalesce(col("extra"), lit(0.0)))
        .withColumn("mta_tax", coalesce(col("mta_tax"), lit(0.0)))
        .withColumn("tip_amount", coalesce(col("tip_amount"), lit(0.0)))
        .withColumn("tolls_amount", coalesce(col("tolls_amount"), lit(0.0)))
        .withColumn("improvement_surcharge", coalesce(col("improvement_surcharge"), lit(0.0)))
        .withColumn("congestion_surcharge", coalesce(col("congestion_surcharge"), lit(0.0)))
        .withColumn("Airport_fee", coalesce(col("Airport_fee"), lit(0.0)))
        .withColumn("cbd_congestion_fee", coalesce(col("cbd_congestion_fee"), lit(0.0)))
        .withColumn(
            "store_and_fwd_flag",
            when(col("store_and_fwd_flag").isNull(), "N").otherwise(trim(upper(col("store_and_fwd_flag")))),
        )
    )


@dp.table(
    comment="Cleaned and prepared NYC yellow taxi trip data from raw_taxis",
)
def prep_raw_taxis() -> DataFrame:
    """Read raw taxi records from raw_taxis and apply cleaning and preparation transformations."""
    df = spark.read.table("raw_taxis")
    return clean_and_prep_taxis(df)
