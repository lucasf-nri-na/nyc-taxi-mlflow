from pyspark import pipelines as dp
from pyspark.sql.functions import col

# This file defines a sample transformation.
# Edit the sample below or add new transformations
# using "+ Add" in the file browser.


@dp.table
def sample_trips_nyc_taxi_mlflow():
    df = spark.read.table("raw_taxis")
    if "pickup_zip" not in df.columns and "PULocationID" in df.columns:
        df = df.withColumn("pickup_zip", col("PULocationID"))
    return df
