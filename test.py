import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as f

if len(sys.argv) < 2:
    sys.exit("Error: missing argument")

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(sys.argv[1], header=True)
split_col = f.split(df["brand"], " ")
df = df.withColumn("brand_name", split_col.getItem(0))
df.groupby("brand_name").count().show()
