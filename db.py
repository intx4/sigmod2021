from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.config("spark.executor.memory", "4g")
    .config("spark.executor.cores", "2")
    .config("spark.cores.max", "2")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)
