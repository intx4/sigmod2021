import sys
from pyspark.sql import functions as f
from pyspark.sql import types as t
import tensorflow_hub as hub

from db import spark

"""
Job:
    1 - remove ssd capacity, dimensions
    2 - infer brand from title
    3 - infer cpu brand and type
    4 - uniform weights
"""


def lowercase(df):
    for c in df.columns:
        df = df.withColumn(c, f.lower(f.col(c)))
    return df


def normalize_nulls(df, column):
    # If empty string, manually reset to None
    return df.withColumn(
        column, f.when(f.col(column) == "", None).otherwise(f.col(column))
    )


def merge_columns(df, column_names, output):
    """
    merge ram columns and cpu columns into one for ram and one for cpu
    """
    df = df.withColumn(output, f.concat_ws(" ", *column_names))
    df = normalize_nulls(df, output)
    return df.drop(*column_names)


def extract_number(df, column, pattern):
    num = f.regexp_extract(column, pattern, 1).cast(t.DoubleType())
    return df.withColumn(column, f.when(num != 0.0, num).otherwise(None))


def clean_notebook_features(df):
    # Remove Amazon.com : prefix from title
    df = df.withColumn("title", f.regexp_replace("title", "amazon.com\s?:\s?", ""))

    df = df.withColumn(
        "hdd_capacity", f.regexp_extract("hdd_capacity", "(\d+\s?\wb)", 1)
    )
    cap = f.regexp_extract("hdd_capacity", "(\d+)", 1).cast(t.DoubleType())
    df = df.withColumn(
        "hdd_capacity", f.when(df.hdd_capacity.contains("t"), cap * 1000).otherwise(cap)
    )

    df = extract_number(df, "cpu_frequency", "(\d+(\.\d+)?)\s?ghz")
    df = extract_number(df, "ram_capacity", "(\d+)\s?gb")
    # Extract brand or infer from title
    df = df.withColumn("brand", f.regexp_extract("brand", "^(\w+)", 0))
    computer_brands = [
        "lenovo",
        "acer",
        "hp",
        "dell",
        "asus",
        "samsung",
        "huawei",
        "surface",
        "apple",
    ]
    computer_brands_pattern = "({})".format("|".join(computer_brands))
    df = df.withColumn(
        "brand",
        f.when(
            df.brand.isNull(),
            f.regexp_extract("title", computer_brands_pattern, 0),
        ).otherwise(df.brand),
    )
    # exctract cpu_brand and infer type if intel
    cpu_brands = ["intel", "apple", "amd", "nvidia", "arm"]
    cpu_pattern = "({})".format("|".join(cpu_brands))
    df = df.withColumn(
        "cpu_model", f.regexp_extract("cpu_model", "(i\d|pentium|celeron|a\d)", 0)
    )
    df = df.withColumn(
        "cpu_model",
        f.when(
            f.isnull(df.cpu_model),
            f.regexp_extract("cpu_brand", "(i\d|pentium|celeron|a\d)", 0),
        ).otherwise(df.cpu_model),
    )
    df = df.withColumn(
        "cpu_brand",
        f.when(
            f.regexp_extract("cpu_brand", cpu_pattern, 0) != "",
            f.regexp_extract("cpu_brand", cpu_pattern, 1),
        ).otherwise(f.regexp_extract("title", cpu_pattern, 0)),
    )
    # Extract weight and convert from kilos to pounds
    df = df.withColumn(
        "weight",
        f.when(
            df.weight.contains("pounds") | df.weight.contains("lbs"),
            f.regexp_extract("weight", "(\d+.?\d)", 0).cast(t.DoubleType()),
        ).otherwise(
            f.regexp_extract("weight", "(\d+.?\d)", 0).cast(t.DoubleType()) * 2.20462
        ),
    )
    df = normalize_nulls(df, "cpu_model")
    df = normalize_nulls(df, "cpu_brand")
    df = df.drop("ssd_capacity", "ram_frequency", "dimensions")
    return df


class NotebookDataset:
    def __init__(self, df):
        self.df = clean_notebook_features(df)
        self.blocking_columns = ["title", "brand", "cpu_type", "ram_type"]
        self.encoding_columns = ["title", "cpu_type", "ram_type"]
        self.sim_columns = [
            "title",
            "brand",
            "cpu_brand",
            "cpu_model",
            "cpu_type",
            "cpu_frequency",
            "ram_capacity",
            "hdd_capacity",
            "weight",
            "title_tokens",
            "brand_tokens",
            "cpu_type_tokens",
            "ram_type_tokens",
            "title_tokens_tfidf",
            "brand_tokens_tfidf",
            "cpu_type_tokens_tfidf",
            "ram_type_tokens_tfidf",
            "title_encoding",
            "cpu_type_encoding",
            "ram_type_encoding",
        ]
        self.name = "notebooks"


def clean_product_features(df):
    for c in df.columns:
        df = df.withColumn(c, f.lower(c))
    df = df.withColumn("price", df.price.cast(t.DoubleType()))
    df = df.withColumn(
        "size",
        f.when(
            df.size.isNull(), f.regexp_extract(df.name, "(\d+ ?(gb|tb|go|to))", 1)
        ).otherwise(df.size),
    )
    size = f.regexp_extract("size", "(\d+)", 1).cast(t.DoubleType())
    df = df.withColumn(
        "size",
        f.when(
            (df.size.contains("tb") | df.size.contains("to")), size * 1000
        ).otherwise(size),
    )
    df = df.withColumn("name", f.regexp_replace("name", "\d+ ?(gb|tb|go|tb)", ""))
    return df


class ProductDataset:
    def __init__(self, df):
        self.df = clean_product_features(df)
        self.blocking_columns = ["name","brand"]
        self.encoding_columns = ["name"]
        self.sim_columns = [
            "name",
            "name_tokens",
            "name_tokens_tfidf",
            "name_encoding",
            "brand",
            "brand_tokens",
            "brand_tokens_tfidf",
            "price",
            "size",
        ]
        self.name = "products"


def read_dataset(path):
    df = spark.read.csv(path, header=True)
    df = lowercase(df)
    df = df.withColumnRenamed("instance_id", "id")
    if "display_size" in df.columns:
        # We don't care about this dataset, so just die
        sys.exit(0)
    if "ram_type" in df.columns:
        return NotebookDataset(df)
    return ProductDataset(df)


def read_matching_labels(path):
    labels = spark.read.csv(path, header=True)
    labels = labels.withColumn("label", labels.label.cast(t.IntegerType()))
    labels = labels.withColumnRenamed("left_instance_id", "src").withColumnRenamed(
        "right_instance_id", "dst"
    )
    # Expand with reverse relations as well
    return labels.union(
        labels.select(f.col("src").alias("dst"), f.col("dst").alias("src"), "label")
    ).filter(labels.src != labels.dst)
