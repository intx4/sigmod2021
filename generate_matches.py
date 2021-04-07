from pyspark.sql import functions as f
from pyspark.ml.tuning import CrossValidatorModel

from datasets import *
from blocking import *
from similarity import with_encodings, compute_similarities
from graphframes import GraphFrame
from training import compute_weights, train_model

columns = ["title", "brand", "cpu"]

df = read_notebooks("data/X2.csv")
df = blocking_keys(df, columns)
pairs = candidate_pairs(df)

# missing = labels.join(pairs, ["lid", "rid"], how="left_anti")
# print("Missing blocked pairs: ", missing.count())
df = with_encodings(df, columns)

nodes = df.withColumnRenamed("instance_id", "id")
edges = pairs.withColumnRenamed("lid", "src").withColumnRenamed("rid", "dst")

cols = [
    "title",
    "brand",
    "cpu",
    "weight",
    "tokens",
    "tfidf",
    "title_encoding",
    "brand_encoding",
    "cpu_encoding",
]

g = GraphFrame(nodes, edges)
dataset = compute_similarities(g, cols)
dataset = dataset.select(
    f.col("src.id").alias("left_instance_id"),
    f.col("dst.id").alias("right_instance_id"),
    "features",
)
model = CrossValidatorModel.load("model")

output = model.transform(dataset)
output = output.filter(output.prediction == 1).select(
    "left_instance_id", "right_instance_id"
)

output.write.mode("overwrite").csv("output")
