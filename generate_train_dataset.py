from datasets import *
from blocking import *
from similarity import with_encodings, compute_similarities
from graphframes import GraphFrame
from training import compute_weights, train_model

columns = ["title", "brand", "cpu"]

df = read_notebooks("data/X2.csv")
df = blocking_keys(df, columns)
df = with_encodings(df, columns)

labels = read_matching_labels("data/Y2.csv")
# missing = labels.join(pairs, ["lid", "rid"], how="left_anti")
# print("Missing blocked pairs: ", missing.count())

nodes = df.withColumnRenamed("instance_id", "id")
edges = labels.withColumnRenamed("lid", "src").withColumnRenamed("rid", "dst")

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
dataset = dataset.select("features", "edge.label")
dataset.write.mode("overwrite").parquet("dataset.parquet")
