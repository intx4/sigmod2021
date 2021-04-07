from datasets import *
from blocking import *
from similarity import with_encodings, compute_similarities
from graphframes import GraphFrame
from training import compute_weights, train_model

blocking_columns = ["title", "brand", "cpu_type", "ram_type"]

df = read_notebooks("data/X2.csv")
df = blocking_keys(df, blocking_columns)
df = with_encodings(df, ["title", "cpu_type", "ram_type"])
pairs = candidate_pairs(df)

labels = read_matching_labels("data/Y2.csv")
matching_labels = labels.filter((labels.src < labels.dst) & (labels.label == 1))
missing = matching_labels.join(pairs, ["src", "dst"], how="left_anti")
print("Missing blocked pairs: ", missing.count())

sim_columns = [
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

g = GraphFrame(df, labels)
dataset = compute_similarities(g, sim_columns)
dataset = dataset.select("features", "edge.label")
dataset.write.mode("overwrite").parquet("dataset.parquet")
