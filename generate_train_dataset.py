import sys
from os import path

from datasets import *
from blocking import *
from similarity import with_encodings, compute_similarities
from graphframes import GraphFrame



instance_file = sys.argv[1] if len(sys.argv) > 1 else "./data/X4.csv"
label_file = instance_file.replace("X", "Y")
dataset_name = path.splitext(path.basename(instance_file))[0]

dataset = read_dataset(instance_file)
df = dataset.df
df = blocking_keys(df, dataset.blocking_columns)
df = with_encodings(df, dataset.encoding_columns)
pairs = candidate_pairs(df)

labels = read_matching_labels(label_file)
matching_labels = labels.filter((labels.src < labels.dst) & (labels.label == 1))
missing = matching_labels.join(pairs, ["src", "dst"], how="left_anti")
print("Missing blocked pairs: ", missing.count())

g = GraphFrame(df, labels)
df = compute_similarities(g, dataset.sim_columns)
df = df.select("features", "edge.label")
df.write.mode("overwrite").parquet(dataset_name + ".parquet")
