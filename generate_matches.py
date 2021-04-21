import sys
from os import path

from pyspark.ml.classification import LinearSVCModel, GBTClassificationModel, LogisticRegressionModel
from pyspark.ml.linalg import Vectors, VectorUDT
from datasets import *
from blocking import *
from similarity import with_encodings, compute_similarities
from graphframes import GraphFrame

instance_files = []
instance_files.append(sys.argv[1])
instance_files.append(sys.argv[2])
instance_files.append(sys.argv[3])

model = {}
model["notebooks"] = GBTClassificationModel.load("model-notebooks")
model["products"] = LinearSVCModel.load("model-products")

for instance_file in instance_files:
    dataset = read_dataset(instance_file)
    df = dataset.df
    df = blocking_keys(df, dataset.blocking_columns)
    df = with_encodings(df, dataset.encoding_columns)
    pairs = candidate_pairs(df)

    g = GraphFrame(df, pairs)
    df = compute_similarities(g, dataset.sim_columns)
    df = df.select(
        f.col("src.id").alias("left_instance_id"),
        f.col("dst.id").alias("right_instance_id"),
        "features",
    )
    output = model[dataset.name].transform(df)

    output = output.filter(output.prediction == 1).select(
    "left_instance_id", "right_instance_id"
    )
    
    output.write.mode("overwrite").csv(dataset.name + ".output")
