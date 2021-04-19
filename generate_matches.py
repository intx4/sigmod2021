import sys
from os import path

from pyspark.ml.classification import LinearSVCModel, GBTClassificationModel, LogisticRegressionModel
from pyspark.ml.linalg import Vectors, VectorUDT
from datasets import *
from blocking import *
from similarity import with_encodings, compute_similarities
from graphframes import GraphFrame

instance_file = sys.argv[1]
label_file = instance_file.replace("X", "Y")
dataset_name = path.splitext(path.basename(instance_file))[0]

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

#model_1 = LinearSVCModel.load("model-" + dataset.name + "_1")
#model_comb = LogisticRegressionModel.load("model-" + dataset.name + "_comb")
model_2 = GBTClassificationModel.load("model-" + dataset.name)


#output = model_1.transform(df).drop('rawPrediction').drop('probability')
output = model_2.transform(df)

"""
@f.udf(returnType=VectorUDT())
def toVec(p1,p2):
    l = [p1,p2]
    return Vectors.dense(l)
"""

#output = output.drop('features')
#output = output.withColumn('features', toVec(f.col('prediction_svc'), f.col('prediction_gbt')))
#output = model_comb.transform(output)
output = output.filter(output.prediction == 1).select(
    "left_instance_id", "right_instance_id"
)

output.write.mode("overwrite").csv(dataset_name + ".output")
