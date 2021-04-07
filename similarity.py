import tensorflow_hub as hub
import numpy as np

from functools import reduce
from operator import add

from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml.feature import VectorAssembler


"""Use universal sentence encoder from tensorflow_hub"""
MODEL = None


def get_model_magic():
    global MODEL
    if MODEL is None:
        MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return MODEL


@f.udf(returnType=VectorUDT())
def encode_sentence(x):
    model = get_model_magic()
    emb = model([x]).numpy()[0]
    return Vectors.dense(emb)


def with_encodings(df, columns):
    for c in columns:
        df = df.withColumn(
            c + "_encoding", encode_sentence(f.coalesce(f.col(c), f.lit("")))
        )
    return df


@f.udf(returnType=t.DoubleType())
def dot(x, y):
    if x is not None and y is not None:
        return float(x.dot(y))
    else:
        return 0


def vector_sim(c1, c2):
    return dot(f.col(c1), f.col(c2))


def levenshtein_sim(c1, c2):
    output = f.when(f.col(c1).isNull() | f.col(c2).isNull(), 0).otherwise(
        1 - f.levenshtein(c1, c2) / f.greatest(f.length(c1), f.length(c2))
    )
    return output


def scalar_sim(c1, c2):
    output = (
        f.when(f.col(c1).isNull() | f.col(c2).isNull(), 0)
        .when((f.col(c1) == 0) & (f.col(c2) == 0), 1)
        .when((f.col(c1) == 0) | (f.col(c2) == 0), 0)
        .otherwise(1 - f.abs(f.col(c1) - f.col(c2)) / f.greatest(c1, c2))
    )
    return output


def token_overlap(c1, c2):
    # is the overlap a significant part of the shorter string
    output = (
        f.when(f.col(c1).isNull() | f.col(c2).isNull(), 0)
        .when(
            (f.size(f.array_distinct(c1)) == 0) | (f.size(f.array_distinct(c2)) == 0), 0
        )
        .otherwise(
            f.size(f.array_intersect(c1, c2))
            / f.least(f.size(f.array_distinct(c1)), f.size(f.array_distinct(c1)))
        )
    )
    return output


def compute_similarities(graph, columns):
    type_dict = dict(graph.vertices.dtypes)
    sim_methods = {
        "string": levenshtein_sim,
        "array<string>": token_overlap,
        "vector": vector_sim,
        "double": scalar_sim,
    }
    df = graph.triplets
    for c in columns:
        sim = sim_methods[type_dict[c]]
        df = df.withColumn(c + "_sim", sim("src." + c, "dst." + c))

    metrics = [c + "_sim" for c in columns]
    df = df.withColumn(
        "overall_sim", reduce(add, [f.col(c) for c in metrics]) / len(metrics)
    )
    metrics.append("overall_sim")

    assembler = VectorAssembler(inputCols=metrics, outputCol="features")
    return assembler.transform(df)
