from pyspark.sql import functions as f
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def compute_weights(df, column="label"):
    w_zero = 1 / df.filter(f.col(column) == 0).count()
    w_one = 1 / df.filter(f.col(column) == 1).count()
    return df.withColumn("weights", f.when(f.col(column) == 0, w_zero).otherwise(w_one))


def train_model(df, max_iter=100):
    model = LinearSVC(
        featuresCol="features", labelCol="label", weightCol="weights", maxIter=1000
    )
    param_grid = (
        ParamGridBuilder().addGrid(model.regParam, [0.5, 0.4, 0.3, 0.2, 0.1]).build()
    )
    cvs = CrossValidator(
        estimator=model,
        estimatorParamMaps=param_grid,
        evaluator=BinaryClassificationEvaluator(),
        numFolds=4,
    )
    return cvs.fit(df)
