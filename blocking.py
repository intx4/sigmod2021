from db import spark

from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql import Window as w
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA
from pyspark.ml.linalg import VectorUDT, Vector
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    Tokenizer,
    RegexTokenizer,
    CountVectorizer,
    StopWordsRemover,
    NGram,
    Normalizer,
    VectorAssembler,
    Word2Vec,
    Word2VecModel,
    PCA,
)

stopW = [
    "softwarecity",
    "amazon",
    "com",
    "pc",
    "windows",
    "computers",
    "computer",
    "accessories",
    "laptop",
    "notebook",
    "kg",
    "inch",
    "processor",
    "memory",
    "gb",
    "ram",
    "hdd",
    "ssd",
    "cpu",
    "display",
    "hz",
    "ghz",
    "tb",
    "rpm",
    "slot",
    "slots",
    "mhz",
    "cache",
    "ram",
    "ddram",
    "dram",
    "hd",
]


def tokenize(df, string_cols):
    """Returns the df with tokenized columns with stopwords removed"""

    @f.udf(returnType=t.ArrayType(t.StringType()))
    def filter_alnum(arr):
        return [t for t in arr if t.isalnum() and len(t) > 2]

    output = df
    for c in string_cols:
        output = output.withColumn("temp", f.coalesce(f.col(c), f.lower(c), f.lit("")))
        tokenizer = RegexTokenizer(
            inputCol="temp", outputCol=c + "_rawtokens", pattern="\\W"
        )
        remover = StopWordsRemover(
            inputCol=c + "_rawtokens", outputCol=c + "_tokens", stopWords=stopW
        )

        output = tokenizer.transform(output)
        output = remover.transform(output).drop(c + "_rawtokens")
        output = output.withColumn(
            c + "_tokens", f.array_distinct(filter_alnum(f.col(c + "_tokens")))
        )
    # output has c+tokens columns
    return output.drop("temp")


def top_keywords(vocab, n=3):
    @f.udf(returnType=t.ArrayType(t.StringType()))
    def _(arr):
        inds = arr.indices
        vals = arr.values
        top_inds = vals.argsort()[-n:][::-1]
        top_keys = inds[top_inds]
        output = []

        for k in top_keys:
            kw = vocab.value[k]
            output.append(kw)

        return output

    return _


def generate_blocking_keys(df, token_cols, min_freq=1):
    """Pipeline:
    1 - CountVectorizer -> TF
    2 - IDF
    3 - LDA
    """
    # merge all tokens in one column
    df = df.withColumn("tokens", f.array_distinct(f.concat(*token_cols)))
    df = df.drop(*token_cols)

    # Vectorize the tokens and find their inverse frequency
    cv = CountVectorizer(inputCol="tokens", outputCol="raw_features").fit(df)
    df = cv.transform(df)

    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=min_freq).fit(
        df
    )
    df = idf.transform(df)

    normalizer = Normalizer(p=2.0, inputCol="features", outputCol="tfidf")
    df = normalizer.transform(df).drop("features", "raw_features")

    k = df.select("brand").distinct().count()
    lda = LDA(k=k, maxIter=1000, featuresCol="tfidf").fit(df)
    vocab = cv.vocabulary

    # returns words for each topic term
    @f.udf(returnType=t.ArrayType(t.StringType()))
    def get_words(token_list):
        return [vocab[token_id] for token_id in token_list]

    # create list of topic keywords
    # i.e topic 1 -> acer, anspire, intel
    topics = (
        lda.describeTopics(3)
        .withColumn("topic_words", get_words(f.col("termIndices")))
        .collect()
    )
    list_of_topics = []
    for r in topics:
        topicW = r["topic_words"]
        for w in topicW:
            list_of_topics.append(w)

    # returns list of 3 'hashtags' i.e keywords for topic
    # from tokens: title, brand, cpu_brand
    @f.udf(returnType=t.ArrayType(t.StringType()))
    def get_key(words):
        l = [w for w in words if w in list_of_topics]
        l = list(set(l))
        l.sort()
        return l[:3]

    df = df.withColumn("blocking_keys", get_key(f.col("tokens")))
    return df


def with_top_tokens(df, token_cols, min_freq=1):
    for pre in token_cols:
        cv = CountVectorizer(
            inputCol=pre, outputCol=pre + "_raw_features", minDF=min_freq
        )
        idf = IDF(
            inputCol=pre + "_raw_features",
            outputCol=pre + "_features",
            minDocFreq=min_freq,
        )
        normalizer = Normalizer(
            p=2.0, inputCol=pre + "_features", outputCol=pre + "_tfidf"
        )
        stages = [cv, idf, normalizer]
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
        df = model.transform(df).drop(pre + "_raw_features", pre + "_features")

        vocab = spark.sparkContext.broadcast(model.stages[0].vocabulary)
        df = df.withColumn(
            pre + "_top", top_keywords(vocab, n=5)(f.col(pre + "_tfidf"))
        )
    return df


def blocking_keys(df, columns):
    df = tokenize(df, columns)
    token_cols = [c + "_tokens" for c in columns]
    df = generate_blocking_keys(df, token_cols)
    return df
    # top_token_cols = [c + "_tokens_top" for c in columns]
    # return df.withColumn("blocking_keys", f.array_distinct(f.concat(*top_token_cols)))


def candidate_pairs(df):
    LARGEST_BLOCK = 200
    df = df.withColumnRenamed("instance_id", "uid")
    keep_pairs = (
        df.select(f.explode("blocking_keys").alias("blocking_key"), "uid")
        .groupBy("blocking_key")
        .agg(
            f.count("uid").alias("block_size"),
            f.collect_set("uid").alias("uid"),
        )
        .filter(f.col("block_size").between(2, LARGEST_BLOCK))
        .select("blocking_key", f.explode("uid").alias("uid"))
    )

    left = keep_pairs.withColumnRenamed("uid", "lid")
    right = keep_pairs.withColumnRenamed("uid", "rid")

    return (
        left.join(right, ["blocking_key"], "inner")
        .filter(f.col("lid") < f.col("rid"))
        .select("lid", "rid")
        .distinct()
    )
