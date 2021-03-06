#!/bin/bash

DATASET_NAME=$(basename $1)
DATASET_NAME="${DATASET_NAME/.csv/}"

spark-submit --packages graphframes:graphframes:0.8.1-spark3.0-s_2.12 generate_train_dataset.py $1
zip -r ${DATASET_NAME}.zip ${DATASET_NAME}.parquet
