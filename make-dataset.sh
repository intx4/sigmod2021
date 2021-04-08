#!/bin/bash

spark-submit --packages graphframes:graphframes:0.8.1-spark3.0-s_2.12 generate_train_dataset.py $1
