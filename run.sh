#!/bin/bash

set -e

pip install -r requirements.txt

for file in X*.csv; do
	spark-submit --packages graphframes:graphframes:0.8.1-spark3.0-s_2.12 generate_matches.py $file
done

