#!/bin/bash

spark-submit --packages graphframes:graphframes:0.8.1-spark3.0-s_2.12 generate_matches.py

echo "left_instance_id,right_instance_id" > output.csv
cat output/*.csv >> output.csv
rm -rf output
