#!/bin/sh

touch ./output.csv
echo "left_instance_id,right_instance_id" > ./output.csv
cat ./X*.output/*.csv >> ./output.csv