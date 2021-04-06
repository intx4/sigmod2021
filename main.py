from datasets import *
from blocking import *

columns = ["title", "brand", "cpu"]

df = read_notebooks("data/X2.csv")
df = blocking_keys(df, columns)
pairs = candidate_pairs(df)

labels = read_matching_labels("data/Y2.csv")
missing = labels.join(pairs, ["lid", "rid"], how="left_anti")
print("Missing blocked pairs: ", missing.count())
