# Entity resolution with state-of-the-art SVM Classifier using PySpark.ml

## 0 - Data cleaning:
* Col 'brand': for null values, take keyword from title (i.e 'HP','Asus')
* Col 'cpu...': collapse cpu columns in brand (i.e 'Intel' or 'AMD'...)
* Col 'ram...': keep capacity
* Col 'hdd and ssd': keep capacity. Collapse into one.
* Col 'weight': keep it but use conversion in kg
* Col 'dimensions': same as weight

## 1 - Blocking:
Simple .groupby('brand').

## 2 - Tokenization and Embedding:
First thing to do is apply a simple Tokenizer and stopWordsRemover to the 'title' column, 
by splitting the sentence into words and removing noisy words (i.e 'Amazon.com : ').
After that a good approach in litterature seems to be applying TF-IDF measure to the tokens.
What it does, is basically mapping each set of tokens to a feature vector where each token is 
mapped to its relative importance in the dataframe. Important: dataframe is one single group. 
(i.e less frequent words are more important. Moreover the feature is normalized with the number of tokens for each set).
TO DO: think how to apply the tokenization and embedding to the other columns.

