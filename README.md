# Entity resolution with state-of-the-art SVM Classifier using PySpark.ml

## How to COMMIT after working on the NOTEBOOK
GIT has some issues with jupyter when it comes to cell ouputs.
Before commiting your code do the following:

### Pycharm
Go to the **jupyter** icon at the bottom left of your screen and click on the URLs with
your access token passed as a GET argument that should open the notebook in your browser

### Then
1. Open the notebook that you have edited 
2. Click on **CELL**
3. Click on **ALL OUTPUT**
4. Click on **CLEAR**
5. Repeat for all notebooks

You can now commit your code :)

## 0 - Data cleaning:
see blocking tentative part 0 
Simple .groupby('brand').

## 2 - Tokenization and Embedding:
First thing to do is apply a simple Tokenizer and stopWordsRemover to the 'title' column, 
by splitting the sentence into words and removing noisy words (i.e 'Amazon.com : ').
After that a good approach in litterature seems to be applying TF-IDF measure to the tokens.
What it does, is basically mapping each set of tokens to a feature vector where each token is 
mapped to its relative importance in the dataframe. Important: dataframe is one single group. 
(i.e less frequent words are more important. Moreover the feature is normalized with the number of tokens for each set).
TO DO: think how to apply the tokenization and embedding to the other columns.

