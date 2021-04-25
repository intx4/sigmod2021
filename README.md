# Entity resolution with (tentative of) state-of-the-art SVM Classifier using PySpark.ml
This work was developed in the context of ACM Sigmod Programming Contest 2021. Topic: Entity resolution on different datasets.

## Context
Entity resolution is a very complex process that consists of finding
and matching entries in data sets that represents the same instance of an entity.

As we discovered through literature it turned out that this is an area heavily
researched with new approaches still being developed. 

After reading <cite>Köpcke, Hanna & Thor, Andreas & Rahm, Erhard. (2010). 
Evaluation of entity resolution approaches on real-world match problems. PVLDB. 3. 484-493</cite>
that gives a comprehensive evaluation of the performance of different entity resolution
technics. It appeared to us the learning-based approach (specifically treating the entity resolution problem as a binary classification problem, where matching pairs have label 1, and non matching 0) was
the best option in our case (we were provided with a labelled dataset containing pairs, either matching or not matching). The non ML approach although gave good results
were proprietary, so we could not investigate further on that. 

## Our Approach
Our approach was comprised the following steps:
1. Source Normalization
2. Feature extraction & Blocking
3. Match scoring

Which is the main method used today for entity resolution regardless of the final method used (ML based or not).
We based our approach on this [guide](https://towardsdatascience.com/practical-guide-to-entity-resolution-part-1-f7893402ea7e)
Which describes each step using the `Pyspark` framework.

## Source Normalization
The goal of this step is to:
1. Normalize the data between different sources
2. Change/trim/calculate the data tailoring it for the blocking and matching steps.

### Specific Approach for the X2 and X3 datasets
The data sets were already normalized and represented different types of computers.
The data set consisted of the following columns:
```json
["title",
"brand",
"cpu_brand",
"cpu_model",
"cpu_type",
"cpu_frequency",
"ram_capacity",
"hdd_capacity",
"weight"]
```

we dropped the ssd_capacity, ram_frequency and dimensions columns since they were either too complex or had
too many null values to be used.

We cleaned up the brand column to only contain the brand name, if the entry was null the brand
name was inferred from the title column. Similar trick was used for the `cpu_model` column infering values when
null from the `cpu_brand` column (i.e if brand is 'intel' look for a regex matching 'i3' or 'i7' or 'pentium'). The `cpu_model` column was also trimmed to only include entries refering to
the broader version of the cpu such as `i3, i5, pentium, amd etc..`

The weights in pounds were converted to kilos.

### Specific Approach to the X4 dataset
This data set consisted of either flash memory cards or cellphones and only had the following columns
```json
["name",
  "brand",
  "price",
  "size"]
```
What made this data set tricky is the fact that the names were written in different
languages which thus affected the price and size units. We believed that working
on the languages of the name would have taken too much time to implement and thus we designed a simpler data cleaning approach.

The size were uniformized to `GB` and converted to double. Some null entries were inferred from the name column.
In order to maximize the match scoring process and avoid the size entries from being used outside the newly cleaned column
we removed that information when present in the name column.

## Feature exctraction & Blocking

### Specific Approach for the X2, X3 datasets
For this step we decided to go the with the column tokenization of the 
`title, brand, cpu_brand, cpu_model, cpu_type, cpu_frequency, ram_capacity` columns, followed by a 'StopWordsRemover' transformation. After that, we computed a TF-IDF matrix merging all tokens into a single vocabulary (using 'PySpark' 'CountVectorizer' and 'IDF' models)and we fed that to an LDA model for identifying topics (we decided to fix the number of topics to the number of different computer brands in the df). For each topic, we then extracted the top 3 tokens representing that token and we merged these 'hashtags' into a keywords list. For each tuple, we then extracted looked for 3 tokens in the keywords list. The blocking key was the resulting list.
This approach performed quite well, since we drammatically reduced the space of tuples where to perform the blocking without missing many matching pairs (i.e in X2 dataframe we missed only 1.6% of matching pairs).
For feature extraction, we decided to use the TF-IDF matrix computed for each column plus the encoding of the 'title' using the Universal Sentence Encoder model available on TensorFlow_Hub. 

### Specific Approach for the X4 datasets
Given the simplicity of the dataset as compared with `X2 and X3` and the fact that the name column
contained entries written in a different language we performed the blocking on tokenization of the 
`brand` and `size` column. This simple approach was very effective and led to no missed pairs.

For feature extraction, we once again used the TF-IDF matrix plus the encoding of the 'name'. This time, in order to tackle the multilinguism, we used the Multilingual Encoder of Tensorflow_Hub.

## Match Scoring
We then generated the pairs by combining the tuples in each block (i.e groupBy('blocking_key')). After we generated the pairs, we performed a join with the dataframe containing now all the features. We thus ended up with a df with the following schema: <left_id, right_id, [left_features], [right_features]>
We thus computed several distance metrics for each left and right features, and we transformed the dataset accordingly (for example, we computed the dot product between left and right title encodings, the levenshtein distance for string fields, the token overlap for tokenized columns and the scalar difference for numeric columns). We thus ended up with a bunch of numeric features and we thus dropped the fields we had before.
Finally, in order to achieve better classification results, we performed a simple feature augmentation using polynomial (degree = 2) and exponential expansion.
We repeated the same process for the provided labelled pairs.

## Classification task using SVM
For the classification task, we used a Support Vector Machine Model available in Pyspark.ml library. The classification task was hard to tackle due to a huge umbalance between the non matching pairs and the matching pairs. We adopted the following steps to try tackling this problem:
1. Added weights to the classes (w = 1 / df.filter('label == 0/1').count())
2. Used stratified split in order to have the same ratio between labels in train and test set.
3. Performed hyperparameter tuning with 4-fold cross validation on the decision threshold of the svm ( > 1).

## Post-Mortem analysis
The result we obtained with the SVM model were good but not optimal (however, it has to be said that ML pipeline we used was very general, exception made, of course, for the data cleaning part. The model can hence be reused for any other Entity Resolution task). We have decided to investigate the results of our model by having a look at the weight vector of the model. We focused on the specific model used for the notebooks.
The features of our data points were the following:

            "title",
            "brand",
            "cpu_brand",
            "cpu_model",
            "cpu_type",
            "cpu_frequency",
            "ram_capacity",
            "hdd_capacity",
            "weight",
            "title_tokens",
            "brand_tokens",
            "cpu_type_tokens",
            "ram_type_tokens",
            "title_tokens_tfidf",
            "brand_tokens_tfidf",
            "cpu_type_tokens_tfidf",
            "ram_type_tokens_tfidf",
            "title_encoding",
            "cpu_type_encoding",
            "ram_type_encoding",

In addition to this, we added the feature "overall_sim" (the average of the features), and, for each of the above features, we add the expantions `exp(x)` and `x**2`, for a total of 61 features.
The weight vector we obtained was `w = 
[0.7126, 0.8041, 0.524, -5.5832, -0.4214, -4.1963, -0.3278, 1.9426, 0.3475, 2.9603, 0.0573, -0.5129, -0.0519, 10.2059, 0.0573, 0.3862, 0.4672, -0.3382, 0.267, 0.4576, 0.1041,
-0.7921, -3.1105, -0.0668, 0.773, -0.3302, 0.5238, 0.1458, 6.6336, -0.2746, -0.8685, -0.0616, 4.0579, -0.0879, 0.4495, -0.1935, -1.6627, -0.2412, -0.1972, 0.0033, -1.4435,
-0.0894, 0.0573, -0.1751, -0.0621, -0.0796, 0.2234, 0.0918, -5.2068, -0.0894, 0.0573, -0.0371, 1.1518, -0.0203, -0.1666, -0.4202, -0.0132, -0.1705, 0.9001,-0.0924, -0.0297]`
The most weighted feature was the "title_tokens_tfidf" one.
We would have expected a higher weight for the features related to "title_encoding", while instead their weight was quite low.
It was quite unexpected to see a high weight also for the "hdd_capacity" feature, whose augmented feature ('exp(x)') is the second most weighted feature.
For the product datasets, the features were:
`           "name",
            "name_tokens",
            "name_tokens_tfidf",
            "name_encoding",
            "brand",
            "brand_tokens",
            "brand_tokens_tfidf",
            "price",
            "size",`
 The weight vector we obtained:
`[ 1.77538695  0.35490182  0.56183909  0.03115297  0.3713699   0.40684609 0.40684609  0.10292789  0.79313035  0.68824758
-2.17364067  1.80791245 -0.48987373  0.3946829  -0.65296184  0.48055447 -0.95034847  1.34965594 -0.12117104
 0.40020543  0.01391267  0.40684609  0.01391267  0.40684609 -0.22578841  0.20818782  0.15055509  2.52889901]`
 Here we see a much more uniform distribution in the weights of the feature. The most weighted feature is the quadratic expantion of the "size" feature.
