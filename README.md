# Entity resolution with state-of-the-art SVM Classifier using PySpark.ml

## Context
Entity resolution is a very complex process that consists of finding
and matching entries in data sets that represents the same instance of an entity.

As we discovered through literature it turned out that this is an area heavily
researched with new approaches still being developed. 

After reading <cite>Köpcke, Hanna & Thor, Andreas & Rahm, Erhard. (2010). 
Evaluation of entity resolution approaches on real-world match problems. PVLDB. 3. 484-493</cite>
that gives a comprehensive evaluation of the performance of different entity resolution
technics. It appeared to us the **Insert machine learning lingo here** was
the best option in our case. The non ML approach although gave good results
were proprietary. Which we believe made the task harder to research.

## Our Approach
Our approach was comprised the following 3 steps:
1. Source Normalization
2. Blocking
3. Match scoring

Which is the main method used today for entity resolution regardless of the final method used

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
too many null values to be used for the blocking step.

We cleaned up the brand column to only contain the brand name, if the entry was null the brand
name was taken from the title column. Similar trick was used for the `cpu_model` column infering values when
null from the `cpu_brand` column. The `cpu_model` column was also trimmed to only include entries refering to
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
What made this data set tricking is the fact that the names were written in different
languages which thus affected the price and size units. We believed that working
on the languages of the name would have taken too much time to implement and thus we designed a simpler data cleaning approach.

The size were uniformized to `GB` and converted to double. Some null entries were inferred from the name column.
In order to maximize the match scoring process and avoid the size entries from being used outside the newly cleaned column
we removed that information when present in the name column.

## Blocking

### Specific Approach for the X2, X3 datasets
**Insert extra information about this process here**
For the blocking we decided to go the with the column tokenization of the 
`title, brand, cpu_brand, cpu_model, cpu_type, cpu_frequency, ram_capacity` columns. The vocabulary used in the title
was analyzed using LDA and IDF models. The blocking keys were then extracted from there.

### Specific Approach for the X4 datasets
Given the simplicity of the dataset as compared with `X2 and X3` and the fact that the name column
contained entries written in a different language we performed the blocking on tokenization of the 
`brand` and `size` column

## Match Scoring
Here is where my journey ends as I have very limited ML experience :(





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



