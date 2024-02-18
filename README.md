# Introduction
Many of you may have encountered, or even applied, decision trees in your projects. This algorithm stands out as one of the simplest yet surprisingly effective tools in the machine learningfor several reasons:
* Decision trees do not assume linear relationships among features, classifying them as 'nonparametric models.' This characteristic is particularly relevant as we all know that real-world relationships rarely conform perfectly to linearity.
* They bypass the need for feature centering or scaling, significantly reducing the time for data preparation.
* However, decision trees can sometimes overfit or struggle with balancing bias and variance. To further explain, overly simplistic trees may exhibit high bias, while overly complex trees can suffer from high variance, both of which compromise the model's performance on unseen datasets. Yet, when decision trees are enhanced with techniques like bagging and boosting, they form some of the most robust algorithms in machine learning, such as Random Forest. This approach combines multiple decision trees through bagging and bootstrapping (random sampling with replacement) to improve prediction accuracy. Before diving deeper into Random Forest or other ensemble methods, gaining a solid understanding of decision trees is crucial. This foundation will make you navigate the complexities of ensemble methods much more effectively.

This blog post kicks is the first of my series dedicated to exploring decision trees. As we begin, our initial goal is to establish a strong foundation by delving into the core elements of the algorithm. To do so, we will explore [a banking dataset from Kaggle](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets?select=test.csv) as our starting point. I strongly encourage you to examine the dataset in detail, noting the number and types of features, the dataset's purpose, etc., to better acquaint yourself with the data. Although the dataset is relatively straightforward and lacks missing values, which does unfortunately not mirror the complexity of real-world data, it serves as an excellent primer. In brief, the dataset centers on a marketing campaign by a Portuguese bank that uses phone calls to encourage clients to subscribe to a term deposit. Our objective is to predict whether a client will agree to subscribe. Kaggle provides both a training set and a testing set for this purpose, and we'll begin our exploration with the training set.

# Data Preparation 
We'll begin by importing the dataset into our Jupyter notebook environment.

```ruby
import pandas as pd
import numpy as np

# Define the file path
file_path = 'indicate your file path here'

import pandas as pd

# Read the CSV file into a pandas DataFrame, specifying the correct delimiter
df = pd.read_csv(file_path, delimiter=';')

```
Let's explore descriptive statistics of the features: 
```ruby
summary_stats = df.describe(include='all')

print(summary_stats)
```
<img width="710" alt="Screen Shot 2024-02-18 at 3 30 05 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/4f818cfe-c758-4bac-a002-62db46788c62">


<img width="624" alt="Screen Shot 2024-02-18 at 3 30 12 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/d1e3a874-610b-40ef-b137-cfb82ba12f00">

<img width="589" alt="Screen Shot 2024-02-18 at 3 30 18 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/815fbd03-97a6-4ae5-95d9-a6676d2b15e6">

Accoridng to the output, you can see that we have both categorical and numerical features. For categorical features, such as job, marital status, and eduacation, 'unique' indicates the number of unique values in those columns (i.e., how many distinct categories are present).'top' indictates mode or which category appears the most in those categorical columns, and 'freq' represents the frequency of the top value in each categorical column. For numerical features like age and balance, the describe() function provides other statistics such as count, mean, std, min, max, and the quartiles (25%, 50%, 75%).While the dataset is relatively clean, it's important to verify that all features are represented in their intended formats.

```ruby
# Get a summary of information about the DataFrame including the type of variables
print(df.info())

# Get the first few rows to confirm it looks correct
print(df.head())
```

<img width="620" alt="Screen Shot 2024-02-18 at 3 38 16 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/c11aace4-05f8-4123-9b92-a20fdbc50c8f">

Upon inspection, you'll notice that certain features are classified as 'object' because they contain string values. To optimize these features for use with decision trees, we will convert them to categorical data.

```ruby
# Select columns to encode: all object dtype columns except 'y'
columns_to_encode = df.select_dtypes(include=['object']).columns.drop('y')

# Use pd.get_dummies() to one-hot encode the selected columns
df_encoded = pd.get_dummies(df, columns=columns_to_encode)
```
After applying the `pd.get_dummies()`, the features originally marked as 'object' will now be of type 'uint8'. In pandas, 'uint8' refers to an unsigned 8-bit integer, meaning it can only hold non-negative integers ranging from 0 to 255. This datatype is particularly efficient for storing categorical data that has been transformed via dummy encoding, as it conserves memory.

<img width="452" alt="Screen Shot 2024-02-18 at 3 40 59 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/c58f09e3-c123-4884-9c05-6540137edfab">

<img width="462" alt="Screen Shot 2024-02-18 at 3 41 19 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/7d081a00-5b3b-4644-a845-2e8e30964137">

<img width="712" alt="Screen Shot 2024-02-18 at 3 41 58 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/6d6c7bd8-c947-44dd-9730-1aefeaf346be">

