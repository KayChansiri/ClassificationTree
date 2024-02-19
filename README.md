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

According to the output, you can see that we have both categorical and numerical features. For categorical features, such as job, marital status, and eduacation, 'unique' indicates the number of unique values in those columns (i.e., how many distinct categories are present).'top' indictates mode or which category appears the most in those categorical columns, and 'freq' represents the frequency of the top value in each categorical column. For numerical features like age and balance, the describe() function provides other statistics such as count, mean, std, min, max, and the quartiles (25%, 50%, 75%).While the dataset is relatively clean, it's important to verify that all features are represented in their intended formats.

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

In addition to all the predictive features, we must also ensure that our target outcome ('y') is encoded as categorical.

```ruby
df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})
```
# Decision Tree Elements 
Now that we've prepared the data, let's delve into the components of a decision tree. For simplicity, I won't use our dataset to fit and visualize a tree just yet. Instead, I've utilized the iris dataset, with which many of us are familiar from statistics classes, to help understand the elements of a decision tree. At this stage, there's no need to concern yourself with interpreting values such as the Gini index or the meanings of the numbers within each box. Instead, let's focus on familiarizing ourselves with the basic structure of a tree.

<img width="721" alt="Screen Shot 2024-02-18 at 3 55 34 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/d87ac48b-34fb-4e5d-aac5-04c5da5bb0ae">

* Root Node: This is the starting point of a decision tree. The root node is the top box, which, according to the figure above, splits based on the number of ‘sepal length <= 5.45’. It's where the first decision is made and splits into two or more branches.
* Parent Node: These are nodes that split into other nodes. Any box that has a subsequent split is a parent node. For example, the node with ‘sepal width’ <= 2.8’ (the node at the first level on the left as the root node is considered the zero level) is a parent node.
* Leaf Node or Terminal Node: These are the final nodes that do not split further and provide the outcome or classification. The boxes which have a 'class' label but no further branches are leaf nodes. They are the boxed at the last level representing the final decision or classification.
* Branch: These are subsections of the tree that split from either the root or parent nodes. Each condition leads to another decision, which could be a leaf or another decision node. In the figure above, the branches are the lines connecting nodes (i.e., boxes).
* Depth of a tree: The root noted is considered depth 0 and each split increases the depth of the tree by one. According to the figure, the depth is 2.

# Types of Decision Trees
Now that you have a basic understanding of what a decision tree is and its key elements, let's explore the first type of decision tree: the classification tree.

## Classification Trees
* This type of tree is designed to predict categorical outcomes, which can be either binary or multiclass. For simplicity, this post will focus on binary targets. 
* While classification trees predict categorical outcomes, the features used to develop the tree can be both categorical and continuous.
* The feature selected as the first node (i.e., root node) is not chosen randomly. Instead, the algorithm selects a feature from the dataset and evaluates how effectively it can categorize the target. This process is iterative, continuing until the algorithm identifies the optimal root node that results in the subsequent nodes being as pure as possible. For instance, in the iris example mentioned above, 'sepal length' is the root node because this feature better purifies the subsequent nodes compared to other features in the dataset.
* You might be wondering how the purity of a node is defined. Imagine selecting two features from your dataset and visualizing them in a 2D space where the features represent the X and Y axes, and the outcomes are data points within this space. The algorithm assesses each feature to find the optimal split point or 'decision boundary' along its axis. All splits are perpendicular to the axis, meaning decision trees have orthogonal decision boundaries.
* But what is 'decision boundary' really? This term refers to the boundary that demarcates the different classes within the feature space. Each region on one side of the boundary is predicted by the algorithm to belong to a specific class. In the most simplest scenario where your dataset comprises only two features, the decision boundary manifests as a line. With three features, it takes the form of a plane. In scenarios involving more than three features, it becomes a hyperplane. It's important to note that the concept of a decision boundary is not exclusive to decision trees but is prevalent across various algorithms. In linear regression models, the decision boundary is linear, delineating a straightforward separation between classes. Conversely, decision trees typically exhibit a non-linear decision boundary, which can be more complex, reflecting the algorithm's ability to capture more nuanced patterns within the data.

<img width="640" alt="Screen Shot 2024-02-19 at 9 10 03 AM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/515debf0-5bf8-4f1e-9b16-4e61d612cd98">

* At each point along each feature axis, the algorithm calculates impurity indices, including the Gini index or entropy. The value on the axis that yields the lowest Gini index or entropy, indicating whether a node is pure enough, is used as the split point. In essence, a node is considered pure enough when the samples or cases within it are more homogeneous than heterogeneous. The algorithm repeats this process with all features until it discovers the optimal decision boundary. Let's delve deeper into the concepts of Gini index and entropy for a clearer understanding of how these parameters influence tree growth.

## 




