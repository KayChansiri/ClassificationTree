# Introduction

Welcome to the first post in my series about decision trees, one of the most intuitive yet profoundly impactful algorithms in the world of machine learning! If you've ever dabbled in machine learning projects, there's a good chance you've come across decision trees. You may have a question: Why are decision trees so compelling?

* **Beyond Linear Constraints**: Unlike many algorithms that seek linear paths through data, decision trees embrace its non-linear nature as nonparametric models. This adaptability is crucial, acknowledging the rarity of perfectly linear relationships in our real-world projects.
* **Preparation Made Simple**: Decision trees streamline the often tedious process of data preparation by eliminating the need for feature scaling or centering. The algorithm makes decisions by selecting features to split the data into subsets based on threshold values. These decisions are based on the order of the data points for a given feature, not on the absolute values. Whether a feature's values are scaled or centered does not change the order of the data points, and therefore does not affect the tree's decision-making process. For instance, whether a feature is measured in centimeters or inches would not change the decisions made by the tree, as it merely looks for the best splitting points regardless of the actual units or scale.
*  **Handling Heterogeneous Data**: Decision trees can handle features with different scales or units naturally, without requiring normalization or standardization. This capability is particularly useful when dealing with heterogeneous data, where different features might represent vastly different types of data (e.g., age in years and income in dollars).

Despite all the perks, decision trees can sometimes overfit or struggle with balancing bias and variance. To further explain, overly simplistic trees may exhibit high bias, while overly complex trees can suffer from high variance, both of which compromise the model's performance on unseen datasets. Yet, when decision trees are enhanced with ensemble techniques, they form some of the most robust algorithms in machine learning, such as Random Forest. This approach combines multiple decision trees through bagging and bootstrapping (random sampling with replacement) to improve prediction accuracy. Before diving deeper into Random Forest or other ensemble methods, gaining a solid understanding of decision trees is crucial. This foundation will make you navigate the complexities of ensembled machine learning algorithms much more effectively.

# Decision Tree Elements

To help simplify the elements of a decision tree, I've utilized the iris dataset, with which many of us are familiar from statistics classes. At this stage, there's no need to concern yourself with interpreting values such as the Gini index or the meanings of the numbers within each box. Instead, let's focus on familiarizing ourselves with the basic structure of a tree.
<img width="721" alt="Screen Shot 2024-02-18 at 3 55 34 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/d87ac48b-34fb-4e5d-aac5-04c5da5bb0ae">

* Root Node: This is the starting point of a decision tree. The root node is the top box, which, according to the figure above, splits based on the number of ‘sepal length <= 5.45’. It's where the first decision is made and splits into two or more branches.
* Parent Node: These are nodes that split into other nodes. Any box that has a subsequent split is a parent node, and the subsequent node is a child node. For example, the node with ‘sepal width’ <= 2.8’ (the node at the first level on the left as the root node is considered the zero level) is a parent node.
* Leaf Node or Terminal Node: These are the final nodes that do not split further and provide the outcome or classification. The boxes which have a 'class' label but no further branches are leaf nodes. They are the boxed at the last level representing the final decision or classification.
* Branch: These are subsections of the tree that split from either the root or parent nodes. Each condition leads to another decision, which could be a leaf or another decision node. In the figure above, the branches are the lines connecting nodes (i.e., boxes).
* Depth of a tree: The root noted is considered depth 0 and each split increases the depth of the tree by one. According to the figure, the depth is 2.

# Types of Decision Trees
Now that you have a basic understanding of what a decision tree is and its key elements, let's explore the first type of decision tree: the classification tree.

## 1. Classification Trees
* This type of tree is designed to predict categorical outcomes, which can be either binary or multiclass. For simplicity, this post will focus on binary targets. 
* While classification trees predict categorical outcomes, the features used to develop the tree can be both categorical and continuous.
* The feature selected as the first node (i.e., root node) is not chosen randomly. Instead, the algorithm selects a feature from the dataset and evaluates how effectively it can categorize the target. This process is iterative, continuing until the algorithm identifies the optimal root node that results in the subsequent nodes being as pure as possible. For instance, in the iris example mentioned above, 'sepal length' is the root node because this feature better purifies the subsequent nodes compared to other features in the dataset.
* You might be wondering how the purity of a node is defined. Imagine selecting two features from your dataset and visualizing them in a 2D space where the features represent the X and Y axes, and the outcomes are data points within this space. The algorithm assesses each feature to find the optimal split point or 'decision boundary' along its axis. All splits are perpendicular to the axis, meaning decision trees have orthogonal decision boundaries.
* But what is 'decision boundary' really? This term refers to the boundary that demarcates the different classes within the feature space. Each region on one side of the boundary is predicted by the algorithm to belong to a specific class. In the most simplest scenario where your dataset comprises only two features, the decision boundary manifests as a line. With three features, it takes the form of a plane. In scenarios involving more than three features, it becomes a hyperplane. It's important to note that the concept of a decision boundary is not exclusive to decision trees but is prevalent across various algorithms. In linear regression models, the decision boundary is linear, delineating a straightforward separation between classes. Conversely, decision trees typically exhibit a non-linear decision boundary, which can be more complex, reflecting the algorithm's ability to capture more nuanced patterns within the data.

<img width="640" alt="Screen Shot 2024-02-19 at 9 10 03 AM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/515debf0-5bf8-4f1e-9b16-4e61d612cd98">

* According to the decision boundary plot created using the Iris dataset above, the areas filled in yellow, purple, and blue represent the *decision regions* as determined by the tree. These regions indicate where the algorithm predicts a specific class (i.e., Setosa, Versicolor, Virginica) for any given point based on the input features (i.e., sepal length and sepal width). The lines separating each of the three colored regions constitute the *decision boundaries*, which demarcate where the model's predictions shift from one flower class to another. Ideally, each decision region should correspond to a single class. However, as observed, the real world rarely aligns perfectly with our models, especially when constructing a decision tree with only a few features. Within the yellow decision region, despite the predominance of Versicolor points, there are several blue dots. These represent instances where the model has misclassified Virginica as Versicolor.
* At each point along each feature axis, the algorithm calculates impurity indices, including *Gini index* or *entropy*. The value on the axis that yields the lowest Gini index or entropy, which indicate whether a node is pure enough, is used as the split point. In essence, a node is considered pure enough when the samples within it are more homogeneous than heterogeneous. The algorithm repeats this process with all features until it discovers the optimal decision boundary. Let's delve deeper into the concepts of Gini index and entropy for a clearer understanding of how these parameters influence classification tree growth.

### Gini Index

* Gini Index is a metric used to determine whether each node of a decision tree is sufficiently pure, containing homogeneous rather than heterogeneous samples. It calculates the probability of a randomly selected sample from a node being incorrectly classified. The closer the Gini index is to zero, the higher the purity of the leaf node, with a lower Gini index indicating greater purity.
* If the split points of a feature result in a high Gini index, the tree might adjust these points or opt for new features to construct an alternative tree that reduces impurity.
* Here's the formula for the Gini Index for a specific node:


<img width="274" alt="Screen Shot 2024-02-19 at 9 38 40 AM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/02cdfad7-d759-4879-8697-38d99cb4d00c">

* *m* is the number of classes. For the iris example, we have three classes. Thus m = 3.
* *P<sub>i</sub>* is the proportion of class *i* within the node. For example, in the decision tree of the Iris dataset below, at the first-level node displayed within the orange box, the Gini index is calculated as $1 - [({45 \over 52})^2 + ({6 \over 52})^2 + ({1 \over 52})^2] = 0.2147$

<img width="772" alt="Screen Shot 2024-02-19 at 9 56 28 AM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/c048947c-5dbd-46a1-9038-3aafe0807fc6">

* The Gini index value of 0.2147 indicates that the node exhibits a certain level of purity, predominantly classifying cases as Setosa. However, being the result of the first split, this value is not optimal. For binary targets, the Gini index ranges from 0 (complete purity) to 0.5 (maximum impurity), which means a value of 0.2147 isn't deemed low enough to consider this node purely homogeneous. In the context of multiclassification targets, the Gini index can reach up to 1, accommodating a broader spectrum of impurity levels.


### Entropy

* Entropy is a concept borrowed from information theory, signifying the measure of uncertainty or unpredictability in the information content. It helps quantify how much information there is in an event's outcome.
* Consider a scenario where a node is associated with three possible classes, each with a distinct probability of occurrence. Although this situation implies the leaf is not perfectly pure, it raises an important question: is it beneficial to grow a tree with such a leaf, or is it better not to grow the tree at all? This is where entropy comes into play.
* Similar to the Gini index, entropy serves as a criterion for determining where to make splits in a decision tree. Both metrics aim to diminish impurity with each subsequent split, with lower values indicating a higher degree of node purity. While some folks lean towards the Gini index due to its marginally lower computational demands, making it quicker to calculate, others prefer entropy for its foundational ties to information theory.
*  For Python users, the `DecisionTreeClassifier` defaults to using the Gini index. However, you can opt for entropy by adjusting the `criterion` hyperparameter. I would recommended to experiment with both criteria, along with other decision tree parameters, to identify the optimal setup for your particular dataset.
*  The entropy of a dataset *D* is calculated using the equation:

<img width="341" alt="Screen Shot 2024-02-19 at 10 11 20 AM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/092cb5ba-59d6-45e0-b268-6a580ce2ac21">

*  *m* represents the total number of classes.
*  *P<sub>i</sub>* denotes the proportion of examples in the dataset that belong to class *i*.
*  Let's revisit the orange box at the first-level node of the iris decision tree for an illustrative example. The node contains a total of 52 cases, with 45 belonging to the class Setosa. Therefore, the proportion *p* or the Setosa class is $45 \over 52$. For the remaining two classes, their proportions are  $6 \over 52$ and  $1 \over 52$, respectively.
*  To calculate the entropy *D*,  we insert these numbers into the entropy equation:


<img width="463" alt="Screen Shot 2024-02-19 at 10 23 04 AM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/53aa3e23-77dc-4ad4-980b-5ad7c24ec607">

* The calculated entropy value of 0.6496 signals a level of purity in the node, significantly shaped by the majority class, Setosa. However, considering that the maximum entropy for binary outcomes can be upto 1, the value of 0.6496 falls short of being considered ideal. A higher entropy value signifies a more equitable class distribution within the node, typically denoting a higher degree of 'impurity disorder.'
* It's worthy to note that in the context of classification problems, such as with classification trees, the Gini index and entropy serve as criteria for deciding whether to split or further develop the tree. **These metrics do not measure model performance**. Instead, performance metrics for classification problems include accuracy, precision, recall, F1 scores, and the area under the curve, among others, which I will explain later in the post.

### When Should We Stop Growing A Tree? - Tuning Hyperparameters

Understanding impurity indices gives rise to a critical question: When and how do we decide to stop tree growth? Numerous criteria, essentially hyperparameters, can guide this decision. Without adjusting these hyperparameters, a tree might grow too complex, fitting too closely to the training data, which leads to overfitting. Decision tree algorithms offer a variety of hyperparameters for fine-tuning, with the following being among the most common across different algorithms if you are a 'scikit-learn' user:

1. **Minimum number of bbservations in a node (`min_samples_leaf`)**: This parameter ensures that a node doesn't split unless it contains a specified minimum number of observations. This criterion is particularly useful for controlling the tree's granularity and can be adjusted based on the class distribution in your dataset. For instance, in imbalanced datasets, setting this parameter thoughtfully considering the sample size of the minority class can prevent the class from being overlooked.

2. **Minimum number of samples required to split (`min_samples_split`)**: This parameter dictates the minimum number of samples a node must have before it can be split. It's a crucial knob for controlling the tree's growth and preventing over-complexity. For instance, if you have million samples, you may not want to set the min_samples_split as 5 because the number is too low and could lead the tree to grow to overfit.

3. **Maximum depth (`max_depth`)**: This parameter limits the tree's depth, with the algorithm halting further splits once the specified depth is reached. An excessively deep tree risks overfitting, capturing noise rather than the underlying data structure, while a shallow tree might underfit, missing out on important patterns. Balancing tree depth is key to achieving a model that generalizes well.

You may have a question: What is the approximate depth of a decision tree trained without restrictions?

The answer to this question hinges on multiple factors, including the dataset's size, the criteria for splitting nodes (such as Gini impurity or entropy in classification trees, or mean squared error for regression trees), and the distribution of features and labels (for instance, whether the target classes are balanced). A general guideline for a perfectly balanced binary tree—where each node has two children and the structure is as compact as possible, with all levels fully occupied except perhaps the last—is to use the base-2 logarithm of the number of instances to estimate depth. For example, with 100,000 instances, the approximate depth calculation would be as follows:

```ruby
import math
# Number of instances
instances = 100_000

# Calculate approximate depth of a balanced binary tree
approx_depth = math.log2(instances)

print(approx_depth)
```
Accoridng to the formular, approximately 16 layers should be good for the tree assuming it is a perfectly balanced binary one. However, real-world datasets seldom present such ideal conditions. Consider the challenge of predicting fraudulent transactions in banking, where fraudulent cases are rare compared to legitimate ones. This leads to an imbalanced tree. Calculating the depth in such scenarios becomes more complex and cannot be summarized by the formula above. Unrestricted tree growth when we have imbalanced data risks overfitting, as the tree might excessively tailor itself to minority classes in the data in an effort to achieve homogeneity at the leaves.

To mitigate potential imbalances and the risk of overfitting, I would recommend always fine-tuning your hyperparameters, such as setting a maximum depth, specifying a minimum number of samples required for a split, or implementing pruning strategies after the tree's development. These adjustments help control the tree's complexity and enhance its predictive power by focusing on the most informative splits, regardless of initial assumptions about the tree's balance.

4. **Additional parameters**: Beyond the basics, parameters like `min_weight_fraction_leaf`, `max_leaf_nodes`, and `max_features` offer further control. For example, `max_features` restricts the number of features evaluated for each split, useful for high-dimensional data. Adjusting these parameters allows for a more nuanced optimization of the tree's structure, considering computational efficiency and model accuracy.

In essence, fine-tuning these hyperparameters involves finding a sweet spot between the model's bias and variance. The goal is to create a decision tree that is complex enough to capture the underlying patterns in the data but simple enough to generalize well to unseen data. *Experimentation* and *cross-validation* are key strategies in finding this balance and selecting the optimal set of hyperparameters.

### Cross-Validation to Find the Best Hyperparameter Values

Cross-validation is a technique to evaluate and refine a model during its training phase, addressing overfitting and gaining insights into the model's potential performance on unseen datasets. This method is particularly useful for determining the optimal settings for hyperparameters. You can apply the cross validation technique to identify best hyperparameters by utilizing `GridSearchCV`, which implements the following steps:

<img width="787" alt="Screen Shot 2024-02-19 at 1 28 43 PM" src="https://github.com/KayChansiri/DecisionTree/assets/157029107/bb03de85-bcd6-4b60-bab4-8a8869f2044c">


1. **Dividing the Training Set**: The first step of cross-validation is segmenting the training dataset into smaller subsets or folds. With `GridSearchCV`, you can specify the number of cross-validation folds ('cv') based on your preferences and dataset size. For larger datasets, a common approach is to use between 5 to 10 folds to manage computational demands efficiently. Smaller datasets may allow for more folds, depending on your available resources and time constraints.

2. **Iterative Training and Validation**: In this phase, the model is trained on *n* - 1 folds and validated on the remaining fold. This procedure is repeated such that each fold serves as the validation set exactly once, ensuring comprehensive utilization of the data for training and validation. For instance, with `cv=5`, your classification tree is trained on 4 folds and validated on 1 fold in each iteration. This iterative approach not only aids in assessing the model's performance but also facilitates the identification of the most effective hyperparameters. Consider testing various combinations, such as 20 values for `max_depth`, 100 for `min_samples_leaf`, and 2 for `criterion` (i.e., `gini` and `entropy`), the Gridsearch would perform 4,000 combinations of hyperparamter testing in total. Given 5-fold cross-validation, the entire process encompasses 20,000 training and evaluation iterations (i.e., 4,000 x 5).

3. **Averaging the Errors**: After completing the iterations, the GridSearch function averages the performance metrics across all iterations. This comprehensive exploration helps pinpoint the combination of hyperparameters that yield the best model performance, which could be assessed by multiple metrics such as accuracy in classification tasks. Besides suggesting the best hyperparameters, the function would also suggests the average model performance score based on the traning data such that you can have a rough idea of how well the model is likely to perform on unseen data. If any overfitting issues arise, we may know at this stage. It's a more robust assessment than the performance on just a single train-test split because it reduces the variance associated with the random choice of the train-test partition.

4. **Final Model Training**: With the optimal hyperparameters identified, your decision tree undergoes one final training session on the entire training set using these selected settings, culminating in the final model. If your approach includes separate training, testing, and validation sets, this final model should also be evaluated using the validation set for an additional layer of assessment.

5. **Test Set Prediction**: The ultimate step involves deploying this final model to make predictions on a separate test set. This crucial phase serves as a definitive evaluation of the model's expected performance on new data, offering a realistic gauge of its generalization capabilities.


# Classification Tree Case Example: Predicting Deposit Subscription for Bank Customers

Now that you have built a crucial foundation regarding classification tree, we will explore [a banking dataset from Kaggle](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets?select=test.csv) as our today usecase. I strongly encourage you to examine the dataset in detail, noting the number and types of features, the dataset's purpose, etc., to better acquaint yourself with the data. Although the dataset is relatively straightforward and lacks missing values, which does unfortunately not mirror the complexity of real-world data, it serves as an excellent primer. In brief, the dataset centers on a marketing campaign by a Portuguese bank that uses phone calls to encourage clients to subscribe to a term deposit. Our objective is to predict whether a client will agree to subscribe. Kaggle provides both a training set and a testing set for this purpose, and we'll begin our exploration with the training set.

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

