### Data Preparation Using the Banking Data from Kaggle
# Import necessary packages
import pandas as pd
import numpy as np

# Define the file path
file_path = 'Insert your file path here'

import pandas as pd

# Read the CSV file into a pandas DataFrame, specifying the correct delimiter
df = pd.read_csv(file_path, delimiter=';')

# Describe the Dataframe
summary_stats = df.describe(include='all')

print(summary_stats)

# Get a summary of information about the DataFrame including the type of variables
print(df.info())

# Get the first few rows to confirm it looks correct
print(df.head())


# Select columns to encode: all object dtype columns except 'y'
columns_to_encode = df.select_dtypes(include=['object']).columns.drop('y')

# Use pd.get_dummies() to one-hot encode the selected columns
df_encoded = pd.get_dummies(df, columns=columns_to_encode)


# Get a summary of information about the DataFrame including the type of variables
print(df_encoded.info())

# Get the first few rows to confirm it looks correct
print(df_encoded.head())


# encode the target variable
df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})

#### Model Training Using max_depth = 4 

# Train the model using max_depth = 4 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Split the data into features and target variable
X = df_encoded.drop(columns=['y'])  # All columns except y as features
y = df_encoded['y']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=42, max_depth = 4)
classifier.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(40, 10))
plot_tree(classifier, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

### Performance Evaluation (max_depth = 4)
#Model Performance Eval (max_depth = 4)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict the test set results
y_pred = classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Calculate precision, recall, and F1-score
class_report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
print(f'Classification Report:\n{class_report}')

#### Fine-Tune Hyperparameters
# use GridSearch to find the best hyperparameters to fine tune

from sklearn.model_selection import GridSearchCV
tree_clf2 = DecisionTreeClassifier(random_state=42)
tree_clf2.fit(X_train, y_train)

# Define the parameter grid to search
param_grid = {
    'max_depth': range(1, 21),  # Exploring max_depth values from 1 to 20
    'min_samples_leaf':range(1, 101),  # Exploring min_samples_leaf from 1 to 101
    'criterion': ['gini', 'entropy']  # Exploring two different criteria
}

# Setup the grid search with cross-validation: 5 folds
grid_search = GridSearchCV(tree_clf2, param_grid, cv=5, scoring='accuracy', return_train_score=True)

# Perform the grid search on the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (accuracy):", grid_search.best_score_)

# Optionally, access the best estimator directly
best_tree_clf = grid_search.best_estimator_


#Training the model again with the best parameters (i.e.,max_depth = 5,  min_samples_leaf = 94) 


# Train the DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=42, max_depth = 5,  min_samples_leaf = 94)
classifier.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(100, 10))
plot_tree(classifier, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

#### Test the Model Performance with the New Parameters
# Model Eval (max_depth = 5,  min_samples_leaf = 94)
# Predict the test set results
y_pred = classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Calculate precision, recall, and F1-score
class_report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
print(f'Classification Report:\n{class_report}')


# Try increasing max_depth to 10 

classifier = DecisionTreeClassifier(random_state=42, max_depth = 10,  min_samples_leaf = 94)
classifier.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(100, 10))
plot_tree(classifier, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()


# Model Eval (max_depth = 10)
# Predict the test set results
y_pred = classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Calculate precision, recall, and F1-score
class_report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
print(f'Classification Report:\n{class_report}')

#### Try Growing the Tree without Fine-Tuning (Demo Purpose)
#grow the tree without limiting any thing 

# Split the data into features and target variable
X = df_encoded.drop(columns=['y'])  # Features
y = df_encoded['y']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf1.fit(X_train, y_train)

#Visualize the unlimited tree
from sklearn.tree import export_graphviz
export_graphviz(
    tree_clf1,
    out_file = "tree_clf1.dot",
    feature_names = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'education_primary', 'education_secondary', 'education_tertiary',
       'education_unknown', 'default_no', 'default_yes', 'housing_no',
       'housing_yes', 'loan_no', 'loan_yes', 'contact_cellular',
       'contact_telephone', 'contact_unknown', 'month_apr', 'month_aug',
       'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
       'poutcome_failure', 'poutcome_other', 'poutcome_success',
       'poutcome_unknown'],
    class_names = ['no', 'yes'],
    rounded = True,
    filled = True)
from graphviz import Source
Source.from_file("tree_clf1.dot")

get_ipython().system('dot -Tpng tree_clf1.dot -o tree_clf1.png')

from IPython.display import Image
Image(filename='tree_clf1.png')


#### Precision - Recall Balance
# Find the best threshold to balance between precision and recall 

from sklearn.metrics import precision_recall_curve
import numpy as np

probabilities = classifier.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)


#Visualizinng the area under the curve

import matplotlib.pyplot as plt

plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Value")
plt.title("Precision and Recall vs. Threshold")
plt.legend()
plt.show()

classifier = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=94)
classifier.fit(X_train, y_train)

# Predict the probabilities for the positive class ([1])
y_proba = classifier.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Apply threshold of 0.8 to determine class predictions
y_pred_threshold = (y_proba >= 0.8).astype(int)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred_threshold)
print(f'Accuracy: {accuracy:.4f}')

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_threshold)
print(f'Confusion Matrix:\n{conf_matrix}')

# Calculate precision, recall, and F1-score
class_report = classification_report(y_test, y_pred_threshold, target_names=['No', 'Yes'])
print(f'Classification Report:\n{class_report}')


#### Test the model with the actual testing set on Kaggle
#Import the testing set: 

# Define the file path
file_path = '/Users/kaychan/Dropbox/DT-test.csv'

import pandas as pd

# Read the CSV file into a pandas DataFrame, specifying the correct delimiter
df_test = pd.read_csv(file_path, delimiter=';')

# Get a summary of information about the DataFrame including the type of variables
print(df_test.info())

# Get the first few rows to confirm it looks correct
print(df_test.head())

# Select columns to encode: all object dtype columns except 'y'
columns_to_encode = df_test.select_dtypes(include=['object']).columns.drop('y')

# Use pd.get_dummies() to one-hot encode the selected columns
df_test_encoded = pd.get_dummies(df_test, columns=columns_to_encode)


# Get a summary of information about the DataFrame including the type of variables
print(df_test_encoded.info())

# Get the first few rows to confirm it looks correct
print(df_test_encoded.head())

# encode the target variable
df_test_encoded['y'] = df_test_encoded['y'].map({'yes': 1, 'no': 0})


#Use the classifier model trained previously with the unseen df_test_encoded

# Splitting df_test_encoded into features (X) and target (y)
X_unseen = df_test_encoded.drop('y', axis=1)  
y_unseen = df_test_encoded['y'] 

# Predict the probabilities for the positive class ([1]) on the unseen dataset using the trained classifier previously
y_proba_unseen = classifier.predict_proba(X_unseen)[:, 1]  # Probability of the positive class

# Apply threshold of 0.8 to determine class predictions for the unseen dataset
y_pred_threshold_unseen = (y_proba_unseen >= 0.8).astype(int)

# Calculate and print the accuracy for the unseen dataset
accuracy_unseen = accuracy_score(y_unseen, y_pred_threshold_unseen)
print(f'Accuracy (Unseen): {accuracy_unseen:.4f}')

# Generate and display the confusion matrix for the unseen dataset
conf_matrix_unseen = confusion_matrix(y_unseen, y_pred_threshold_unseen)
print(f'Confusion Matrix (Unseen):\n{conf_matrix_unseen}')

# Calculate precision, recall, and F1-score for the unseen dataset
class_report_unseen = classification_report(y_unseen, y_pred_threshold_unseen, target_names=['No', 'Yes'])
print(f'Classification Report (Unseen):\n{class_report_unseen}')




