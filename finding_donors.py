#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 23:40:48 2017

@author: khchanaq
"""
#####################################################################################################
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))

#####################################################################################################

# TODO: Total number of records
n_records = len(data)

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = (data.iloc[:,-1].values == ">50K").sum()

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = (data.iloc[:,-1].values == "<=50K").sum()

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (n_greater_50k *100.0 / (n_greater_50k+n_at_most_50k))

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

#####################################################################################################

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)

#####################################################################################################

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)

#####################################################################################################

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 1))

#####################################################################################################
# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()

X = features_raw.values

#delete one from each feature to eliminate linear dependancy
f_workclass = pd.get_dummies(pd.Series(X[:,1])).iloc[:,:-1]
f_education_level = pd.get_dummies(pd.Series(X[:,2])).iloc[:,:-1]
f_maritial_status = pd.get_dummies(pd.Series(X[:,4])).iloc[:,:-1]
f_occupation = pd.get_dummies(pd.Series(X[:,5])).iloc[:,:-1]
f_relationship = pd.get_dummies(pd.Series(X[:,6])).iloc[:,:-1]
f_race = pd.get_dummies(pd.Series(X[:,7])).iloc[:,:-1]
f_sex = pd.get_dummies(pd.Series(X[:,8])).iloc[:,:-1]
f_native_country = pd.get_dummies(pd.Series(X[:,12])).iloc[:,:-1]

features = pd.concat([f_workclass,f_education_level,f_maritial_status,f_occupation,
                      f_relationship,f_race,f_sex,f_native_country],axis=1)

# TODO: Encode the 'income_raw' data to numerical values
income = pd.get_dummies(pd.Series(income_raw)).iloc[:,-1]

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

#####################################################################################################

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

#####################################################################################################
# predict always >50K
y_pred = pd.Series([1] * len(y_test), name = '>50K')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# TODO: Calculate accuracy
accuracy = (cm[0,0]+cm[1,1]) * 1.0/cm.sum()

precision = (cm[1,1] * 1.0/cm[:,1].sum())

recall = (cm[1,1] * 1.0/cm[1,:].sum())

beta = 0.5

# TODO: Calculate F-score using the formula above for beta = 0.5
fscore = (1 + beta * beta) * (precision) / ((beta * beta * precision) + recall)

# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)

#####################################################################################################

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    X_train = X_train[:sample_size]
    y_train = y_train[:sample_size]
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train, y_train)
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train, predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train, predictions_train, beta = 0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results

#####################################################################################################

# TODO: Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# TODO: Initialize the three models
clf_A = DecisionTreeClassifier()
clf_B = LogisticRegression()
clf_C = RandomForestClassifier()

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(len(X_train) * 0.01)
samples_10 = int(len(X_train) * 0.1)
samples_100 = len(X_train)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

#####################################################################################################

# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

"""# TODO: Initialize the classifier
clf = RandomForestClassifier()

# TODO: Create the parameters list you wish to tune
parameters = [{'max_depth': range(12,22,1) , 'max_features': range(5,17,1), 'random_state': [None, 100, 200, 500]}]

#14, 10
"""

clf = LogisticRegression()

# TODO: Create the parameters list you wish to tune
parameters = {'solver': ['newton-cg', 'lbfgs', 'sag', 'liblinear'],
              'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'random_state': [0]}

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta = beta)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = scorer,
                           cv = 10,
                           verbose=10)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)


best_score = grid_fit.best_score_
best_parameters = grid_fit.best_params_

# Get the estimator
best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))

#####################################################################################################

# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

# TODO: Train the supervised model on the training set 
model = GradientBoostingClassifier().fit(X_train, y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)

# show scores
predictions = model.predict(X_test)
model_accuracy = accuracy_score(y_test, predictions)
model_fscore = fbeta_score(y_test, predictions, beta=0.5)
print "\nModel accuracy:", model_accuracy, ", fscore: ", model_fscore

# show most importance features
a = np.array(importances)
factors = pd.DataFrame(data = np.array([importances.astype(float), features.columns]).T,
                       columns = ['importances', 'features'])
factors = factors.sort_values('importances', ascending=False)

print "\n top 10 important features"
display(factors[:10])

#####################################################################################################

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))

#####################################################################################################


