#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sept  30 03:49:12 2018

@author name: Ragavander Rohit Walthaty
          ID: 999992844
"""

'''
# Supervised K-Nearest Neighbours classification with Breast Cancer Dataset
'''

# Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the est dataset: est
est = pd.read_csv('data.csv')
#est = datasets.load_est()

# delete the unwanted id column
est.drop(['id'], 1, inplace=True)

# creating the binary presentation of the dataset
diagnosis = {'M': 1,'B': 0} 
est.diagnosis = [diagnosis[item] for item in est.diagnosis] 
s = est['diagnosis']
est.drop(['diagnosis'], 1, inplace=True)

# Print the shape of the data
print(est)
print(est.shape)
print(s)

#est.drop(['perimeter_mean'], 1, inplace=True)
#est.drop(['area_mean'], 1, inplace=True)

# Importing necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create freature and target arrays
x = est #features
y = s #labels

# Training your KNN model using the training data set
# Evaluatie your KNN model using the test data set

# Split into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1,
                                                    random_state = 19, stratify = y)

# Create a K-NN classifier with 9 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 9)

# Fit the classifier to the training data
knn.fit(x_train, y_train)

# KNN is Supervised learning, you have to provide both features X and labels y
y_pred = knn.predict(x_test)

# Print the accuracy
print(knn.score(x_train, y_train)) #training data accuracy: 0.941
print(knn.score(x_test,y_test)) # test data accuracy: 0.929

# y_test (true labels)
# y_pred (prediction labels)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Overfitting and underfitting

# Setup arrays to store train and test accuracies
neighbors = np.arange(1,30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)
    
    # Fit the classifier to the training data
    knn.fit(x_train, y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)
    
    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(x_test, y_test)
    
# Generate plot
plt.figure()
plt.title('BREAST CANCER \n k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

