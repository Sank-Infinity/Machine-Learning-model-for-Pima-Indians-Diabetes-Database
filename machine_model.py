# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:12:48 2020

@author: Sanket Kale
"""

#Importing required libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Diabetes_Dataset.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,8].values
    
tim = list(Y) #typecasting of Y array into list

#Taking care of missing data
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,[2,3,4,5]])
X[:,[2,3,4,5]] = imputer.transform(X[:,[2,3,4,5]])

#Splitting dataset in trainset and test set 
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 , random_state= 0)

#Importing RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =300, criterion= "entropy", random_state=0)
classifier.fit(X_train, Y_train)

#Predicting the results
Y_pred = classifier.predict(X_test)   #prediction for test set 
Y_prediction = classifier.predict(X_train)    #prediction for train set

#Comparing real results with predicted results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)   #result of test set
CM= cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
print("According to confusion matrix accuracy rate of our machine learning model for test_set is :", ((cm[0][0]+cm[1][1])/CM)*100, "%")

cm_1 = confusion_matrix(Y_train, Y_prediction)    #result of train set
CM_1= cm_1[0][0]+cm_1[0][1]+cm_1[1][0]+cm_1[1][1]
print("According to confusion matrix accuracy rate of our machine learning model for train_set is :", ((cm_1[0][0]+cm_1[1][1])/CM_1)*100, "%")

"""
Class Distribution: (class value 1 is interpreted as "tested positive for
   diabetes")

   Class Value  Number of instances
   0            500
   1            268
   

"""
 