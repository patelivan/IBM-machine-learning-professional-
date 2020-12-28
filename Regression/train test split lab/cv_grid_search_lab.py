#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:00:42 2020

@author: ivanpatel
"""
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# Read boston housing data
boston = pickle.load(open('/Users/ivanpatel/Desktop/boston_housing_clean.pickle', 'rb'))
boston.keys()

# Get boston dataframe
boston_data = boston['dataframe']
print(boston_data.info()); print('\n', boston_data.head())

# Separate the response from the predictors
x = boston_data.drop('MEDV', axis=1)
y = boston_data['MEDV']

# Cross validation using a linear regression-------------------------------

# Instantiate a KFold object and a pipeline
kf = KFold(shuffle=True, random_state=72018, n_splits=3)
estimator = Pipeline([ ('scaler', StandardScaler() ), 
                      ('regression', LinearRegression() ) ])

# Scores on the validation test
scores = cross_val_score(estimator, x, y, cv=kf)
print(scores); print('\n', np.mean(scores))

# Hyperparameter tuning the traditional way -------------------------------
scores = []
alphas = np.geomspace(1e-9, 1e0, num=10)

for alpha in alphas:
    las = Lasso(alpha=alpha, max_iter=100000)
    
    estimator = Pipeline([ ('standard_scaler', StandardScaler() ), 
                           ('regression', las)])
    
    score = cross_val_score(estimator, x, y, cv=kf)
    scores.append(np.mean(score))

# Print the scores list, get the best score's index, and get its alpha
print(scores); print('\n', alphas[np.argmax(scores)])

# Plot the alphas and the scores
plt.semilogx(alphas, scores, '-o')

#  Hyperparameter tuning using Grid Search CV ---------------------------
# polynomial regression

from sklearn.model_selection import GridSearchCV

# Define a pipeline and a params dictionary
estimator = Pipeline([ ('standard_scaler', StandardScaler() ), 
                      ('polynomial_features', PolynomialFeatures() ), 
                      ('ridge', Ridge() ) ])

params = {'polynomial_features__degree': [1,2,3], 
          'ridge__alpha': np.geomspace(4,20, 30) }

# Instantiate a grid object, and tune the parameters. 
grid = GridSearchCV(estimator, params, cv=kf)
grid.fit(x, y)

print('Best Score: ', grid.best_score_); print('\n', 'Best Params: ', grid.best_params_)

# Use the best Parameters on the entire dataset
y_pred = grid.predict(x)
r2_score(y, y_pred)