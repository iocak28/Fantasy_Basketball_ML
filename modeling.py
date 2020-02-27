# Modeling Code

import pandas as pd
import numpy as np
import time
from basketball_reference_web_scraper import client
import os
import gc
import datetime
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# gc collect
gc.collect()

# Configuration
git_path = 'C:/Users/iocak/OneDrive/Masaüstü/git/Fantasy_Basketball_ML/'
feature = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/model_data/'

# Read Data
dataset = pd.read_parquet(feature + 'all_data.parquet')

# Train Test Split
train = dataset[dataset['season_end_year'] <= 2018].drop(columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])
train_labels = dataset[dataset['season_end_year'] <= 2018][['fantasy_point']]
test = dataset[dataset['season_end_year'] > 2018].drop(columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])
test_labels = dataset[dataset['season_end_year'] > 2018][['fantasy_point']]

# Normalize data
normalizer = sklearn.preprocessing.Normalizer().fit(train)

train_n = normalizer.transform(train)
test_n = normalizer.transform(test)

# Try a basic model

## Linear Regression

### Create linear regression object
regr = linear_model.LinearRegression()

### Train the model using the training sets
regr.fit(train_n, train_labels)

### Make predictions using the testing set
y_pred = regr.predict(test_n)

### The coefficients
print('Coefficients: \n', regr.coef_)
### General Error & Bias
lr_err = np.mean(np.abs(y_pred - test_labels)) / np.mean(test_labels)
lr_bias = np.mean((y_pred - test_labels)) / np.mean(test_labels)

## Lasso
### Create linear regression object
regr = linear_model.Ridge(alpha = 0.7)

### Train the model using the training sets
regr.fit(train_n, train_labels)

### Make predictions using the testing set
y_pred = regr.predict(test_n)

### The coefficients
print('Coefficients: \n', regr.coef_)
### General Error & Bias
np.mean(np.abs(y_pred - test_labels)) / np.mean(test_labels)
np.mean((y_pred - test_labels)) / np.mean(test_labels)
