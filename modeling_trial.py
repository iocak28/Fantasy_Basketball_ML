# Modeling Code

import pandas as pd
import numpy as np
import time
# from basketball_reference_web_scraper import client
import os
import gc
import datetime
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from itertools import product
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import xgboost
from matplotlib import pyplot
#from keras.callbacks import ModelCheckpoint
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten, Dropout
##from talos.model import layers
#from keras.regularizers import l1

# gc collect
gc.collect()

# Configuration
feature = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/model_data/'

# Read Data
dataset = pd.read_parquet(feature + 'all_data.parquet')

# Train Test Split
season_end_year_list = [2015, 2017]

train = dataset[(dataset['season_end_year'] >= season_end_year_list[0]) & 
                (dataset['season_end_year'] < season_end_year_list[1])].drop(
                columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])

train_labels = dataset[(dataset['season_end_year'] >= season_end_year_list[0]) & 
                (dataset['season_end_year'] < season_end_year_list[1])][['fantasy_point']]

test = dataset[(dataset['season_end_year'] == season_end_year_list[1])].drop(
        columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])

test_labels = dataset[(dataset['season_end_year'] == season_end_year_list[1])][['fantasy_point']]

# Scale data
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train)

# Apply transform to both the training set and the test set.
train_n = scaler.transform(train)
test_n = scaler.transform(test)

# PCA

## Make an instance of the Model
pca = PCA(.90)
pca.fit(train_n)

train_n = pca.transform(train_n)
test_n = pca.transform(test_n)

# Plot % Variance explained
variance = pca.explained_variance_ratio_ #calculate variance ratios

var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features

pyplot.ylabel('% Variance Explained')
pyplot.xlabel('# of Features')
pyplot.title('PCA Analysis')
pyplot.ylim(30,100.5)
pyplot.style.context('seaborn-whitegrid')


pyplot.plot(var)

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

### Create lasso object
regr = linear_model.Lasso(alpha = 0.1)

### Train the model using the training sets
regr.fit(train_n, train_labels)

### Make predictions using the testing set
y_pred = regr.predict(test_n)

### The coefficients
print('Coefficients: \n', regr.coef_)
### General Error & Bias
lr_err = np.mean(np.abs(y_pred[:, None] - test_labels)) / np.mean(test_labels)
lr_bias = np.mean((y_pred[:, None] - test_labels)) / np.mean(test_labels)

## Decision Tree

# Time Series CV
max_depth = [5, 10, 50]
min_samples_split = [2, 5]
min_samples_leaf = [1, 5]
max_features = ['auto']

parameters = {'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_features': max_features}

clf = GridSearchCV(DecisionTreeRegressor(), 
                   parameters, 
                   n_jobs=-1, 
                   cv = TimeSeriesSplit(max_train_size=None, n_splits=3))
clf.fit(X = train_n, y = train_labels)
tree_model = clf.best_estimator_

### Train the model using the training sets
tree_model.fit(train_n, train_labels)

### Make predictions using the testing set
y_pred = tree_model.predict(test_n)

### General Error & Bias
np.mean(np.abs(y_pred[:, None] - test_labels)) / np.mean(test_labels)
np.mean((y_pred[:, None] - test_labels)) / np.mean(test_labels)


# ideas:
'''
- try rf, xgb, nn if they don't work, try them again by handpicking some features
- selective pca, or some other dim reduction

'''

## Random Forest

# Time Series CV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 100, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

parameters = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

clf = GridSearchCV(RandomForestRegressor(), 
                   parameters, 
                   n_jobs=-1, 
                   cv = TimeSeriesSplit(max_train_size=None, n_splits=3),
                   verbose = 2)
clf.fit(X = train_n, y = train_labels)
tree_model = clf.best_estimator_

### Train the model using the training sets
tree_model.fit(train_n, train_labels)

### Make predictions using the testing set
y_pred = tree_model.predict(test_n)

### General Error & Bias
np.mean(np.abs(y_pred[:, None] - test_labels)) / np.mean(test_labels)
np.mean((y_pred[:, None] - test_labels)) / np.mean(test_labels)

# xgboost

parameters = {'nthread':[-1], #when use hyperthread, xgboost may become slower
          'objective':['reg:linear'],
          'learning_rate': [.03, 0.05, .07], #so called `eta` value
          'max_depth': [5, 6, 7],
          'min_child_weight': [1, 4, 10],
          'silent': [1],
          'subsample': [0.7],
          'colsample_bytree': [0.7],
          'n_estimators': [50, 100]}

xgb1 = xgboost.XGBRegressor()
clf = GridSearchCV(xgb1, 
                   parameters, 
                   n_jobs=-1, 
                   cv = TimeSeriesSplit(max_train_size=None, n_splits=3),
                   verbose = 2)
clf.fit(X = train, y = train_labels)

clf.best_estimator_

best_model = clf.best_estimator_

### Train the model using the training sets
best_model.fit(train, train_labels)


### Make predictions using the testing set
y_pred = best_model.predict(test)

### General Error & Bias
print(np.mean(np.abs(y_pred[:, None] - test_labels)) / np.mean(test_labels))
print(np.mean((y_pred[:, None] - test_labels)) / np.mean(test_labels))

# plot feature importance
xgboost.plot_importance(best_model)
pyplot.figure(figsize=(200,50))
pyplot.show()

feat_imp = pd.DataFrame({'col_name' : train.columns,
              'feature_imp' : best_model.feature_importances_})

feat_imp = feat_imp.sort_values(by = 'feature_imp', ascending=False)