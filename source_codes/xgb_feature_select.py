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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

# gc collect
gc.collect()

# Configuration
feature = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/model_data/'
target = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/tuning/'

# Read Data
dataset = pd.read_parquet(feature + 'all_data.parquet')

# Train Test Split
season_end_year_list = [2015, 2017]

train = dataset[(dataset['season_end_year'] >= season_end_year_list[0]) & 
                (dataset['season_end_year'] < season_end_year_list[1])].drop(
                columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])

train_colnames = train.columns

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
train = scaler.transform(train)
test = scaler.transform(test)

# parameters from a known xgb model from previous trials
winner_model = xgboost.XGBRegressor(alpha=0.9, base_score=0.5, booster=None, colsample_bylevel=1,
                                colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
                                importance_type='gain', interaction_constraints=None,
                                learning_rate=0.05, max_delta_step=0, max_depth=3,
                                min_child_weight=1, monotone_constraints=None,
                                n_estimators=200, n_jobs=0, num_parallel_tree=1,
                                objective='reg:linear', random_state=0, reg_alpha=0.899999976,
                                reg_lambda=0.5, scale_pos_weight=1, subsample=0.7,
                                tree_method=None, validate_parameters=False, verbosity=None)

### Train the model using the training sets
winner_model.fit(train, train_labels)

### Make predictions using the testing set
y_pred = winner_model.predict(test)

### General Error & Bias
err = np.subtract(pd.DataFrame(y_pred), test_labels)
sq_err = np.subtract(pd.DataFrame(y_pred), test_labels)**2

test_mae = float(np.mean(np.abs(err)))
test_rmse = float(np.sqrt(np.mean(sq_err)))
test_bias = float(np.mean(err))

# plot error density
pyplot.hist(err[0], bins = 50, density = True)

## Feature imp

feat_imp = pd.DataFrame({'col_name' : train_colnames,
                         'f_score' : winner_model.feature_importances_}).sort_values(by = ['f_score'], ascending = False)
feat_imp = feat_imp.reset_index(drop = True)
feat_imp['f_score_cumsum'] = feat_imp['f_score'].cumsum()

pyplot.plot(list(feat_imp.index), feat_imp['f_score_cumsum'])
pyplot.title('XGBoost Cumulative f-score')
pyplot.ylabel('Cumulative f-score')
pyplot.xlabel('Number of Features')

feat_imp.to_csv('C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/feature_selection/xgb_tuned_feat_select_2.csv')

feat_imp[feat_imp['f_score_cumsum'] < 0.9]['col_name']


# Run Model with desired features


# Read Data
dataset = pd.read_parquet(feature + 'all_data.parquet')

# Train Test Split
season_end_year_list = [2015, 2017]

train = dataset[(dataset['season_end_year'] >= season_end_year_list[0]) & 
                (dataset['season_end_year'] < season_end_year_list[1])].drop(
                columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])

train_colnames = train.columns

train_labels = dataset[(dataset['season_end_year'] >= season_end_year_list[0]) & 
                (dataset['season_end_year'] < season_end_year_list[1])][['fantasy_point']]

test = dataset[(dataset['season_end_year'] == season_end_year_list[1])].drop(
        columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])

test_labels = dataset[(dataset['season_end_year'] == season_end_year_list[1])][['fantasy_point']]

# Filter desires cols
train = train[feat_imp[feat_imp['f_score_cumsum'] < 0.9]['col_name']]
test = test[feat_imp[feat_imp['f_score_cumsum'] < 0.9]['col_name']]

# Scale data
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train)

# Apply transform to both the training set and the test set.
train = scaler.transform(train)
test = scaler.transform(test)

# parameters from a known xgb model from previous trials
winner_model = xgboost.XGBRegressor(alpha=0.5, base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None, 
             learning_rate=0.05, max_delta_step=0, max_depth=3,
             min_child_weight=1, monotone_constraints=None,
             n_estimators=200, n_jobs=0, num_parallel_tree=1,
             objective='reg:linear', random_state=0, reg_alpha=0.5,
             reg_lambda=1, scale_pos_weight=1, silent=1, subsample=0.7,
             tree_method=None, validate_parameters=False, verbosity=2)

### Train the model using the training sets
winner_model.fit(train, train_labels)

### Make predictions using the testing set
y_pred = winner_model.predict(test)

### General Error & Bias
err_2 = np.subtract(pd.DataFrame(y_pred), test_labels)
sq_err_2 = np.subtract(pd.DataFrame(y_pred), test_labels)**2

float(np.mean(np.abs(err_2)))
float(np.sqrt(np.mean(sq_err_2)))
float(np.mean(err_2))






