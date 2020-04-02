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

# Trial param for func
season_end_year_list = [2015, 2017]
model_obj = linear_model.Lasso()
parameters = {'alpha' : [0.001, 0.01, 0.05, 0.2]}

# tuner function
def ts_cv_tuner(dataset, season_end_year_list, model_obj, parameters):
    '''
    dataset: all data before train-test split (dataframe),
    season_end_year_list: first end last season end year to be used in training - testing (list of ints),
    model_obj: your model objects, 
    parameters: your hyper parameters in a dict
    '''
    # Train Test Split
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
    train = scaler.transform(train)
    test = scaler.transform(test)
    
    # Cross Validiation for Best Param
    clf = GridSearchCV(model_obj, 
                   parameters, 
                   n_jobs=-1, 
                   cv = TimeSeriesSplit(max_train_size=None, n_splits=3),
                   verbose = 2)
    clf.fit(X = train, y = train_labels)
    
    winner_model = clf.best_estimator_
    
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
    
    test_metrics = {'test_mae' : float(test_mae), 
                    'test_rmse' : float(test_rmse), 
                    'test_bias' : float(test_bias)}
    
    return winner_model, test_metrics
    

# Run a loop for desired models - params

# Trial param for func
season_end_year_list = [2015, 2017]

model_param_list = [(linear_model.LinearRegression(), {}), 
                    (linear_model.Lasso(), {'alpha' : [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9]}),
                    (xgboost.XGBRegressor(), {'objective':['reg:linear'],
                                              'learning_rate': [0.01, 0.05, 0.1, 0.3], #so called `eta` value
                                              'max_depth': [2, 5, 8],
                                              'min_child_weight': [1, 4, 10],
                                              'silent': [1],
                                              'subsample': [0.7, 1],
                                              'colsample_bytree': [0.7, 1],
                                              'n_estimators': [50, 100, 200],
                                              'lambda' : [0.01, 0.2, 0.5, 1],
                                              'alpha' : [0.01, 0.2, 0.5, 1]}),
                    (AdaBoostRegressor(), {'n_estimators' : [20, 40, 60, 100, 200],
                                           'learning_rate' : [0.01, 0.1, 0.5, 1],
                                           'loss' : ['linear', 'square', 'exponential']}),
                    (RandomForestRegressor(), {'n_estimators': [200, 500],
                                               'max_features': ['auto', 'sqrt'],
                                               'max_depth': [5, 10, 50, None],
                                               'min_samples_split': [2, 5, 10],
                                               'min_samples_leaf': [1, 5, 10],
                                               'bootstrap': [True, False]}),
                    (SVR(), {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
                             'C' : [1,5,10],
                             'degree' : [3,8],
                             'coef0' : [0.01,10,0.5],
                             'gamma' : ('auto','scale')})]
    
result_list = []
counter = 0

# call function

for i in model_param_list:
    temp_winner, temp_metrics = ts_cv_tuner(dataset, season_end_year_list, i[0], i[1])
    result_list.append([temp_winner, temp_metrics])
    
    temp_metrics['model'] = str(temp_winner)
    pd.DataFrame(temp_metrics, index = [0]).to_csv(target + f'model_{counter}.csv')
    
    print('\n Result: \n', temp_winner, '\n', temp_metrics, '\n \n')
    counter += 1


##########
# Tune lasso    
i = model_param_list[1]
temp_winner, temp_metrics = ts_cv_tuner(dataset, season_end_year_list, i[0], i[1])
temp_metrics['model'] = str(temp_winner)
pd.DataFrame(temp_metrics, index = [0]).to_csv(target + f'model_{counter}_lasso.csv')
##########