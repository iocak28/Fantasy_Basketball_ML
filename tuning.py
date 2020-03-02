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
    
    test_mae = float(np.mean(np.abs(err))) / np.mean(test_labels)
    test_rmse = float(np.sqrt(np.mean(sq_err))) / np.mean(test_labels)
    test_bias = float(np.mean(err)) / np.mean(test_labels)
    
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

#{'test_mae': 0.34015561047684284,
# 'test_rmse': 0.43327618262566864,
# 'test_bias': -0.008486924916293822,
# 'model': "Lasso(alpha=0.05, copy_X=True, fit_intercept=True, max_iter=1000,\n      normalize=False, positive=False, precompute=False, random_state=None,\n      selection='cyclic', tol=0.0001, warm_start=False)"}

# Tune neural network
   
# Load libraries
import numpy as np
from keras import models
from keras import layers
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


# Create function returning a compiled network
def create_network(optimizer='rmsprop', loss = 'mean_absolute_error', neur_first = 500, neur_hid = 1000, dropout_val = 0.2, number_hidden = 1):
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=neur_first, activation='relu', input_shape=(511,)))

    # dropout
    network.add(Dropout(dropout_val))
    
    for i in range(number_hidden):
        # Add fully connected layer with a ReLU activation function
        network.add(layers.Dense(units=neur_hid, activation='relu'))
        
        # dropout
        network.add(Dropout(dropout_val))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='linear'))

    # Compile neural network
    network.compile(loss=loss,
                    optimizer=optimizer) 
    
    # Return compiled network
    return network

neural_network = KerasRegressor(build_fn=create_network, verbose=2)

# Create hyperparameter space
epochs = [30, 100]
batches = [1000]
optimizers = ['Adadelta']
validation_splits = [0.2]
losses =['mean_absolute_error']
neur_first = [100, 500]
neur_hid = [2000]
dropout_val = [0.2]
number_hidden = [1, 2, 3]

#epochs = [30]
#batches = [1000]
#optimizers = ['Adadelta']
#validation_splits = [0.2]
#losses =['mean_squared_error']

# surprisingly this seems to work

# Create hyperparameter options
hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, 
                       validation_split = validation_splits, loss = losses,
                       neur_first = neur_first, neur_hid = neur_hid, dropout_val= dropout_val,
                       number_hidden = number_hidden)

# tune
season_end_year_list = [2015, 2017]
temp_winner, temp_metrics = ts_cv_tuner(dataset, season_end_year_list, neural_network, hyperparameters)

## Winner

#{'test_mae': 0.39348045800303527,
# 'test_rmse': 0.5166015843001839,
# 'test_bias': -0.22493961576689697}
#
#{'verbose': 2,
# 'batch_size': 1000,
# 'dropout_val': 0.2,
# 'epochs': 30,
# 'loss': 'mean_absolute_error',
# 'neur_first': 100,
# 'neur_hid': 2000,
# 'optimizer': 'Adadelta',
# 'validation_split': 0.2,
# 'build_fn': <function __main__.create_network(optimizer='rmsprop', loss='mean_absolute_error', neur_first=500, neur_hid=1000, dropout_val=0.2)>}


