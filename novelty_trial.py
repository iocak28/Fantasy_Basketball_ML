# Fantasy Basketball feature code
import pandas as pd
import numpy as np
import time
from basketball_reference_web_scraper import client
import os
import gc
import datetime
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# gc collect
gc.collect()

# Configuration
git_path = 'C:/Users/iocak/OneDrive/Masaüstü/git/Fantasy_Basketball_ML/'
player_data_path = 'sample_data/player_data/'
target_path = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/model_data/'

# Merge player data
player_files = os.listdir(git_path + player_data_path)

player_df = pd.DataFrame()

for i in player_files:
    temp_file = pd.read_csv(git_path + player_data_path + i)
    player_df = pd.concat([player_df, temp_file], axis = 0)
    print(i)
           
## Sort Player Data
player_df = player_df.sort_values(by = ['name', 'date'])

## Reset Index
player_df = player_df.reset_index()
player_df = player_df.drop(columns = ['index', 'Unnamed: 0'])

## Edit Team Names if the name changed
player_df.loc[(player_df['team'] == 'Team.CHARLOTTE_BOBCATS'), 'team'] = 'Team.CHARLOTTE_HORNETS'
player_df.loc[(player_df['opponent'] == 'Team.CHARLOTTE_BOBCATS'), 'opponent'] = 'Team.CHARLOTTE_HORNETS'

player_df.loc[(player_df['team'] == 'Team.NEW_JERSEY_NETS'), 'team'] = 'Team.BROOKLYN_NETS'
player_df.loc[(player_df['opponent'] == 'Team.NEW_JERSEY_NETS'), 'opponent'] = 'Team.BROOKLYN_NETS'

player_df.loc[(player_df['team'] == 'Team.NEW_ORLEANS_HORNETS'), 'team'] = 'Team.NEW_ORLEANS_PELICANS'
player_df.loc[(player_df['opponent'] == 'Team.NEW_ORLEANS_HORNETS'), 'opponent'] = 'Team.NEW_ORLEANS_PELICANS'

# Calculate Fantasy Points
player_df['rebounds'] = player_df['defensive_rebounds'] + player_df['offensive_rebounds']

player_df['double_count'] = (player_df[['points_scored', 
                                        'rebounds', 
                                        'assists', 
                                        'blocks', 
                                        'steals']] >= 10).astype(int).sum(axis = 1)
    
player_df['double_double'] = np.where(player_df['double_count'] >= 2, 1, 0)
player_df['triple_double'] = np.where(player_df['double_count'] >= 3, 1, 0)


player_df['fantasy_point'] = (1 * player_df['points_scored'] + 
         0.5 * player_df['made_three_point_field_goals'] + 
         1.25 * player_df['rebounds'] + 
         1.5 * player_df['assists'] + 
         2.0 * player_df['steals'] + 
         2.0 * player_df['blocks'] + 
         -0.5 * player_df['turnovers'] + 
         1.5 * player_df['double_double'] + 
         3.0 * player_df['triple_double'])

# Add Features

## Write lag function (bring values from the games that a player played)

def feature_lagger(df, col, lag_n):
    df[col + '_lag_' + str(lag_n)] = df.groupby('name')[col].shift(periods = lag_n)
    condition = (df.groupby('name')['seconds_played'].shift(periods = lag_n) <= 0)
    df.loc[condition, col + '_lag_' + str(lag_n)] = None
    df[col + '_lag_' + str(lag_n)] =df.groupby('name')[col + '_lag_' + str(lag_n)].fillna(method="ffill")
    #return df[[col + '_lag_' + str(lag_n)]]

### Lag 1 date
feature_lagger(player_df, 'date', 1)

### Days since last game
player_df['days_since_last_game'] = (pd.to_datetime(player_df['date'], format = '%Y-%m-%d') 
                                    - pd.to_datetime(player_df['date_lag_1'], format = '%Y-%m-%d')).dt.days

### Can we determine the optimal weighted avg of lag features
feature_source_wgt = 'fantasy_point'

### weight optimizer function

def wma_feature_optimizer(feature_source_wgt = 'fantasy_point', train_before = 2015, look_lag = 30):
    '''
    Optimize weights in feature extraction:
        Provide:
            feature_source_wgt = 'fantasy_point' (feature name), 
            train_before = 2015 (optimize using the data before this season end year, including), 
            look_lag = 30, (search weights in maximum this lag)
            
        Output: Updates player_df
    '''
    ### Call Lag Function
    for j in range(look_lag):
        feature_lagger(player_df, feature_source_wgt, j + 1)
        print(feature_source_wgt, j + 1)

    ### Fill all NAs with 0
    player_df.fillna(0, inplace = True)
    
    ### filter df to before 2015 data to prevent data leak
    filtered_df = player_df[player_df['season_end_year'] <= train_before]

    ### fit a linear regression to find the weights
    train = filtered_df[[f'{feature_source_wgt}_lag_{str(i + 1)}' for i in range(look_lag)]]
    train_labels = filtered_df[['fantasy_point']]
    
    ### fit model
    lr_model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    
    ### Train the model using the training sets
    lr_model.fit(train, train_labels)
    
    ### find coefs
    coefs = lr_model.coef_.T
    
    ### calculate new feature
    player_df[f'{feature_source_wgt}_wma_{look_lag}'] = np.dot(np.array(player_df[[f'{feature_source_wgt}_lag_{str(i + 1)}' for i in range(look_lag)]]), coefs)
    
    ### drop helper columns
    player_df.drop(columns = [f'{feature_source_wgt}_lag_{str(i + 1)}' for i in range(look_lag)], inplace = True)
    
    
    