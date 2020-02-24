# Fantasy Basketball feature code
import pandas as pd
import numpy as np
import time
from basketball_reference_web_scraper import client
import os
import gc
import datetime

# gc collect
gc.collect()

# Configuration
git_path = 'C:/Users/iocak/OneDrive/Masaüstü/git/Fantasy_Basketball_ML/'
player_data_path = 'sample_data/player_data/'
target_path = 'model_data/'

# Merge player data
player_files = os.listdir(git_path + player_data_path)

player_df = pd.DataFrame()

for i in player_files:
    temp_file = pd.read_csv(git_path + player_data_path + i)
    player_df = pd.concat([player_df, temp_file], axis = 0)
    print(i)
    
player_df = player_df.reset_index()
player_df = player_df.drop(columns = ['index', 'Unnamed: 0'])
       
## Sort Player Data
player_df = player_df.sort_values(by = ['name', 'date'])

# Calculate Fantasy Points
player_df['rebounds'] = player_df['defensive_rebounds'] + player_df['offensive_rebounds']

player_df['double_count'] = (player_df[['points_scored', 
                                        'rebounds', 
                                        'assists', 
                                        'blocks', 
                                        'steals']] >= 10).astype(int).sum(axis = 1)
    
player_df['double_double'] = np.where(player_df['double_count'] >= 2, 1, 0)
player_df['triple_double'] = np.where(player_df['double_count'] >= 3, 1, 0)


#player_df[['points_scored', 
#           'made_three_point_field_goals', 
#           'rebounds', 
#           'assists', 
#           'steals', 
#           'blocks', 
#           'turnovers',
#           'double_double',
#           'triple_double']]

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

## Lag 1 date
feature_lagger(player_df, 'date', 1)

## Days since last game
player_df['days_since_last_game'] = (pd.to_datetime(player_df['date'], format = '%Y-%m-%d') 
                                    - pd.to_datetime(player_df['date_lag_1'], format = '%Y-%m-%d')).dt.days

## Lag 1 features
feature_lagger(player_df, 'assists', 1)
feature_lagger(player_df, 'attempted_field_goals', 1)
feature_lagger(player_df, 'attempted_free_throws', 1)
feature_lagger(player_df, 'attempted_three_point_field_goals', 1)
feature_lagger(player_df, 'defensive_rebounds', 1)
feature_lagger(player_df, 'made_field_goals', 1)
feature_lagger(player_df, 'made_free_throws', 1)
feature_lagger(player_df, 'made_three_point_field_goals', 1)
feature_lagger(player_df, 'personal_fouls', 1)
feature_lagger(player_df, 'points_scored', 1)
feature_lagger(player_df, 'seconds_played', 1)
feature_lagger(player_df, 'steals', 1)
feature_lagger(player_df, 'turnovers', 1)
feature_lagger(player_df, 'rebounds', 1)
feature_lagger(player_df, 'double_count', 1)
feature_lagger(player_df, 'double_double', 1)
feature_lagger(player_df, 'triple_double', 1)
feature_lagger(player_df, 'fantasy_point', 1)
feature_lagger(player_df, 'days_since_last_game', 1)

## Lag 2 features
feature_lagger(player_df, 'assists', 2)
feature_lagger(player_df, 'attempted_field_goals', 2)
feature_lagger(player_df, 'attempted_free_throws', 2)
feature_lagger(player_df, 'attempted_three_point_field_goals', 2)
feature_lagger(player_df, 'defensive_rebounds', 2)
feature_lagger(player_df, 'made_field_goals', 2)
feature_lagger(player_df, 'made_free_throws', 2)
feature_lagger(player_df, 'made_three_point_field_goals', 2)
feature_lagger(player_df, 'personal_fouls', 2)
feature_lagger(player_df, 'points_scored', 2)
feature_lagger(player_df, 'seconds_played', 2)
feature_lagger(player_df, 'steals', 2)
feature_lagger(player_df, 'turnovers', 2)
feature_lagger(player_df, 'rebounds', 2)
feature_lagger(player_df, 'double_count', 2)
feature_lagger(player_df, 'double_double', 2)
feature_lagger(player_df, 'triple_double', 2)
feature_lagger(player_df, 'fantasy_point', 2)
feature_lagger(player_df, 'days_since_last_game', 2)

## Lag 3 features
feature_lagger(player_df, 'assists', 3)
feature_lagger(player_df, 'attempted_field_goals', 3)
feature_lagger(player_df, 'attempted_free_throws', 3)
feature_lagger(player_df, 'attempted_three_point_field_goals', 3)
feature_lagger(player_df, 'defensive_rebounds', 3)
feature_lagger(player_df, 'made_field_goals', 3)
feature_lagger(player_df, 'made_free_throws', 3)
feature_lagger(player_df, 'made_three_point_field_goals', 3)
feature_lagger(player_df, 'personal_fouls', 3)
feature_lagger(player_df, 'points_scored', 3)
feature_lagger(player_df, 'seconds_played', 3)
feature_lagger(player_df, 'steals', 3)
feature_lagger(player_df, 'turnovers', 3)
feature_lagger(player_df, 'rebounds', 3)
feature_lagger(player_df, 'double_count', 3)
feature_lagger(player_df, 'double_double', 3)
feature_lagger(player_df, 'triple_double', 3)
feature_lagger(player_df, 'fantasy_point', 3)
feature_lagger(player_df, 'days_since_last_game', 3)

# Drop NAs
player_df = player_df.dropna()

# Save data
player_df.to_parquet(git_path + target_path + 'all_data.parquet', index = False)


