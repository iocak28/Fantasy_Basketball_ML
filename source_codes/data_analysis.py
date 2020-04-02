# Modeling Code

import pandas as pd
import numpy as np
import time
# from basketball_reference_web_scraper import client
import os
import gc
import datetime
import matplotlib.pyplot as plt

# gc collect
gc.collect()

# Configuration
git_path = 'C:/Users/iocak/OneDrive/Masaüstü/git/Fantasy_Basketball_ML/'
feature = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/model_data/'
player_data_path = 'sample_data/player_data/'
prediction_path = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/pred_xgb/'

# Read Data
model_data_notclean = pd.read_parquet(feature + 'all_data_v0_notclean.parquet')
model_data_clean = pd.read_parquet(feature + 'all_data.parquet')

# Raw Data
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

# Read best pred data

pred_years = [2017, 2018, 2019]
pred_files = ['model_xgboost_year_[2015, 2017]_selectedfeat_preds_xgbfinalpred.csv', 
              'model_xgboost_year_[2015, 2018]_selectedfeat_preds_xgbfinalpred.csv', 
              'model_xgboost_year_[2015, 2019]_selectedfeat_preds_xgbfinalpred.csv']

pred_xgb = pd.DataFrame()

for i in pred_files:
    temp_pred = pd.read_csv(prediction_path + i)
    temp_pred.drop(columns = ['Unnamed: 0'], inplace = True)
    temp_pred = temp_pred.rename(columns = {'0' : 'xgb_pred'})
    
    pred_xgb = pd.concat([pred_xgb, temp_pred], axis = 0)
    
# Analysis on Raw Data

## Summarize Fantasy Points

### Uncleaned Data
model_data_notclean[model_data_notclean['season_end_year'].isin(list(np.arange(2015, 2020)))][['fantasy_point']].agg(['mean', 'std', 'min', 'max'])

#      fantasy_point
#mean      20.642885
#std       14.030004
#min       -1.500000
#max      103.500000

plt.hist(model_data_notclean[model_data_notclean['season_end_year'].isin(list(np.arange(2015, 2020)))]['fantasy_point'], bins = 50, density = True)
plt.title('Distribution of Fantasy Points 2014-2015 to 2018-2019')
plt.show()

### Uncleaned Data
model_data_clean[model_data_clean['season_end_year'].isin(list(np.arange(2015, 2020)))][['fantasy_point']].agg(['mean', 'std', 'min', 'max'])

#      fantasy_point
#mean      20.748286
#std       13.602257
#min        0.000000
#max       70.000000

plt.hist(model_data_clean[model_data_clean['season_end_year'].isin(list(np.arange(2015, 2020)))]['fantasy_point'], bins = 50, density = True)
plt.title('Distribution of Fantasy Points, Cleaned, 2014-2015 to 2018-2019')
plt.show()

# Plots Fantasy Points vs Lag Features
lag_data = model_data_notclean[['fantasy_point', 'fantasy_point_lag_1', 'seconds_played_lag_1']]

# fantasy_point_lag_1
plt.subplot(1, 2, 1)
plt.scatter(lag_data['fantasy_point_lag_1'], 
            lag_data['fantasy_point'], 
            c = 'b', 
            alpha = 0.1,
            s = 1)
plt.xlabel('fantasy_point_lag_1')
plt.ylabel('fantasy_point')
plt.title('Fantasy Points Lag 1 vs Fantasy Points')

# seconds played
plt.subplot(1, 2, 2)
plt.scatter(lag_data['seconds_played_lag_1'], 
            lag_data['fantasy_point'], 
            c = 'b', 
            alpha = 0.1,
            s = 1)
plt.xlabel('seconds_played_lag_1')
plt.ylabel('fantasy_point')
plt.title('Seconds Played Lag 1 vs Fantasy Points')

# What happens if we use lag features as predictor, naive predictor, people base their decision on these
case = model_data_clean[model_data_clean['season_end_year'].isin(list(np.arange(2017, 2020)))][['date', 
                        'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point',
                        'fantasy_point_lag_1', 'fantasy_point_lag_1_rollmean_5']]

# merge xgb pred
case = pd.merge(case, pred_xgb[['date', 'name', 'xgb_pred']], how = 'left', on = ['date', 'name'])

case['error_fantasy_point_lag_1'] = case['fantasy_point_lag_1'] - case['fantasy_point']
case['error_fantasy_point_lag_1_rollmean_5'] = case['fantasy_point_lag_1_rollmean_5'] - case['fantasy_point']
case['error_xgb_pred'] = case['xgb_pred'] - case['fantasy_point']

# mae
mae_naive = np.abs(case[['error_fantasy_point_lag_1', 'error_fantasy_point_lag_1_rollmean_5', 'error_xgb_pred']]).agg(['mean'])

# rmse
rmse_naive = np.sqrt((case[['error_fantasy_point_lag_1', 'error_fantasy_point_lag_1_rollmean_5', 'error_xgb_pred']]**2).agg(['mean']))

# bias naive
bias_naive = case[['error_fantasy_point_lag_1', 'error_fantasy_point_lag_1_rollmean_5', 'error_xgb_pred']].agg(['mean'])

## case study: stephen curry

# filter
case_spec = case[case['name'] == 'Stephen Curry']

# mae
np.abs(case_spec[['error_fantasy_point_lag_1', 'error_fantasy_point_lag_1_rollmean_5', 'error_xgb_pred']]).agg(['mean']).T

# rmse
np.sqrt((case_spec[['error_fantasy_point_lag_1', 'error_fantasy_point_lag_1_rollmean_5', 'error_xgb_pred']]**2).agg(['mean'])).T

# bias naive
case_spec[['error_fantasy_point_lag_1', 'error_fantasy_point_lag_1_rollmean_5', 'error_xgb_pred']].agg(['mean']).T



