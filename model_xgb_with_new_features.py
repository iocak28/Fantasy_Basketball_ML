# Modeling Code

import pandas as pd
import numpy as np
import time
#from basketball_reference_web_scraper import client
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
target = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/predictions/'

# Read Data
dataset = pd.read_parquet(feature + 'all_data.parquet')

# selected features from former all feat xgb trainings, 0.9 cumul fscore
selected_features = ['predictor_double_count_wma_30',
                    'predictor_fantasy_point_wma_30',
                    'predictor_seconds_played_wma_30',
                    'salary_edited',
                    'predictor_points_scored_wma_30',
                    'starter_yes',
                    'player_ly_total_turnovers',
                    'starter_no',
                    'predictor_made_field_goals_wma_30',
                    'fantasy_point_lag_1_rollmean_4',
                    'predictor_attempted_free_throws_wma_30',
                    'no_salary_info',
                    'team_date_sum_predictor_triple_double_wma_30',
                    'team_date_sum_predictor_points_scored_wma_30',
                    'team_date_sum_predictor_fantasy_point_wma_30',
                    'predictor_attempted_field_goals_wma_30',
                    'team_date_sum_predictor_days_since_last_game_wma_30',
                    'team_date_sum_predictor_rebounds_wma_30',
                    'per_minute_player_ly_points',
                    'predictor_turnovers_wma_30',
                    'per_minute_player_ly_games_played',
                    'team_date_sum_predictor_seconds_played_wma_30',
                    'seconds_played_lag_1',
                    'team_date_sum_predictor_made_field_goals_wma_30',
                    'team_date_sum_predictor_attempted_field_goals_wma_30',
                    'team_date_sum_predictor_personal_fouls_wma_30',
                    'team_date_sum_seconds_played_lag_1',
                    'opponent_date_sum_predictor_days_since_last_game_wma_30',
                    'team_date_sum_predictor_double_count_wma_30',
                    'fantasy_point_lag_1_rollmean_5',
                    'team_date_sum_predictor_attempted_free_throws_wma_30',
                    'team_date_sum_predictor_made_free_throws_wma_30',
                    'team_date_sum_predictor_attempted_three_point_field_goals_wma_30',
                    'team_date_sum_seconds_played_lag_1_rollmean_4',
                    'team_date_sum_predictor_turnovers_wma_30',
                    'player_ly_total_attempted_free_throws',
                    'days_since_last_game',
                    'fantasy_point_lag_1_rollmean_3',
                    'team_date_sum_seconds_played_lag_2',
                    'team_date_sum_predictor_defensive_rebounds_wma_30',
                    'team_date_sum_predictor_made_three_point_field_goals_wma_30',
                    'opponent_date_sum_predictor_attempted_three_point_field_goals_wma_30',
                    'double_double_lag_1_rollmean_4',
                    'player_ly_total_steals',
                    'odd_team',
                    'per_minute_player_ly_turnovers',
                    'opponent_date_sum_predictor_made_three_point_field_goals_wma_30',
                    'per_minute_player_ly_made_field_goals',
                    'team_date_sum_predictor_double_double_wma_30',
                    'steals_lag_1_rollmean_5',
                    'player_ly_total_offensive_rebounds',
                    'turnovers_lag_1_rollmean_4',
                    'attempted_field_goals_lag_3',
                    'opponent_date_sum_predictor_assists_wma_30',
                    'opponent_avg_player_ly_total_personal_fouls',
                    'opponent_avg_player_ly_total_points',
                    'team_avg_player_ly_total_made_three_point_field_goals',
                    'per_minute_player_cumul_assists',
                    'team_avg_player_ly_total_points',
                    'opponent_date_sum_seconds_played_lag_1_rollmean_5',
                    'team_date_sum_seconds_played_lag_3',
                    'team_date_sum_attempted_field_goals_lag_1_rollmean_3',
                    'opponent_avg_player_ly_total_attempted_field_goals',
                    'odd_opponent',
                    'opponent_avg_player_ly_total_attempted_free_throws',
                    'rebounds_lag_1_rollmean_5',
                    'team_avg_per_minute_player_cumul_offensive_rebounds',
                    'is_home',
                    'team_date_sum_predictor_assists_wma_30',
                    'team_date_sum_made_field_goals_lag_1_rollmean_4',
                    'predictor_double_double_wma_30',
                    'double_count_lag_1',
                    'predictor_assists_wma_30',
                    'team_date_sum_rebounds_lag_1_rollmean_3',
                    'per_minute_player_ly_steals',
                    'opponent_avg_per_minute_player_cumul_defensive_rebounds',
                    'opponent_date_sum_rebounds_lag_1_rollmean_5',
                    'team_date_sum_seconds_played_lag_1_rollmean_3',
                    'opponent_date_sum_predictor_steals_wma_30',
                    'team_date_sum_points_scored_lag_1_rollmean_5',
                    'per_minute_player_cumul_made_field_goals',
                    'team_date_sum_fantasy_point_lag_1',
                    'turnovers_lag_3',
                    'defensive_rebounds_lag_1_rollmean_5',
                    'opponent_avg_player_ly_total_blocks',
                    'opponent_date_sum_predictor_made_free_throws_wma_30',
                    'team_date_sum_assists_lag_1_rollmean_5',
                    'team_date_sum_turnovers_lag_1_rollmean_3',
                    'team_avg_per_minute_player_ly_made_free_throws',
                    'team_date_sum_turnovers_lag_1_rollmean_5',
                    'opponent_avg_age',
                    'fantasy_point_lag_3',
                    'turnovers_lag_1_rollmean_3',
                    'team_date_sum_rebounds_lag_1_rollmean_5',
                    'team_date_sum_made_field_goals_lag_2',
                    'per_minute_player_cumul_steals',
                    'opponent_date_sum_defensive_rebounds_lag_2',
                    'team_avg_per_minute_player_cumul_attempted_free_throws',
                    'per_minute_player_ly_assists',
                    'steals_lag_1_rollmean_3',
                    'team_date_sum_rebounds_lag_1',
                    'predictor_days_since_last_game_wma_30',
                    'opponent_avg_per_minute_player_ly_attempted_three_point_field_goals',
                    'opponent_date_sum_attempted_three_point_field_goals_lag_1',
                    'made_three_point_field_goals_lag_1_rollmean_5',
                    'team_date_sum_attempted_free_throws_lag_1_rollmean_4',
                    'team_avg_player_ly_total_blocks',
                    'per_minute_player_cumul_made_three_point_field_goals',
                    'player_ly_total_assists',
                    'attempted_field_goals_lag_2',
                    'team_avg_per_minute_player_cumul_turnovers',
                    'team_date_sum_attempted_field_goals_lag_1_rollmean_4',
                    'per_minute_player_ly_blocks',
                    'team_date_sum_rebounds_lag_1_rollmean_4',
                    'opponent_date_sum_seconds_played_lag_1_rollmean_3',
                    'opponent_avg_per_minute_player_cumul_blocks',
                    'team_avg_player_ly_total_defensive_rebounds',
                    'team_avg_per_minute_player_cumul_games_played',
                    'opponent_date_sum_predictor_personal_fouls_wma_30',
                    'team_date_sum_points_scored_lag_3',
                    'team_avg_per_minute_player_ly_blocks',
                    'opponent_avg_player_ly_total_offensive_rebounds',
                    'per_minute_player_cumul_turnovers',
                    'opponent_date_sum_predictor_rebounds_wma_30',
                    'player_ly_total_blocks',
                    'per_minute_player_cumul_made_free_throws',
                    'team_date_sum_attempted_three_point_field_goals_lag_1_rollmean_4',
                    'team_avg_per_minute_player_cumul_games_started',
                    'rebounds_lag_1_rollmean_4',
                    'opponent_avg_player_ly_total_made_field_goals',
                    'made_field_goals_lag_1_rollmean_4',
                    'player_ly_total_defensive_rebounds',
                    'opponent_avg_per_minute_player_ly_made_free_throws',
                    'opponent_date_sum_predictor_attempted_field_goals_wma_30',
                    'points_scored_lag_2',
                    'opponent_date_sum_attempted_three_point_field_goals_lag_1_rollmean_5',
                    'Pos_SF',
                    'points_scored_lag_1',
                    'opponent_avg_per_minute_player_cumul_made_free_throws',
                    'team_avg_Pos_PF',
                    'team_date_sum_predictor_steals_wma_30',
                    'opponent_avg_player_ly_total_attempted_three_point_field_goals',
                    'opponent_date_sum_predictor_double_double_wma_30',
                    'opponent_date_sum_attempted_free_throws_lag_1_rollmean_5',
                    'team_date_sum_made_free_throws_lag_1_rollmean_5',
                    'team_avg_player_ly_total_made_field_goals',
                    'team_date_sum_rebounds_lag_2',
                    'age',
                    'team_date_sum_assists_lag_1_rollmean_4',
                    'assists_lag_1_rollmean_5',
                    'opponent_avg_per_minute_player_cumul_attempted_free_throws',
                    'defensive_rebounds_lag_1',
                    'per_minute_player_ly_games_started',
                    'opponent_date_sum_points_scored_lag_1',
                    'opponent_date_sum_seconds_played_lag_1',
                    'opponent_date_sum_points_scored_lag_3',
                    'predictor_made_free_throws_wma_30',
                    'opponent_avg_Pos_PG',
                    'per_minute_player_ly_defensive_rebounds',
                    'opponent_avg_per_minute_player_cumul_steals',
                    'team_avg_per_minute_player_ly_points',
                    'opponent_date_sum_turnovers_lag_1_rollmean_5',
                    'team_date_sum_double_count_lag_2',
                    'team_date_sum_double_double_lag_2',
                    'opponent_date_sum_made_three_point_field_goals_lag_2',
                    'opponent_date_sum_steals_lag_2',
                    'predictor_triple_double_wma_30',
                    'opponent_date_sum_predictor_defensive_rebounds_wma_30',
                    'opponent_date_sum_double_count_lag_1',
                    'team_date_sum_fantasy_point_lag_1_rollmean_5',
                    'seconds_played_lag_2',
                    'opponent_avg_per_minute_player_cumul_personal_fouls',
                    'opponent_date_sum_fantasy_point_lag_3',
                    'attempted_free_throws_lag_1_rollmean_4',
                    'per_minute_player_cumul_attempted_free_throws',
                    'turnovers_lag_1',
                    'opponent_date_sum_predictor_fantasy_point_wma_30',
                    'team_avg_per_minute_player_ly_attempted_free_throws',
                    'opponent_date_sum_attempted_free_throws_lag_2',
                    'opponent_date_sum_predictor_triple_double_wma_30',
                    'per_minute_player_cumul_offensive_rebounds',
                    'team_avg_age',
                    'team_avg_per_minute_player_cumul_made_free_throws',
                    'opponent_avg_Pos_C',
                    'team_avg_player_ly_total_minutes_played',
                    'attempted_three_point_field_goals_lag_1_rollmean_3',
                    'opponent_avg_cumulative_season_experience_past',
                    'per_minute_player_cumul_defensive_rebounds',
                    'opponent_date_sum_predictor_turnovers_wma_30',
                    'predictor_defensive_rebounds_wma_30']

# Run a loop for desired models - params
## Trial param for func
season_end_year_list = [[2015, 2017], [2015, 2018], [2015, 2019]]

model_param_list = [('lin_reg', linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)),
                    ('lasso', linear_model.Lasso(alpha=0.05, copy_X=True, fit_intercept=True, max_iter=1000,
                                    normalize=False, positive=False, precompute=False, random_state=None,
                                    selection='cyclic', tol=0.0001, warm_start=False)),
                    ('xgboost', xgboost.XGBRegressor(alpha=0.9, base_score=0.5, booster=None, colsample_bylevel=1,
                                colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
                                importance_type='gain', interaction_constraints=None,
                                learning_rate=0.05, max_delta_step=0, max_depth=3,
                                min_child_weight=1, monotone_constraints=None,
                                n_estimators=200, n_jobs=0, num_parallel_tree=1,
                                objective='reg:linear', random_state=0, reg_alpha=0.899999976,
                                reg_lambda=0.5, scale_pos_weight=1, subsample=0.7,
                                tree_method=None, validate_parameters=False, verbosity=None))]
                    
for i in model_param_list:
    for j in season_end_year_list:
        # Train Test Split
        train = dataset[(dataset['season_end_year'] >= j[0]) & 
                        (dataset['season_end_year'] < j[1])].drop(
                        columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])

        train_labels = dataset[(dataset['season_end_year'] >= j[0]) & 
                        (dataset['season_end_year'] < j[1])][['fantasy_point']]

        test = dataset[(dataset['season_end_year'] == j[1])].drop(
                columns = ['date', 'opponent', 'team', 'name', 'slug', 'season_end_year', 'fantasy_point'])

        test_labels = dataset[(dataset['season_end_year'] == j[1])][['fantasy_point']]

        # Filter desires cols (not an input of the function, used from global env)
        train = train[selected_features]
        train_col_names = list(train.columns)
        
        test = test[selected_features]

        # Scale data
        scaler = StandardScaler()

        # Fit on training set only.
        scaler.fit(train)

        # Apply transform to both the training set and the test set.
        train = scaler.transform(train)
        test = scaler.transform(test)
        
        # Define Model
        winner_model = i[1]

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
        
        pd.DataFrame(test_metrics, index = [0]).to_csv(target + f'model_{i[0]}_year_{str(j)}_selectedfeat_errormetrics_withnewfeat_predictorfeat.csv')
        pd.DataFrame(y_pred).to_csv(target + f'model_{i[0]}_year_{str(j)}_selectedfeat_preds_withnewfeat_predictorfeat.csv')
        
        print(i, j, test_metrics, '\n')


feat_imp = pd.DataFrame({'col_name' : train_col_names,
                         'f_score' : winner_model.feature_importances_}).sort_values(by = ['f_score'], ascending = False)
feat_imp = feat_imp.reset_index(drop = True)
feat_imp['f_score_cumsum'] = feat_imp['f_score'].cumsum()

pyplot.plot(list(feat_imp.index), feat_imp['f_score_cumsum'])
pyplot.title('XGBoost Cumulative f-score')
pyplot.ylabel('Cumulative f-score')
pyplot.xlabel('Number of Features')
