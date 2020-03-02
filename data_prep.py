# Fantasy Basketball feature code
import pandas as pd
import numpy as np
import time
from basketball_reference_web_scraper import client
import os
import gc
import datetime
import matplotlib.pyplot as plt

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

# Player Features

## Home-Away
player_df['is_home'] = 0
player_df.loc[player_df['location'] == 'Location.HOME', 'is_home'] = 1

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

### Lag 1, 2 ,3 Features
feature_sources = ['assists',
                   'attempted_field_goals',
                   'attempted_free_throws',
                   'attempted_three_point_field_goals',
                   'defensive_rebounds',
                   'made_field_goals',
                   'made_free_throws',
                   'made_three_point_field_goals',
                   'personal_fouls',
                   'points_scored',
                   'seconds_played',
                   'steals',
                   'turnovers',
                   'rebounds',
                   'double_count',
                   'double_double',
                   'triple_double',
                   'fantasy_point',
                   'days_since_last_game']

### Call Lag Function
for i in feature_sources:
    for j in range(3):
        feature_lagger(player_df, i, j + 1)
        print(i, j + 1)

## Write roll_mean function: Provide feature name but uses lag_1 column

def feature_roll_mean(df, cols, roll_n):
    df[[i + '_lag_' + str(1) + f'_rollmean_{roll_n}' for i in cols]] = df.groupby('name')[[i + '_lag_' + str(1) for i in cols]].rolling(window = roll_n).mean().reset_index(drop = True)

### Call Lag Function
for i in range(3):
    feature_roll_mean(player_df, feature_sources, i + 3)
    print(i + 1)
    
### Fill all NAs with 0
player_df.fillna(0, inplace = True)

# Player Career Features
player_season_stats = pd.read_csv(git_path + 'sample_data/player_season_data/player_season.csv')

## A player might play in two diff teams in same year, group them
player_season_stats = player_season_stats.groupby(['name', 'slug', 'season_end_year']).agg({'age' : 'mean',
                           'assists' : 'sum',
                           'attempted_field_goals' : 'sum', 
                           'attempted_free_throws' : 'sum', 
                           'attempted_three_point_field_goals' : 'sum', 
                           'blocks'  : 'sum', 
                           'defensive_rebounds' : 'sum', 
                           'games_played' : 'sum', 
                           'games_started' : 'sum', 
                           'made_field_goals' : 'sum', 
                           'made_free_throws' : 'sum',
                           'made_three_point_field_goals' : 'sum', 
                           'minutes_played' : 'sum', 
                           'offensive_rebounds' : 'sum', 
                           'personal_fouls' : 'sum', 
                           'points' : 'sum', 
                           'positions' : 'unique', 
                           'steals' : 'sum', 
                           'turnovers' : 'sum'}).reset_index()
    
if len(player_season_stats) != len(player_season_stats[['slug', 'season_end_year']].drop_duplicates()):
    print('!!! There are duplicates in player_season_stats')

## Sort season data
player_season_stats = player_season_stats.sort_values(by = ['name', 'season_end_year'])
player_season_stats.reset_index(inplace = False)

## Positions
all_positions = [list(i) for i in player_season_stats['positions']]
all_positions = sum(all_positions, [])
all_positions = list(set(all_positions))

player_season_stats['Pos_SG'] = 0
player_season_stats['Pos_PF'] = 0
player_season_stats['Pos_PG'] = 0
player_season_stats['Pos_C'] = 0
player_season_stats['Pos_SF'] = 0

for i in range(len(player_season_stats)):
    temp = list(player_season_stats['positions'][i])
    
    for j in range(len(temp)):
        if temp[j] == "[<Position.SHOOTING_GUARD: 'SHOOTING GUARD'>]":
            player_season_stats.loc[i, 'Pos_SG'] = 1
        if temp[j] == "[<Position.POWER_FORWARD: 'POWER FORWARD'>]":
            player_season_stats.loc[i, 'Pos_PF'] = 1
        if temp[j] == "[<Position.POINT_GUARD: 'POINT GUARD'>]":
            player_season_stats.loc[i, 'Pos_PG'] = 1
        if temp[j] == "[<Position.CENTER: 'CENTER'>]":
            player_season_stats.loc[i, 'Pos_C'] = 1
        if temp[j] == "[<Position.SMALL_FORWARD: 'SMALL FORWARD'>]":
            player_season_stats.loc[i, 'Pos_SF'] = 1
            

grouped_pos = player_season_stats.groupby('name')[['Pos_SG', 'Pos_PF', 'Pos_PG', 'Pos_C', 'Pos_SF']].sum().reset_index()
grouped_pos[['Pos_SG', 'Pos_PF', 'Pos_PG', 'Pos_C', 'Pos_SF']] = np.multiply(grouped_pos[['Pos_SG', 'Pos_PF', 'Pos_PG', 'Pos_C', 'Pos_SF']], 1 / grouped_pos[['Pos_SG', 'Pos_PF', 'Pos_PG', 'Pos_C', 'Pos_SF']]) # Warning is OK
grouped_pos[['Pos_SG', 'Pos_PF', 'Pos_PG', 'Pos_C', 'Pos_SF']] = grouped_pos[['Pos_SG', 'Pos_PF', 'Pos_PG', 'Pos_C', 'Pos_SF']].fillna(0).astype(int)

player_season_stats.drop(columns = ['Pos_SG', 'Pos_PF', 'Pos_PG', 'Pos_C', 'Pos_SF'], inplace = True)
player_season_stats = pd.merge(player_season_stats, grouped_pos, how = 'left', on = 'name')

## Years Played in Nba, Cumulative
player_season_stats['played_this_season'] = 0
player_season_stats.loc[player_season_stats['minutes_played'] > 30, 'played_this_season'] = 1
player_season_stats['played_last_season'] = player_season_stats.groupby('name')['played_this_season'].shift(periods = 1)
player_season_stats.loc[player_season_stats['played_last_season'].isnull(), 'played_last_season'] = 0

player_season_stats['cumulative_season_experience_past'] = player_season_stats[['name', 'played_last_season']].groupby('name').cumsum()

## last year features 
feature_sources_year = ['assists',
                        'attempted_field_goals', 
                        'attempted_free_throws',
                        'attempted_three_point_field_goals', 
                        'blocks', 
                        'defensive_rebounds',
                        'games_played', 
                        'games_started', 
                        'made_field_goals', 
                        'made_free_throws',
                        'made_three_point_field_goals', 
                        'minutes_played', 
                        'offensive_rebounds',
                        'personal_fouls', 
                        'points', 
                        'steals', 
                        'turnovers']

### LY Total
player_season_stats[['player_ly_total_' + i for i in feature_sources_year]] = player_season_stats.groupby('name')[feature_sources_year].shift(periods = 1)
player_season_stats.update(player_season_stats[['player_ly_total_' + i for i in feature_sources_year]].fillna(0))

### LY Per Minute
player_season_stats[['per_minute_player_ly_' + i for i in feature_sources_year]] = player_season_stats[['player_ly_total_' + i for i in feature_sources_year]].div(player_season_stats['player_ly_total_minutes_played'], axis = 0)
player_season_stats.update(player_season_stats[['per_minute_player_ly_' + i for i in feature_sources_year]].fillna(0))

#### Fix infinity
player_season_stats.loc[player_season_stats['per_minute_player_ly_defensive_rebounds'] == np.inf, 'per_minute_player_ly_defensive_rebounds'] = 0
player_season_stats.loc[player_season_stats['per_minute_player_ly_games_played'] == np.inf, 'per_minute_player_ly_games_played'] = 0

### Cumulative Per Minute Until LY
player_season_stats[['per_minute_player_cumul_' + i for i in feature_sources_year]] = player_season_stats.groupby('name')[['player_ly_total_' + i for i in feature_sources_year]].expanding().mean().div(player_season_stats.groupby('name')['player_ly_total_minutes_played'].expanding().mean(), axis = 0).reset_index(inplace = False)[['player_ly_total_' + i for i in feature_sources_year]]
player_season_stats.update(player_season_stats[['per_minute_player_cumul_' + i for i in feature_sources_year]].fillna(0))

# Team Features

## Player lag-mean features converted to team
team_feature_set_lag = ['assists_lag_1',
'assists_lag_2',
'assists_lag_3',
'attempted_field_goals_lag_1',
'attempted_field_goals_lag_2',
'attempted_field_goals_lag_3',
'attempted_free_throws_lag_1',
'attempted_free_throws_lag_2',
'attempted_free_throws_lag_3',
'attempted_three_point_field_goals_lag_1',
'attempted_three_point_field_goals_lag_2',
'attempted_three_point_field_goals_lag_3',
'defensive_rebounds_lag_1',
'defensive_rebounds_lag_2',
'defensive_rebounds_lag_3',
'made_field_goals_lag_1',
'made_field_goals_lag_2',
'made_field_goals_lag_3',
'made_free_throws_lag_1',
'made_free_throws_lag_2',
'made_free_throws_lag_3',
'made_three_point_field_goals_lag_1',
'made_three_point_field_goals_lag_2',
'made_three_point_field_goals_lag_3',
'personal_fouls_lag_1',
'personal_fouls_lag_2',
'personal_fouls_lag_3',
'points_scored_lag_1',
'points_scored_lag_2',
'points_scored_lag_3',
'seconds_played_lag_1',
'seconds_played_lag_2',
'seconds_played_lag_3',
'steals_lag_1',
'steals_lag_2',
'steals_lag_3',
'turnovers_lag_1',
'turnovers_lag_2',
'turnovers_lag_3',
'rebounds_lag_1',
'rebounds_lag_2',
'rebounds_lag_3',
'double_count_lag_1',
'double_count_lag_2',
'double_count_lag_3',
'double_double_lag_1',
'double_double_lag_2',
'double_double_lag_3',
'triple_double_lag_1',
'triple_double_lag_2',
'triple_double_lag_3',
'fantasy_point_lag_1',
'fantasy_point_lag_2',
'fantasy_point_lag_3',
'assists_lag_1_rollmean_3',
'attempted_field_goals_lag_1_rollmean_3',
'attempted_free_throws_lag_1_rollmean_3',
'attempted_three_point_field_goals_lag_1_rollmean_3',
'defensive_rebounds_lag_1_rollmean_3',
'made_field_goals_lag_1_rollmean_3',
'made_free_throws_lag_1_rollmean_3',
'made_three_point_field_goals_lag_1_rollmean_3',
'personal_fouls_lag_1_rollmean_3',
'points_scored_lag_1_rollmean_3',
'seconds_played_lag_1_rollmean_3',
'steals_lag_1_rollmean_3',
'turnovers_lag_1_rollmean_3',
'rebounds_lag_1_rollmean_3',
'double_count_lag_1_rollmean_3',
'double_double_lag_1_rollmean_3',
'triple_double_lag_1_rollmean_3',
'fantasy_point_lag_1_rollmean_3',
'assists_lag_1_rollmean_4',
'attempted_field_goals_lag_1_rollmean_4',
'attempted_free_throws_lag_1_rollmean_4',
'attempted_three_point_field_goals_lag_1_rollmean_4',
'defensive_rebounds_lag_1_rollmean_4',
'made_field_goals_lag_1_rollmean_4',
'made_free_throws_lag_1_rollmean_4',
'made_three_point_field_goals_lag_1_rollmean_4',
'personal_fouls_lag_1_rollmean_4',
'points_scored_lag_1_rollmean_4',
'seconds_played_lag_1_rollmean_4',
'steals_lag_1_rollmean_4',
'turnovers_lag_1_rollmean_4',
'rebounds_lag_1_rollmean_4',
'double_count_lag_1_rollmean_4',
'double_double_lag_1_rollmean_4',
'triple_double_lag_1_rollmean_4',
'fantasy_point_lag_1_rollmean_4',
'assists_lag_1_rollmean_5',
'attempted_field_goals_lag_1_rollmean_5',
'attempted_free_throws_lag_1_rollmean_5',
'attempted_three_point_field_goals_lag_1_rollmean_5',
'defensive_rebounds_lag_1_rollmean_5',
'made_field_goals_lag_1_rollmean_5',
'made_free_throws_lag_1_rollmean_5',
'made_three_point_field_goals_lag_1_rollmean_5',
'personal_fouls_lag_1_rollmean_5',
'points_scored_lag_1_rollmean_5',
'seconds_played_lag_1_rollmean_5',
'steals_lag_1_rollmean_5',
'turnovers_lag_1_rollmean_5',
'rebounds_lag_1_rollmean_5',
'double_count_lag_1_rollmean_5',
'double_double_lag_1_rollmean_5',
'triple_double_lag_1_rollmean_5',
'fantasy_point_lag_1_rollmean_5']

#### Team
player_df_team = player_df.groupby(['team', 'date'])[team_feature_set_lag].sum().reset_index()
player_df_team.rename(columns = dict(zip(team_feature_set_lag, ['team_date_sum_' + i for i in team_feature_set_lag])), inplace = True)

#### Modify team to represent opponent
player_df_opponent = player_df_team.copy()
player_df_opponent.rename(columns = {'team' : 'opponent'}, inplace = True)
player_df_opponent.rename(columns = dict(zip(['team_date_sum_' + i for i in team_feature_set_lag], ['opponent_date_sum_' + i for i in team_feature_set_lag])), inplace = True)

# Team player Career avg
team_avg_player_career = pd.merge(player_df[['name', 'slug', 'date', 'team', 'season_end_year']],
                                  player_season_stats,
                                  how = 'left',
                                  on = ['name', 'slug', 'season_end_year'])

team_avg_career_features = ['age',
                            'Pos_SG',
                            'Pos_PF',
                            'Pos_PG',
                            'Pos_C',
                            'Pos_SF',
                            'played_last_season',
                            'cumulative_season_experience_past',
                            'player_ly_total_assists',
                            'player_ly_total_attempted_field_goals',
                            'player_ly_total_attempted_free_throws',
                            'player_ly_total_attempted_three_point_field_goals',
                            'player_ly_total_blocks',
                            'player_ly_total_defensive_rebounds',
                            'player_ly_total_games_played',
                            'player_ly_total_games_started',
                            'player_ly_total_made_field_goals',
                            'player_ly_total_made_free_throws',
                            'player_ly_total_made_three_point_field_goals',
                            'player_ly_total_minutes_played',
                            'player_ly_total_offensive_rebounds',
                            'player_ly_total_personal_fouls',
                            'player_ly_total_points',
                            'player_ly_total_steals',
                            'player_ly_total_turnovers',
                            'per_minute_player_ly_assists',
                            'per_minute_player_ly_attempted_field_goals',
                            'per_minute_player_ly_attempted_free_throws',
                            'per_minute_player_ly_attempted_three_point_field_goals',
                            'per_minute_player_ly_blocks',
                            'per_minute_player_ly_defensive_rebounds',
                            'per_minute_player_ly_games_played',
                            'per_minute_player_ly_games_started',
                            'per_minute_player_ly_made_field_goals',
                            'per_minute_player_ly_made_free_throws',
                            'per_minute_player_ly_made_three_point_field_goals',
                            'per_minute_player_ly_minutes_played',
                            'per_minute_player_ly_offensive_rebounds',
                            'per_minute_player_ly_personal_fouls',
                            'per_minute_player_ly_points',
                            'per_minute_player_ly_steals',
                            'per_minute_player_ly_turnovers',
                            'per_minute_player_cumul_assists',
                            'per_minute_player_cumul_attempted_field_goals',
                            'per_minute_player_cumul_attempted_free_throws',
                            'per_minute_player_cumul_attempted_three_point_field_goals',
                            'per_minute_player_cumul_blocks',
                            'per_minute_player_cumul_defensive_rebounds',
                            'per_minute_player_cumul_games_played',
                            'per_minute_player_cumul_games_started',
                            'per_minute_player_cumul_made_field_goals',
                            'per_minute_player_cumul_made_free_throws',
                            'per_minute_player_cumul_made_three_point_field_goals',
                            'per_minute_player_cumul_minutes_played',
                            'per_minute_player_cumul_offensive_rebounds',
                            'per_minute_player_cumul_personal_fouls',
                            'per_minute_player_cumul_points',
                            'per_minute_player_cumul_steals',
                            'per_minute_player_cumul_turnovers']

team_avg_player_career = team_avg_player_career.groupby(['date', 'team'])[team_avg_career_features].mean().reset_index()
team_avg_player_career.rename(columns = dict(zip(team_avg_career_features, ['team_avg_' + i for i in team_avg_career_features])), inplace = True)

## opponent avg player career
opponent_avg_player_career = team_avg_player_career.rename(columns = {'team' : 'opponent'}).copy()
opponent_avg_player_career.rename(columns = dict(zip(['team_avg_' + i for i in team_avg_career_features], ['opponent_avg_' + i for i in team_avg_career_features])), inplace = True)

# Betting Data

## Read betting data
bet_data = pd.read_csv(git_path + '/sample_data/betting_data/odds.csv')

bet_data['team'] = pd.Series([i[0] for i in bet_data['sides'].str.split(pat = ' - ')])
bet_data['opponent'] = pd.Series([i[1] for i in bet_data['sides'].str.split(pat = ' - ')])

## Change naming, make it similar to original data (we can extract team performance etc)
naming_map = {'Atlanta Hawks' : 'Team.ATLANTA_HAWKS',
            'Boston Celtics' : 'Team.BOSTON_CELTICS',
            'Charlotte Hornets' : 'Team.CHARLOTTE_HORNETS',
            'Dallas Mavericks' : 'Team.DALLAS_MAVERICKS',
            'Houston Rockets' : 'Team.HOUSTON_ROCKETS',
            'Los Angeles Clippers' : 'Team.LOS_ANGELES_CLIPPERS',
            'Miami Heat' : 'Team.MIAMI_HEAT',
            'Minnesota Timberwolves' : 'Team.MINNESOTA_TIMBERWOLVES',
            'Oklahoma City Thunder' : 'Team.OKLAHOMA_CITY_THUNDER',
            'Orlando Magic' : 'Team.ORLANDO_MAGIC',
            'Portland Trail Blazers' : 'Team.PORTLAND_TRAIL_BLAZERS',
            'Toronto Raptors' : 'Team.TORONTO_RAPTORS',
            'Utah Jazz' : 'Team.UTAH_JAZZ',
            'Washington Wizards' : 'Team.WASHINGTON_WIZARDS',
            'Chicago Bulls' : 'Team.CHICAGO_BULLS',
            'Golden State Warriors' : 'Team.GOLDEN_STATE_WARRIORS',
            'Los Angeles Lakers' : 'Team.LOS_ANGELES_LAKERS',
            'Phoenix Suns' : 'Team.PHOENIX_SUNS',
            'Brooklyn Nets' : 'Team.BROOKLYN_NETS',
            'Denver Nuggets' : 'Team.DENVER_NUGGETS',
            'Detroit Pistons' : 'Team.DETROIT_PISTONS',
            'Indiana Pacers' : 'Team.INDIANA_PACERS',
            'Milwaukee Bucks' : 'Team.MILWAUKEE_BUCKS',
            'New York Knicks' : 'Team.NEW_YORK_KNICKS',
            'Philadelphia 76ers' : 'Team.PHILADELPHIA_76ERS',
            'Sacramento Kings' : 'Team.SACRAMENTO_KINGS',
            'San Antonio Spurs' : 'Team.SAN_ANTONIO_SPURS',
            'New Orleans Pelicans' : 'Team.NEW_ORLEANS_PELICANS',
            'Cleveland Cavaliers' : 'Team.CLEVELAND_CAVALIERS',
            'Memphis Grizzlies' : 'Team.MEMPHIS_GRIZZLIES'}

bet_data.replace({"team": naming_map}, inplace = True)
bet_data.replace({"opponent": naming_map}, inplace = True)

### Edit Dates
bet_data.loc[(bet_data['season'] == '2019-2020') & (bet_data['date'].str.len() < 7), 'date'] = bet_data['date'] + '2020'
bet_data['date'] = (pd.to_datetime(bet_data['date'], format = '%d.%m.%Y') - pd.to_timedelta(1, unit='d')).astype(str)

team_bet_data = bet_data[['date', 'team', 'opponent', 'odd_1', 'odd_2']]
team_bet_data.rename(columns = dict(zip(['date', 'team', 'opponent', 'odd_1', 'odd_2'], ['date', 'team', 'opponent', 'odd_team', 'odd_opponent'])), inplace = True)

opponent_bet_data = bet_data[['date', 'opponent', 'team', 'odd_2', 'odd_1']]
opponent_bet_data.rename(columns = dict(zip(['date', 'opponent', 'team', 'odd_2', 'odd_1'], ['date', 'team', 'opponent', 'odd_team', 'odd_opponent'])), inplace = True)

bed_data_conc = pd.concat([team_bet_data, opponent_bet_data], axis = 0)

# Edit - Merge All data

### Used Tables
#player_df
#player_season_stats
#player_df_team
#player_df_opponent
#team_avg_player_career
#opponent_avg_player_career
#bed_data_conc

all_features = player_df.copy()
all_features.drop(columns = ['date_lag_1',
                            'assists',
                            'attempted_field_goals',
                            'attempted_free_throws',
                            'attempted_three_point_field_goals',
                            'blocks',
                            'defensive_rebounds',
                            'game_score',
                            'location',
                            'made_field_goals',
                            'made_free_throws',
                            'made_three_point_field_goals',
                            'offensive_rebounds',
                            'outcome',
                            'personal_fouls',
                            'plus_minus',
                            'points_scored',
                            #'seconds_played', drop this later, use this in data cleaning
                            'steals',
                            'turnovers',
                            'rebounds',
                            'double_count',
                            'double_double',
                            'triple_double'], inplace = True)

player_season_stats.drop(columns = ['assists',
                            'attempted_field_goals',
                            'attempted_free_throws',
                            'attempted_three_point_field_goals',
                            'blocks',
                            'defensive_rebounds',
                            'games_played',
                            'games_started',
                            'made_field_goals',
                            'made_free_throws',
                            'made_three_point_field_goals',
                            'minutes_played',
                            'offensive_rebounds',
                            'personal_fouls',
                            'points',
                            'positions',
                            'steals',
                            'turnovers',
                            'played_this_season'], inplace = True)

all_features = pd.merge(all_features, player_season_stats, how = 'left', on = ['name', 'slug', 'season_end_year'])

all_features = pd.merge(all_features, player_df_team, how = 'left', on = ['team', 'date'])
all_features = pd.merge(all_features, player_df_opponent, how = 'left', on = ['opponent', 'date'])

all_features = pd.merge(all_features, team_avg_player_career, how = 'left', on = ['team', 'date'])
all_features = pd.merge(all_features, opponent_avg_player_career, how = 'left', on = ['opponent', 'date'])

all_features = pd.merge(all_features, bed_data_conc, how = 'left', on = ['date', 'team', 'opponent'])

# Data Cleaning

## Check the dist of fantasy points, eliminate outliers
plt.hist(all_features['fantasy_point'], bins = 50, density=True)
plt.show()

len(all_features[all_features['fantasy_point'] > 70]) / len(all_features)
len(all_features[all_features['fantasy_point'] < 0]) / len(all_features)

all_features = all_features.loc[all_features['fantasy_point'] <= 70]
all_features = all_features.loc[all_features['fantasy_point'] >= 0]

## Eliminate the records where the player got low points and did not play too much
len(all_features[(all_features['fantasy_point'] <= 0) & (all_features['seconds_played'] < 60)]) / len(all_features)
len(all_features[(all_features['fantasy_point'] <= 3) & (all_features['seconds_played'] < 60)]) / len(all_features)

all_features = all_features.loc[~((all_features['fantasy_point'] <= 3) & (all_features['seconds_played'] < 60))]

all_features.drop(columns = ['seconds_played'], inplace = True)

# Check missing data 

## There is a mismatch in betting data date. subtracting 1 solved. 
nancounts = all_features.isnull().sum()

# Data Save
        
## Drop NAs
all_features = all_features.dropna()

## reset index
all_features = all_features.reset_index(drop = True)

## Save data
all_features.to_parquet(target_path + 'all_data.parquet', index = False)


