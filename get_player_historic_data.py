# PREPARE DATA FOR NBA FANTASY

import pandas as pd
import numpy as np
import time
from basketball_reference_web_scraper import client

## Find Players Names and Active Years From 2010

def find_player_name_active_years(year_start, year_end):
    '''
    Find player name, name codes (slug), withing the entered year intervals.
    
    Inputs:
        year_start: Ending year of the first season, int
        year_end: Ending year of the last season, int
    
    Outputs:
        player_active_years: the table of name, slug, and season_end_year, pandas dataframe
        
    The output of this function will be used to query player statistics.
    '''
    
    years = [i for i in range(year_start, year_end + 1)]
    player_active_years = pd.DataFrame()
    
    for i in years:
        temp = pd.DataFrame(client.players_season_totals(season_end_year = i))
        
        player_year_name = temp[['name', 'slug']]
        player_year_name['season_end_year'] = i
        
        player_active_years = pd.concat(
                [player_active_years, player_year_name], 
                axis = 0)
    
    ### Some players have multiple records due to in-season transfers, remove them
    player_active_years = player_active_years.drop_duplicates()
    dups = player_active_years.groupby(['slug', 'season_end_year']).count()
    
    if len(player_active_years) != len(dups):
        print('Warning: There are DUPLICATE records in player_active_years')
    
    player_active_years = player_active_years.reset_index()
    player_active_years = player_active_years.drop(columns = ['index'])
    
    return player_active_years


## Get Player Box Scores

def get_player_box_scores(player_slug_year):
    '''
    Get game by game player box scores.
    
    Input:
        player_slug_year: pandas data frame of name, slug, and season_end_year 
                          (see the function find_player_name_active_years for more info)

    Returns:
        player_game_raw_data: game by game box score data for every player queried.
        
    Returns raw data.
    '''
    
    player_slug_year = player_slug_year.reset_index()
    player_slug_year = player_slug_year.drop(columns = ['index'])
    
    player_game_raw_data = pd.DataFrame()
    counter = 0
    error_counter = 0
    error_gate = 0
    
    while counter < len(player_slug_year):
        player_name = player_slug_year['name'][counter]
        identifier = player_slug_year['slug'][counter]
        year = player_slug_year['season_end_year'][counter]
        
        try:
            temp = pd.DataFrame(client.regular_season_player_box_scores(
                    player_identifier = identifier,
                    season_end_year = year))
            temp['name'] = player_name
            temp['slug'] = identifier
            temp['season_end_year'] = year
            
            player_game_raw_data = pd.concat(
                    [player_game_raw_data, temp], 
                    axis = 0)
        except:
            temp = pd.DataFrame()
        
        if len(temp) > 0:
            print(f'Player {counter + 1} out of {len(player_slug_year)} completed')
            counter += 1
        else:
            print(f'!!Player {counter + 1} out of {len(player_slug_year)} retrying')
            from basketball_reference_web_scraper import client
            time.sleep(5)
            error_counter += 1
            
            if error_counter == 9:
                print(f'!!SKIPPED Player {counter + 1} out of {len(player_slug_year)} retrying')
                counter += 1
                error_counter = 0
                error_gate = 1
        
    player_game_raw_data = player_game_raw_data.reset_index()
    player_game_raw_data = player_game_raw_data.drop(columns = ['index'])
    
    return player_game_raw_data, error_gate

#### Get Player names, slugs between the seasons ending 2010 to 2020
player_slug_year = find_player_name_active_years(2010, 2020)


## Get and Save player data
for i in range(np.ceil(len(player_slug_year) / 100).astype(int)):
    temp = player_slug_year.loc[100 * i : 100 * (i + 1) - 1,:]
    player_game_raw_data, error_happened = get_player_box_scores(temp)
    player_game_raw_data.to_csv(f'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/sample_data/player_data/d_{i}_error{error_happened}.csv')
