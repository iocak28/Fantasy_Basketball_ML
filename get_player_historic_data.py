# PREPARE DATA FOR NBA FANTASY
import pandas as pd
import numpy as np
import time
from basketball_reference_web_scraper import client

import os
import re
import time
import glob
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook as tqdm

from difflib import SequenceMatcher

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

## Player Season Data
    
def find_player_season_stats(year_start, year_end):
    
    years = [i for i in range(year_start, year_end + 1)]
    player_stats = pd.DataFrame()
    
    for i in years:
        temp = pd.DataFrame(client.players_season_totals(season_end_year = i))
        temp['season_end_year'] = i
        
        player_stats = pd.concat(
                [player_stats, temp], 
                axis = 0)
    
    player_stats = player_stats.reset_index()
    player_stats = player_stats.drop(columns = ['index'])
    
    return player_stats

# get player salary data

#### function Scraping DraftKings salary data from RotoGuru.com
#### reference: https://github.com/KengoA/fantasy-basketball
def scrape_fantasy_salary_on_a_date(date):
    url_roto = "http://rotoguru1.com/cgi-bin/hyday.pl?mon={month}&day={day}&year={year}&game=dk"  
    
    teams, positions, players, starters, salaries = [], [], [], [], []
    
    soup_rows = 0
    
    # try until you get a result
    while True:
        try:
            url_date = url_roto.format(month=date[5:7],day=date[8:10],year=date[0:4])
            soup = BeautifulSoup(urlopen(url_date),'lxml')
        
            #Check if there were any games on a given date
            soup_table = soup.find('body').find('table', border="0", cellspacing="5")
        
            soup_rows = soup_table.find_all('tr')
        except:
            'error trying again in 3 secs'
            time.sleep(3)

        if soup_rows != 0:
            break

    # get data in a loop
    for row in soup_rows:
        if row.find('td').has_attr('colspan') == False:
            if row.find('a').get_text() != '':

                position = row.find_all('td')[0].get_text()

                player_tmp = row.find('a').get_text().split(", ")
                player = player_tmp[1] + ' ' + player_tmp[0]

                starter_tmp = row.find_all('td')[1].get_text()

                if '^' in starter_tmp:
                    starter = 1
                else:
                    starter =0

                salary_tmp = row.find_all('td')[3].get_text()
                salary = re.sub('[$,]', '', salary_tmp)

                team = row.find_all('td')[4].get_text()

                positions.append(position)
                players.append(player)
                starters.append(starter)
                salaries.append(salary)
                teams.append(team)

    df = pd.DataFrame({'Date': [date for i in range(len(players))], 
                       'Team': [team.upper() for team in teams],
                       'Starter': starters,
                       'Pos': positions,
                       'Name': players,
                       'Salary': salaries})

    df = df.loc[:,['Date','Team','Pos','Name','Starter','Salary']]
    return df

####################
# MAIN

#### Get Player names, slugs between the seasons ending 2010 to 2020
player_slug_year = find_player_name_active_years(2010, 2020)

## Get and Save player data
for i in range(np.ceil(len(player_slug_year) / 100).astype(int)):
    temp = player_slug_year.loc[100 * i : 100 * (i + 1) - 1,:]
    player_game_raw_data, error_happened = get_player_box_scores(temp)
    player_game_raw_data.to_csv(f'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/sample_data/player_data/d_{i}_error{error_happened}.csv')

## Player Season Data
player_year_stats_df = find_player_season_stats(1995, 2020)
player_year_stats_df.to_csv('C:/Users/iocak/OneDrive/Masaüstü/git/Fantasy_Basketball_ML/sample_data/player_season_data/player_season.csv', index = False)

#### Call salary data scraper
data_path = 'C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/sample_data/player_data/'
files = os.listdir(data_path)    

## find all dates
player_df = pd.DataFrame()

for i in files:
    player_game_raw_data = pd.read_csv(data_path + i)
    player_df = pd.concat([player_df, player_game_raw_data], axis = 0)
    print(i)
    
dates = list(player_df['date'].drop_duplicates().sort_values())
del player_df

## get salary data
salary_frame = pd.DataFrame()
for j in dates:
    if j > '2014-09-30':
        try:
            temp = scrape_fantasy_salary_on_a_date(j)
            salary_frame = pd.concat([salary_frame, temp], axis = 0)
            print(j)
        except:
            print(f'error: {j}')

salary_frame.rename(columns = {'Date' : 'date', 
                               'Team' : 'team', 
                               'Name' : 'name', 
                               'Starter' : 'starter', 
                               'Salary' : 'salary'}, inplace = True)

## merge salary data to raw files

for i in files:
    player_game_raw_data = pd.read_csv(data_path + i)    
    merged_data = pd.merge(player_game_raw_data, 
                           salary_frame[['date', 'name', 'starter', 'salary']], 
                           how = 'left', 
                           on = ['date', 'name'])

    # mark missing data
    merged_data.loc[merged_data['starter'].isnull(), 'starter'] = -1
    merged_data.loc[merged_data['salary'].isnull(), 'salary'] = -1
    merged_data.to_csv(data_path + i, index = False)
    print('Process ', i)
    

        # This part is incomplete, it is to be completed
        
        ## for NAs conduct string match 
        ## I SHOULD ADD TEAM MATCH HERE AS WELL
#        raw_names = list(merged_data[merged_data['salary'].isnull()]['name'].drop_duplicates())
#        salary_names = list(main_frame['name'].drop_duplicates())
#        
#        similarity_list = []
#        
#        for x in raw_names:
#            temp_list = []
#            for y in salary_names:
#                temp_list.append(SequenceMatcher(None, x, y).ratio())
#            similarity_list.append(temp_list)
#        
#        similarity_df = pd.DataFrame(similarity_list)
#        
#        similarity_df.rename(columns = dict(zip([i for i in range(len(salary_names))], salary_names)), inplace = True)
#        similarity_df.rename(index = dict(zip([i for i in range(len(raw_names))], raw_names)), inplace = True)
#        
#        similarity_df['None'] = 0.65
#        corrections = pd.DataFrame(similarity_df.idxmax(axis = 1)).reset_index()
#        
#        for x in range(len(corrections)):
#            
#            if corrections.iloc[x, 1] != 'None':
#                merged_data[merged_data['name'] == corrections.iloc[x, 0]]
#            
#                for y in range(len(merged_data[merged_data['name'] == corrections.iloc[x, 0]])):
#                    merged_data.loc[merged_data['name'] == corrections.iloc[x, 0]][158]
        
        