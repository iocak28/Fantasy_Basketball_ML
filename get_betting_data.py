import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

seasons = ['2009-2010', '2010-2011', '2011-2012','2012-2013', '2013-2014', '2014-2015', '2015-2016', 
           '2016-2017', '2017-2018', '2018-2019', '2019-2020']
counter = 0
yearly_odds = pd.DataFrame()

while counter < len(seasons):

    try:
        season = seasons[counter]
        page = f'https://www.betexplorer.com/basketball/usa/nba-{season}/results/'
        
        
        # Go to Season Entrance Page
        response = requests.get(page)
        soup = BeautifulSoup(response.text, 'html.parser')
        temp = soup.find_all('a', {'class': 'list-tabs__item__in'})
        
        # Find the link for main
        main_index = ['Main' in box for box in temp]
        extensions = [box['href'] for box in temp]
        main_extension = extensions[main_index.index(True)]
        
        # Click on Main - Get Inside
        response = requests.get(page + main_extension + '&month=all')
        soup = BeautifulSoup(response.text, 'html.parser')
        temp = soup.find_all('tr')
        
        # odds table
        odds_df = pd.DataFrame()
        
        i = 0
        for i in range(len(temp) - 1):
            game_container = temp[i]
            
            children = list(game_container.children)
            
            sides = children[0].text
            score = children[1].text
            
            odd_1 = str(children[2])
            str_start = odd_1.find('data-odd=') + 10
            str_end = odd_1.find('"></')
            odd_1_final = odd_1[str_start:str_end]
            
            odd_2 = str(children[3])
            str_start = odd_2.find('data-odd=') + 10
            str_end = odd_2.find('"></')
            odd_2_final = odd_2[str_start:str_end]
            
            date = children[4].text
            
            temp_dict = {'season' : season,
                         'sides' : sides,
                         'score' : score,
                         'odd_1' : odd_1_final,
                         'odd_2' : odd_2_final,
                         'date' : date}
            
            # join games to that season's df
            odds_df = pd.concat([odds_df, pd.DataFrame(temp_dict, index = [i])], 
                                 axis = 0)
        
        # join season betting data to main df
        yearly_odds = pd.concat([yearly_odds, odds_df], axis = 0)
        print(f'Season: {season} completed')
        counter = counter + 1
    except:
        print(f'!!! Season: {season} Error')
        time.sleep(5)
    
yearly_odds.to_csv('C:/Users/iocak/OneDrive/Masaüstü/WI20/ECE 271B/Project/sample_data/betting_data/odds.csv')