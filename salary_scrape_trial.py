import os
import re
import time
import glob

import numpy as np
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook as tqdm

#from constants import DATA_DIR, SEASON_DATES, SECONDS_SLEEP

date_list = list(dataset['date'].drop_duplicates().sort_values().reset_index(drop = True))
date = '2014-28-10'

url_roto = "http://rotoguru1.com/cgi-bin/hyday.pl?mon={month}&day={day}&year={year}&game=dk"

for date in date_list:
    print(date)
    teams, positions, players, starters, salaries = [], [], [], [], []

    url_date = url_roto.format(month=date[5:7],day=date[8:10],year=date[0:4])
    soup = BeautifulSoup(urlopen(url_date),'lxml')

    #Check if there were any games on a given date
    soup_table = soup.find('body').find('table', border="0", cellspacing="5")

    soup_rows = soup_table.find_all('tr')

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
    
    
    
    
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

date = '2014-28-10'


month, day, year = date[5:7], date[8:10], date[0:4]
page = f"http://rotoguru1.com/cgi-bin/hyday.pl?mon={month}&day={day}&year={year}&game=dk"        

# Go to Page
response = requests.get(page)
soup = BeautifulSoup(response.text, 'html.parser')
rows = soup.table.find(border="0", cellspacing="5")
rows.find_all('td')
