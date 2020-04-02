# Fantasy Basketball Prediction Using Machine Learning

![Basketball-reference sample data](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/img/breference_sample.png)

## Summary
The growing power of the internet created many new industries after the 1990s. This phenomenon also created a new area called Fantasy 
Basketball. On a given night, users draft real basketball players and earn points based on the real performances of these drafts. In this 
multi-billiondollar industry, making accurate predictions for player performances or fantasy points is crucial. This project aims to use 
machine learning for this purpose. In this project, we used different machine learning models in feature extraction, feature selection 
and, prediction processes. Such models include, XGBoost, RandomForest, AdaBoost, Artificial Neural Networks, Linear Regression and Lasso. 
Comparing the performances of these different models we found that XGBoost can be successfully used for feature selection and prediction 
processes in the Fantasy Basketball prediction area. Additionally, we developed a feature extraction method that optimizes the weights of 
moving average features using Linear Regression.

For detailed information please refer to the [report](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/Report%20-%20Fantasy%20Basketball%20Prediction%20Using%20Machine%20Learning.pdf).

## Data Source
The website basketball-reference.com offers detailed player summaries for every player in the NBA for games dating back to the 1940s. We have scraped the game logs for each player from basketball-reference to gather relevant nightly scoring summaries using the “basketball-referenceweb- scraper” Python package. We also scraped betting data from betexplorer.com and obtained daily betting lines that represent the relative powers of the teams playing in the NBA. Also, using rotoguru.net we scraped daily fantasy salary data of the basketball players. Finally, we compiled these sources into a dataset. The raw data consisted of columns such as date, player name, team, opponent, assists, fantasy points, betting odds, fantasy salary, etc. There are many columns that were not mentioned here but all of these columns are indicators of a player’s performance on a given game night.

## What's New?
- The existing literature in the Fantasy Basketball area is limited and the majority of the studies are projects that are available online. The use of XGBoost was not a new approach in Fantasy basketball area. However, feature selection with XGBoost is a novel approach in the area. Our studies prove that XGBoost can be successfully used in dimensionality reduction as well.

![XGB Features](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/img/recent_feat_imp.png)

- The method ‘Optimized Moving Average Features’ is a totally new and unique approach in Fantasy Basketball. Using this approach we optimized and automated the feature extraction process. Instead of a feature extraction process that relies solely on industry knowledge, we proposed a simple but effective machine learning approach. In the existing literature, using weighted moving average features is not a novel thing.  However, the weights in the existing literature are based on intuitions and industry knowledge.

![Steps](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/img/steps.png)

- The importance of cross-validation in tuning is an undisputed fact. However, the existing literature does not focus on Walk-Forward validation using timeseries splitting. Walk-forward validation is an approach where we could apply to sports forecasting, demand forecasting, and the areas where the time dimension is important. In this project, we successfully used this approach to tune the model hyperparameters.

![Time-Series Cross Validation](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/img/walkforward.png)
<img src="https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/img/walkforward.png" width="1000">

## Codes
You can find the steps that were mentioned in this project below:

1. [get_player_historic_data.py](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/get_player_historic_data.py):
Get player stats and fantasy salary data
2. [get_betting_data.py](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/get_betting_data.py):
Get betting data
3. [data_prep.py](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/data_prep.py):
Merge raw data, feature extraction, weighted feature optimization
4. [tuning.py](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/tuning.py):
Tuning structure for different models
5. [modeling_trial.py](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/modeling_trial.py):
Modeling structure for different models
6. [xgb_feature_select.py](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/xgb_feature_select.py):
Feature selection with XGBoost feature importance
7. [model_xgb_with_new_features.py](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/model_xgb_with_new_features.py):
Try XGBoost model with optimized weighted moving average features
8. [data_analysis.py](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/data_analysis.py):
Preliminary and post data analyses
9. [tuning_code_v3_xgbselectedfeat_newfeat_smartlag_wma.ipynb](https://github.com/iocak28/Fantasy_Basketball_ML/blob/master/source_codes/tuning_code_v3_xgbselectedfeat_newfeat_smartlag_wma.ipynb):
Tuning code for XGBoost.

## References
1. “Daily Fantasy Sports For Cash.” DraftKings, draftkings.
com.
2. “Daily Fantasy Football, MLB, NBA, NHL Leagues
for Cash.” FanDuel, fanduel.com.
3. Smith, Brian, et al. “Decision Making in Online
Fantasy Sports Communities.” Interactive Technology
and Smart Education, vol. 3, no. 4, 2006, pp.
347–360., DOI:10.1108/17415650680000072.
4. Miller, Bennett, director. Moneyball. Universal,
2011.
5. “Basketball Statistics and History.” Basketball,
www.basketball-reference.com/.
6. “Basketball-Reference-Web-Scraper.” PyPI,
pypi.org/project/basketball-reference-web-scraper/.
7. Barry, et al. Beating DraftKings at Daily Fantasy
Sports.
8. Hermann, Eric. “Machine Learning Applications in
Fantasy Basketball.” (2015).
9. Shivakumar, Shreyas. Learning to Turn Fantasy Basketball
Into Real Money Introduction to Machine
Learning. shreyasskandan.github.io/.
10. Hoerl, Arthur E., and Robert W. Kennard. “Ridge
Regression: Biased Estimation for Nonorthogonal
Problems.” Technometrics 42 (2000): 80-86.
8
11. Tibshirani, Robert. “Regression Shrinkage and Selection
Via the Lasso.” Journal of the Royal Statistical
Society: Series B (Methodological), vol.
58, no. 1, 1996, pp. 267–288., doi:10.1111/j.2517-
6161.1996.tb02080.x.
12. Arao, Kengo. NBA Player Performance
Prediction and Lineup Optimization.
github.com/KengoA/fantasy-basketball.
13. Chen, Tianqi, and Carlos Guestrin. “XGBoost.” Proceedings
of the 22nd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining KDD 16, 2016, doi:10.1145/2939672.2939785.
14. Yoav Freund and Robert E. Schapire. Experiments
with a new boosting algorithm. In Machine Learning:
Proceedings of the Thirteenth International
Conference, pages 148–156, 1996
15. Breiman, Leo (2001). “Random Forests”.
Machine Learning. 45 (1): 5–32.
doi:10.1023/A:1010933404324.
16. “Betexplorer.” BetExplorer Soccer Stats -
Results, Tables, Soccer Stats and Odds,
www.betexplorer.com/.
17. “Daily Blurbs.” RotoGuru, www.rotoguru.net/.
