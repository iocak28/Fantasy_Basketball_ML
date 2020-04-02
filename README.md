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

## What's New?
- The existing literature in the Fantasy Basketball area is limited and the majority of the studies are projects that are available online. The use of XGBoost was not a new approach in Fantasy basketball area. However, feature selection with XGBoost is a novel approach in the area. Our studies prove that XGBoost can be successfully used in dimensionality reduction as well.

- The method ‘Optimized Moving Average Features’ is a totally new and unique approach in Fantasy Basketball. Using this approach we optimized and automated the feature extraction process. Instead of a feature extraction process that relies solely on industry knowledge, we proposed a simple but effective machine learning approach. In the existing literature, using weighted moving average features is not a novel thing.  However, the weights in the existing literature are based on intuitions and industry knowledge.

- The importance of cross-validation in tuning is an undisputed fact. However, the existing literature does not focus on Walk-Forward validation using timeseries splitting. Walk-forward validation is an approach where we could apply to sports forecasting, demand forecasting, and the areas where the time dimension is important. In this project, we successfully used this approach to tune the model hyperparameters.

## Codes
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
