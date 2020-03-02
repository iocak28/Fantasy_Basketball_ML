# Grid Search Parameters
max_depth = [i * 150 + 1 for i in range(4)]

min_samples_split = [2, 5]

min_samples_leaf = [1, 5]

max_features = ['auto']

random_grid = {'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_features': max_features}

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())
   
def walk_forward_validation(random_grid, train_n, train_labels, split = 1, split_ratio = 0.75, error_metric = 'rmse'):
    random_grid_df = expand_grid(random_grid)    
    error_list = []
    
    for i in range(len(random_grid_df)):
        error_list_sub = []
        data_size = int(len(train_n) / split)
        for j in range(split):
            training_features = train_n[int(j * data_size * split_ratio) : int((j + 1) * data_size * split_ratio)]
            training_labels = np.array(train_labels)[int(j * data_size * split_ratio) : int((j + 1) * data_size * split_ratio)]
            valid_features = train_n[int((j + 1) * data_size * split_ratio) : int((j + 1) * data_size)]
            valid_labels = np.array(train_labels)[int((j + 1) * data_size * split_ratio) : int((j + 1) * data_size)]

            ### Create model object
            params = dict(zip(random_grid_df.columns, random_grid_df.iloc[i]))
            regr = DecisionTreeRegressor(**params)
            
            ### Train the model using the training sets
            regr.fit(training_features, training_labels)
            
            ### Make predictions using the testing set
            valid_pred = regr.predict(valid_features)
            
            ### General Error & Bias
            valid_rmse = np.sqrt(np.mean(np.abs(valid_pred[:, None] - valid_labels)**2))
            
            error_list_sub.append(valid_rmse)
            print(i, j)
        
        error_list.append(np.mean(error_list_sub))

##### Trial
### Create model object
params = dict(max_depth = 10)
regr = DecisionTreeRegressor(**params)

### Train the model using the training sets
regr.fit(train_n, train_labels)

### Make predictions using the testing set
y_pred = regr.predict(test_n)

### The coefficients
### General Error & Bias
np.mean(np.abs(y_pred[:, None] - test_labels)) / np.mean(test_labels)
np.mean((y_pred[:, None] - test_labels)) / np.mean(test_labels)