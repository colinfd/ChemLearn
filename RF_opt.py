import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model,ensemble,gaussian_process
from sklearn.model_selection import train_test_split
from ML_prep import train_prep_pdos,split_by_cols,train_prep,add_noise
from matplotlib.pyplot import cm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
from scipy.stats import randint as sp_randint

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mae = np.mean(errors)
    print('Model Performance')
    print('Average Error: {:0.4f} eV.'.format(mae))
    return mae

if __name__ == '__main__':
    df = pickle.load(open('data/pairs_pdos.pkl'))

    features = 'moments'# #pdos,'moments'
    bayes = True

    #Feature Selection
    if features == 'moments':
        X,y = train_prep(df)
    elif features == 'pdos':
        X,y = train_prep_pdos(df,stack=False,include_WF=False,dE=0.1)
    
    X_train,X_dev,X_test,y_train,y_dev,y_test,groups = split_by_cols(df,X,y,['comp','ads_a','ads_b'],ret_groups=True)
	    
    rf = ensemble.RandomForestRegressor(n_estimators=100)

    group_kfold = GroupKFold(n_splits=3)

    #print(X_train.shape[1]),np.sqrt(X_train.shape[1])
    if bayes:
        random_grid = {#'n_estimators': (5,100),
                   'max_features': (int(np.sqrt(X_train.shape[1])),X_train.shape[1]),
                   'max_depth': (5, 50),
                   'min_samples_split': (2,10),
                   'min_samples_leaf': (2,5),
                   'bootstrap': [True,False]}

        rf_random = BayesSearchCV(rf, 
            random_grid,
            n_iter = 50, 
            cv = group_kfold, 
            verbose=10, 
            random_state=42,
            n_jobs = -1,
            scoring='neg_mean_absolute_error',
            )
    else:
        random_grid = {'n_estimators': sp_randint(5, 100),
                   'max_features': sp_randint(5,X_train.shape[1]//2),
                   'max_depth': sp_randint(5, 50),
                   'min_samples_split': sp_randint(2,10),
                   'min_samples_leaf': sp_randint(2,5),
                   'bootstrap': [True,False]}

        rf_random = RandomizedSearchCV(estimator=rf, 
            param_distributions = random_grid, 
            n_iter = 100, 
            cv = group_kfold, 
            verbose=10, 
            random_state=42, 
            n_jobs = -1,
            scoring='neg_mean_absolute_error',
            )

    rf_random.fit(X_train, y_train,groups=groups)
    print(features,rf_random.best_estimator_)
    evaluate(rf_random.best_estimator_, X_test, y_test)
