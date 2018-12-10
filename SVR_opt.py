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
from scipy.stats import uniform as sp_uniform
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
    bayes = False

    #Feature Selection
    if features == 'moments':
        X,y = train_prep(df)
    elif features == 'pdos':
        X,y = train_prep_pdos(df,stack=False,include_WF=True,dE=0.1)
    
    np.random.seed(42)
    X_train,X_dev,X_test,y_train,y_dev,y_test,groups = split_by_cols(df,X,y,['comp','ads_a','ads_b'],ret_groups=True)
	    
    model = SVR()

    group_kfold = GroupKFold(n_splits=3)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    
    base_gamma = 1.0/X_train.shape[1]/X_train.std()
    print(base_gamma)
    
    if bayes:
        random_grid = {
                'alpha': (1,5,'log-uniform'),
                'gamma':(),
                'kernel':['rbf']
                   }

        kr_random = BayesSearchCV(kr, 
            random_grid,
            n_iter = 500, 
            cv = group_kfold, 
            verbose=10, 
            random_state=42,
            n_jobs = -1,
            scoring='neg_mean_absolute_error',
            )
    else:
        random_grid = {
                        'C': (1e-3,1e-2,1e-1,1,10,100),
                        #'gamma':[1e-23,1e-24,1e-22,1e-21],#(0.001,0.01,0.1,1,10,100),
                        'gamma':[base_gamma*0.1, base_gamma,base_gamma*10],#(0.001,0.01,0.1,1,10,100),
                        'kernel':['rbf','linear','poly'],#,'cosine'],
                        'epsilon':sp_uniform(0.1,0.9)
                        }

        model_random = RandomizedSearchCV(estimator=model, 
            param_distributions = random_grid, 
            n_iter = 300, 
            cv = group_kfold, 
            verbose=10, 
            random_state=42, 
            n_jobs = -1,
            scoring='neg_mean_absolute_error',
            )

    model_random.fit(X_train, y_train,groups=groups)
    print(features,model_random.best_estimator_)
    X_dev = scaler.transform(X_dev)
    evaluate(model_random.best_estimator_, X_dev, y_dev)
    evaluate(model_random.best_estimator_, X_train, y_train)
    
