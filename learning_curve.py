import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model,ensemble
from ML_prep import train_prep_pdos,split_by_cols,train_prep,add_noise
from matplotlib.pyplot import cm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
from scipy.stats import randint as sp_randint
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mae = np.mean(errors)
    #print('Model Performance')
    #print('Average Error: {:0.4f} eV.'.format(mae))
    return mae


df = pickle.load(open('data/pairs_pdos.pkl'))

features = ['moments','pdos']# #pdos,'moments'
models = ['lr','rf','krr']

for feature in features:
    for m in models:
        if ( m == 'lr' or m == 'krr' ) and feature == 'pdos':
            continue
        
        if feature == 'moments':
            X,y = train_prep(df)
        elif feature == 'pdos':
            X,y = train_prep_pdos(df,stack=False,include_WF=True,dE=0.1)
        
        np.random.seed(42)
        X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
        if m == 'rf':
            if feature == 'pdos':	    
                model = ensemble.RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=18,
                  max_features='sqrt', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=5,
                  min_weight_fraction_leaf=0.0, n_estimators=26, n_jobs=None,
                  oob_score=False, random_state=None, verbose=0, warm_start=False)
            elif feature == 'moments':
                model = ensemble.RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=18,
                    max_features=11, max_leaf_nodes=None, min_impurity_decrease=0.0,
                    min_impurity_split=None, min_samples_leaf=2,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=None, oob_score=False,
                    random_state=None, verbose=0, warm_start=False)
        elif m == 'lr':
            model = linear_model.LinearRegression()
        elif m == 'krr':
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_dev = scaler.transform(X_dev)
            model = KernelRidge(alpha=0.00089, coef0=1, degree=3, gamma=0.04545, kernel='rbf', kernel_params=None)

        num_examples = X_train.shape[0]
        mae_dev = np.zeros(10)
        mae_train = np.zeros(10)
        for i in range(10):
            X_train_i = X_train[0:num_examples//10*(i+1)]
            y_train_i = y_train[0:num_examples//10*(i+1)]
            model.fit(X_train_i, y_train_i)
            mae_dev[i] = evaluate(model, X_dev, y_dev)
            mae_train[i] = evaluate(model, X_train_i, y_train_i)
        np.save('learning_curve/mae_train_%s_%s.npy'%(m, feature),mae_train)
        np.save('learning_curve/mae_dev_%s_%s.npy'%(m, feature),mae_dev)
        
        plt.plot(mae_dev,label=m+' '+feature+' dev')
        plt.plot(mae_train,label=m+' '+feature+' train')
plt.legend()
plt.ylim(0,1.5)
plt.savefig('learning_curve/learning_curve.pdf')
