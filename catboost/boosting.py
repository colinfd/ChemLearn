import sys
sys.path.insert(0,'../')
from ML_prep import load_data, split_by_cols, train_prep_pdos, train_prep
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from catboost import CatBoostRegressor
from scipy.stats import norm,uniform
import numpy as np
import pickle

"""
General script for training GB models with Catboost
"""

type = 'pdos'
np.random.seed(100)

df = pickle.load(open('../data/pairs_pdos.pkl'))
X,y = train_prep_pdos(df,include_WF=True,dE=0.1)
#X,y = train_prep(df,include_WF=True)
X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
#X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['comp'])
#X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['ads_a','ads_b'])

"""
frac = 1
m = X_train.shape[0]
X_train = X_train[:int(frac*m)]
y_train = y_train[:int(frac*m)]
model = CatBoostRegressor(loss_function='MAE',iterations=3e4,learning_rate=0.03)#,logging_level='Silent')#,use_best_model=True)
model.fit(X_train,y_train,eval_set=(X_dev,y_dev))

exit()
"""

model = CatBoostRegressor(loss_function='MAE',iterations=1.2e4)
model.fit(X_train,y_train,eval_set=(X_dev,y_dev))
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

np.save('boost_y_test.npy',y_test)
np.save('boost_y_test_pred.npy',y_test_pred)
np.save('boost_y_train_pred.npy',y_train_pred)

exit()


if False:
    param_distributions = {
            'learning_rate':uniform(0.01,0.1),
            'depth':range(1,10),
            'l2_leaf_reg':uniform(loc=0,scale=6),
            'random_strength':uniform(loc=0,scale=1)
            }

    bayes_param_distributions = {
            'learning_rate':(0.01,0.1,'uniform'),
            'depth':range(1,10),
            'l2_leaf_reg':range(0,6),
            'random_strength':(0,1,'uniform')
            }

    X = np.vstack((X_train,X_dev))
    y = np.append(y_train,y_dev)
    #model2 = RandomizedSearchCV(model,param_distributions,n_iter=50,scoring='neg_mean_absolute_error',cv=dummy_cv,verbose=2)
    model2 = BayesSearchCV(model,bayes_param_distributions,n_iter=50,scoring='neg_mean_absolute_error',cv=dummy_cv,verbose=10,fit_params={'eval_set':(X_dev,y_dev)})
    model2.fit(X,y)



##learning curve
n = 10
m = X_train.shape[0]
train_MAE = []
dev_MAE = []
for i in range(n):
    model = CatBoostRegressor(loss_function='MAE',iterations=1e4)
    Xi = X_train[:(i+1)*m/n]
    yi = y_train[:(i+1)*m/n]
    model.fit(Xi,yi,eval_set=(X_dev,y_dev))
    train_MAE.append(np.abs(model.predict(Xi) - yi).mean())
    dev_MAE.append(np.abs(model.predict(X_dev) - y_dev).mean())
