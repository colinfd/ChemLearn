import sys
sys.path.insert(0,'../../')
from ML_prep import load_data, split_by_cols, train_prep_pdos, train_prep
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from catboost import CatBoostRegressor
from scipy.stats import norm,uniform
import numpy as np
import pickle

"""
Used to generate trained GB models with different train-test splits.
"""

np.random.seed(100)

df = pickle.load(open('../../data/pairs_pdos.pkl'))

if True:
    X,y = train_prep_pdos(df,include_WF=True,dE=0.1)
    model_type = 'pdos'
else:
    X,y = train_prep(df,include_WF=True)
    model_type = 'moments'

if False:
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
    split_type = 'comp_rxn'
elif True:
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['comp'])
    split_type = 'comp'
elif False:
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['ads_a','ads_b'])
    split_type = 'rxn'
else:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test,y_test,test_size=0.5)
    split_type = 'random'

model = CatBoostRegressor(loss_function='MAE',iterations=1.5e4)
model.fit(X_train,y_train,eval_set=(X_dev,y_dev))
model.save_model('%s_%s.cbm'%(model_type,split_type))

test_preds = model.predict(X_test)
MAE = np.abs(test_preds - y_test).mean()
print "Test Error for %s-%s = %.3f"%(model_type,split_type,MAE)
