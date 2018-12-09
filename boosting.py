from ML_prep import load_data, split_by_cols, train_prep_pdos
from catboost import CatBoostRegressor
import numpy as np
import pickle

type = 'pdos'
np.random.seed(100)

df = pickle.load(open('data/pairs_pdos.pkl'))
X,y = train_prep_pdos(df,include_WF=False,dE=0.1)
X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
#X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['comp'])
#X_train, X_dev, X_test, y_train, y_dev, y_test = split_by_cols(df,X,y,['ads_a','ads_b'])

model = CatBoostRegressor(loss_function='MAE',iterations=1e4,depth=3)
model.fit(X_train,y_train,eval_set=(X_dev,y_dev))
