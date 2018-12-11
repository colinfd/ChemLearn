import sys
sys.path.insert(0,'../../')
from ML_prep import train_prep_pdos, split_by_cols
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

np.random.seed(100)

df = pickle.load(open('../../data/pairs_pdos.pkl'))
X,y = train_prep_pdos(df,include_WF=False,stack=True)

for split_type in ['comp','rxn','comp_rxn','random']:
    print split_type
    if split_type == 'comp':
        cols = ['comp']
    elif split_type == 'rxn':
        cols = ['ads_a','ads_b']
    elif split_type == 'comp_rxn':
        cols = ['comp','ads_a','ads_b']
    else:
        cols = None

    if cols != None:
        X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,cols)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
        X_dev, X_test, y_dev, y_test = train_test_split(X_test,y_test,test_size=0.5)
    
    np.save('X_%s_train.npy'%split_type,X_train)
    np.save('X_%s_dev.npy'%split_type,X_dev)
    np.save('X_%s_test.npy'%split_type,X_test)
    np.save('y_%s_train.npy'%split_type,y_train)
    np.save('y_%s_dev.npy'%split_type,y_dev)
    np.save('y_%s_test.npy'%split_type,y_test)
