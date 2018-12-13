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
import PlotFormat

"""
Plots feature importance of tree based methods using the moments as feature vectors
"""

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mae = np.mean(errors)
    #print('Model Performance')
    #print('Average Error: {:0.4f} eV.'.format(mae))
    return mae

df = pickle.load(open('data/pairs_pdos.pkl'))

feature = 'moments'
models = ['rf','boost']

colors = PlotFormat.muted_colors

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)

shift = 0
for m in models:
    
    X,y = train_prep(df)
    
    np.random.seed(100)

    X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
    if m == 'rf':
        model = ensemble.RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=18,
                max_features=11, max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=2,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=100, n_jobs=None, oob_score=False,
                random_state=None, verbose=0, warm_start=False)

        model.fit(X_train, y_train)
        feature_imp = model.feature_importances_
    elif m == 'boost':
        feature_imp =  np.load('feature_importance/boost_moments.npy')
    
    num_examples = X_train.shape[0]
    feature_imp/=feature_imp.max()
    #mae_dev = evaluate(model, X_dev, y_dev)
    #mae_train = evaluate(model, X_train, y_train)

    plt.bar(np.arange(0,len(feature_imp),1)+shift,feature_imp,width = 0.25,color=colors[m])
    shift+=0.25

plt.gca().tick_params(axis='both',which='both',top=False,bottom=False,labeltop=False,labelbottom=False,labelleft=False,left=False)
plt.savefig('feature_importance/feature_importance_%s.pdf'%(feature))
