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

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mae = np.mean(errors)
    #print('Model Performance')
    #print('Average Error: {:0.4f} eV.'.format(mae))
    return mae


df = pickle.load(open('data/pairs_pdos.pkl'))

features = ['pdos']#,'pdos']# #pdos,'moments'
models = ['rf']#['lr','rf']

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

for feature in features:
    for m in models:
        if m == 'lr' and feature == 'pdos':
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

        num_examples = X_train.shape[0]
        
        model.fit(X_train, y_train)
        mae_dev = evaluate(model, X_dev, y_dev)
        mae_train = evaluate(model, X_train, y_train)

        ax.axvline(200,ls='--',c='r',alpha=0.5)
        ax.axvline(552,ls='--',c='r',alpha=0.5)
        ax.bar(range(len(model.feature_importances_)),model.feature_importances_)
        
        #plt.bar(range(len(model.feature_importances_)),model.feature_importances_)


        #np.save('learning_curve/mae_train_%s_%s.npy'%(m, feature),mae_train)
        #np.save('learning_curve/mae_dev_%s_%s.npy'%(m, feature),mae_dev)
        
        #plt.plot(mae_dev,label=m+' '+feature+' dev')
        #plt.plot(mae_train,label=m+' '+feature+' train')
#plt.legend()
#plt.ylim(0,1.5)
#plt.savefig('feature_importance/feature_importance_%s.pdf'%(feature))

#plt.clf()

if feature == 'pdos':
    #for i in range(20):
    ax2 = ax.twinx()
    ax2.plot(X_train.mean(axis=0),lw=.1)
    ax2.set_ylim((0,7))
    #plt.plot(X_train.mean(axis=0))

    plt.savefig('feature_importance/X1_%s.pdf'%(feature))

