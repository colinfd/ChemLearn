import pickle
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
import seaborn as sns
from sklearn import linear_model,ensemble,gaussian_process
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow import keras
from ML_prep import train_prep_pdos,split_by_cols,train_prep
import PlotFormat
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

"""
Plots parity plots for all models except CNN
For CNN parity see convnet/eval.py
"""

colors = PlotFormat.muted_colors

def evaluate(y_train,train_preds,y_test,test_preds,model,features):
    """
    Get MAE, plot parity and optional plot histogram of error
    """
    
    plt.figure(figsize=(3.5,3))
    train_mae = np.abs(y_train-train_preds).mean()
    test_mae = np.abs(y_test-test_preds).mean()
    print(model, split)
    print('Test MAE: %4.2f'%(test_mae))
    print('Train MAE: %4.2f'%(train_mae))
    plt.plot(y_train,train_preds,'.',c='grey',ms=3,label='Train MAE: %4.2f'%train_mae,alpha=1)
    plt.plot(y_test,test_preds,'.',c=colors[model],ms=3,label='Test MAE: %4.2f'%test_mae,alpha=1)
    #plt.title('%s: Train MAE = %.2f eV; Test MAE = %.2f eV'%(model,np.abs(y_train-train_preds).mean(),np.abs(y_test-test_preds).mean()),fontsize=10)

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    
    alim = (min(xlim[0],ylim[0]),max(xlim[1],ylim[1]))

    plt.plot(alim,alim,'-k',lw=1)
    plt.gca().set_ylim(alim)
    plt.gca().set_xlim(alim)

    plt.xlabel('$\Delta E$ (eV)')
    plt.ylabel('$\Delta E$ Predicted (eV)')

    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('./parity/'+model+'_'+features+'_'+split+'_parity.pdf')
    
    #Functionality to plot histogram of error if desired

    #plt.figure()
    #plt.hist(preds-y,bins=100)
    #plt.xlabel('$\hat{y} - y$ (eV)')
    #plt.title('%s %s: MAE = %.2f eV'%(model,tt,np.abs(y-preds).mean()))

    #xlim = np.array(plt.gca().get_xlim())
    #xmax = np.abs(xlim).max()
    #plt.gca().set_xlim([-xmax,xmax])

    #plt.savefig('./output/'+model+'_'+features+'_'+split+'_'+tt+'_hist.pdf')

    return

if __name__ == '__main__':
    key = ['comp', 'bulk', 'facet', 'coord', 'site_b', 'ads_a', 'ads_b', 'comp_g', 'dE']
    df = pickle.load(open('data/pairs_pdos.pkl'))

    #models = ['rf','lr','krr']#,'NN']
    models = ['rf','boost']
    features = 'pdos'# #pdos,'moments'
    split = 'pairs'

    #Feature Selection
    if features == 'moments':
        X,y = train_prep(df)
    elif features == 'pdos':
        X,y = train_prep_pdos(df,stack=False,include_WF=True)
        #X,y = train_prep_pdos(df,include_WF=True,stack=True)
    
    
    for m in models:
        #Train/test split
        if split == 'none':
            X_train = X
            X_test = X
            y_train = y
            y_test = y
        elif split=='random':
            np.random.seed(100)
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) #need to add dev
            X_dev,X_test,y_dev,y_test = train_test_split(X_test,y_test,test_size=0.5) #need to add dev
        elif split=='composition':
            np.random.seed(100)
            X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,['comp'])
            print "%d training examples, %d test examples"%(len(y_train),len(y_test))
        elif split=='reaction':
            np.random.seed(100)
            X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,['ads_a','ads_b'])
            print "%d training examples, %d test examples"%(len(y_train),len(y_test))
        elif split=='pairs':
            np.random.seed(100)
            X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
            print "%d training examples, %d test examples"%(len(y_train),len(y_test))

        if features == 'moments':
            if m == 'rf':
                model = ensemble.RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=18,
                        max_features=11, max_leaf_nodes=None, min_impurity_decrease=0.0,
                        min_impurity_split=None, min_samples_leaf=2,
                        min_samples_split=2, min_weight_fraction_leaf=0.0,
                        n_estimators=100, n_jobs=None, oob_score=False,
                        random_state=None, verbose=0, warm_start=False)
            elif m == 'lr':
                model = linear_model.LinearRegression()
            elif m == 'krr':
                model =KernelRidge(alpha=0.00089, coef0=1, degree=3, gamma=0.0454545, kernel='rbf', kernel_params=None)
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_dev = scaler.transform(X_dev)
                X_test = scaler.transform(X_test)
            elif m == 'boost':
                #read in boost text data
                train_preds = np.load('boost_y_train_pred_moments.npy')
                test_preds = np.load('boost_y_test_pred_moments.npy')
                evaluate(y_train, train_preds,y_test,test_preds,m,features)
                continue

        elif features == 'pdos':
            if m=='rf':
                model = ensemble.RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=18,
                  max_features='sqrt', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=5,
                  min_weight_fraction_leaf=0.0, n_estimators=26, n_jobs=None,
                  oob_score=False, random_state=None, verbose=0, warm_start=False)
            elif m=='boost':
                train_preds = np.load('boost_y_train_pred_pdos.npy')
                test_preds = np.load('boost_y_test_pred_pdos.npy')
                evaluate(y_train, train_preds,y_test,test_preds,m,features)
                continue

            

        print "Training Model"
        model.fit(X_train,y_train)

        print "Testing Model"
        train_preds = model.predict(X_train).flatten()
        #evaluate(y_train,preds,m,tt='train')

        test_preds = model.predict(X_test).flatten()
        evaluate(y_train, train_preds,y_test,test_preds,m,features)
