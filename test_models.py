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

def build_nn():
    model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

def evaluate(y_train,train_preds,y_test,test_preds,model,features):
    
    plt.figure(figsize=(4,3))
    plt.plot(y_train,train_preds,'.b',ms=3,label='train',alpha=0.5)
    plt.plot(y_test,test_preds,'.r',ms=3,label='test',alpha=1)
    plt.title('%s: Train MAE = %.2f eV; Test MAE = %.2f eV'%(model,np.abs(y_train-train_preds).mean(),np.abs(y_test-test_preds).mean()),fontsize=10)

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
    plt.savefig('./output/'+model+'_'+features+'_'+split+'_parity.pdf')

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

    #df = df[df['site_b']=='ontop'][df['ads_b']=='O'][df['ads_a'] == 's'][df['bulk'] == 'fcc'][df['comp'] != 'Sr'][df['comp'] != 'Ca'][df['comp'] != 'Pb'][df['facet'] == '111'][df['comp']!='Al']
    #df = df[df['ads_b']=='OH'][df['ads_a'] == 'O']
    #df = df[df['site_b']=='ontop']


    models = ['RF']#,'LinReg','GP']#,'NN']
    features = 'moments'# #pdos,'moments'
    split = 'pairs'#'random',pairs

    #Feature Selection
    if features == 'moments':
        X,y = train_prep(df)
    elif features == 'pdos':
        X,y = train_prep_pdos(df,stack=False,include_WF=False)
        print(X.shape,y.shape)
        #X,y = train_prep_pdos(df,include_WF=True,stack=True)
    
    np.save('X',X)
    np.save('y',y)

    #Train/test split
    if split == 'none':
        X_train = X
        X_test = X
        y_train = y
        y_test = y
    elif split=='random':
        X_train,X_test,y_train,y_test = train_test_split(X,y) #need to add dev
    elif split=='pairs':
        #X_train,X_dev,X_test,y_train,y_dev,y_test = is_fs_split(df,X,y)
        np.random.seed(42)
        X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
        print "%d training examples, %d test examples"%(len(y_train),len(y_test))
    elif split == 'au':
        no_Au = (df.comp != 'Au').values
        Au = (df.comp == 'Au').values
        X_train = X[no_Au,:]
        X_test = X[Au,:]
        y_train = y[no_Au]
        y_test = y[Au]
    
    for m in models:
        if m == 'RF':
            #model = ensemble.RandomForestRegressor()
            model = ensemble.RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=18,
                    max_features=11, max_leaf_nodes=None, min_impurity_decrease=0.0,
                    min_impurity_split=None, min_samples_leaf=2,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=None, oob_score=False,
                    random_state=None, verbose=0, warm_start=False)
        elif m == 'LinReg':
            model = linear_model.LinearRegression()
        elif m == 'GP':
            model = gaussian_process.GaussianProcessRegressor(normalize_y=True)
        elif m == 'NN':
            model = build_nn()

        print "Training Model"
        if m == 'NN':
            model.fit(X_train,y_train,epochs=200)
        else:
            model.fit(X_train,y_train)

        print "Testing Model"
        train_preds = model.predict(X_train).flatten()
        #evaluate(y_train,preds,m,tt='train')

        dev_preds = model.predict(X_dev).flatten()
        evaluate(y_train, train_preds,y_dev,dev_preds,m,features)
