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

def train_prep_moms(df):
    y = df['dE'].values#.reshape(-1,1)
    #moms = [i for i in df.columns if 'moment_1' in i]
    moms = [i for i in df.columns if 'mom' in i]
    
    #moms.append('WF_a')
    #moms.append('WF_b')
    #moms.append('coord')
    
    #df['moment_1_a'] -= df['WF_a']
    #df['moment_1_g'] -= df['WF_g']
    
    X = df[moms].values

    return X,y

def train_prep_pdos(df,include_WF=True):
    y = df['dE'].values#.reshape(-1,1)
    
    e_a_base = df.loc[df.apply(lambda x: len(x.engs_a),axis=1).idxmax()].engs_a
    e_g_base = df.loc[df.apply(lambda x: len(x.engs_g),axis=1).idxmax()].engs_g
    
    X = np.zeros((df.shape[0],len(e_a_base) + len(e_g_base)))

    for i in range(X.shape[0]):
        pdos_a = interp1d(df.iloc[i].engs_a,df.iloc[i].pdos_a,bounds_error=False,fill_value=0)(e_a_base) \
                * df.iloc[i].coord_b
        pdos_g = interp1d(df.iloc[i].engs_g,df.iloc[i].pdos_g,bounds_error=False,fill_value=0)(e_g_base)
        
        X[i,:] = np.concatenate((pdos_a,pdos_g))

    if include_WF:
        X = np.append(X,df['WF_a'].values[...,np.newaxis],axis=1)
        X = np.append(X,df['WF_g'].values[...,np.newaxis],axis=1)
    
    return X,y


def is_fs_split(df,X,y,f=0.75):
    """
    Perform a train-test split such that there are no identical reactions
    in both the train and test sets from a composition stantdpoint. 
        e.g. Au + CO --> Au-CO can exist in the training set many times 
            (different facets, sites), but nowhere in the test set.
    """
    comp_a_b = df[['comp','ads_a','ads_b']].drop_duplicates()
    split = int(comp_a_b.shape[0] * 0.75)
    comp_a_b_train = comp_a_b.iloc[:split]
    comp_a_b_test = comp_a_b.iloc[split:]

    X_train = np.empty((0,X.shape[1]))
    X_test = np.empty((0,X.shape[1]))
    y_train = []
    y_test = []

    for i in range(df.shape[0]):
        s = df.iloc[i]
        if len(comp_a_b_train[(comp_a_b_train.comp == s.comp) & (comp_a_b_train.ads_a == s.ads_a) \
                & (comp_a_b_train.ads_b == s.ads_b)]) > 0:
            X_train = np.vstack((X_train, X[i]))
            y_train.append(y[i])
        elif len(comp_a_b_test[(comp_a_b_test.comp == s.comp) & (comp_a_b_test.ads_a == s.ads_a) \
                & (comp_a_b_test.ads_b == s.ads_b)]) > 0:
            X_test = np.vstack((X_test, X[i]))
            y_test.append(y[i])
        else:
            raise Exception()

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test

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

def multiply_zeroth_mom(row):
    if row['site_a'] == 's':
        return row['moment_0_a'] * row['coord_b']
    else:
        return row['moment_0_a']

def evaluate(y,preds,model,tt='train'):
    
    plt.figure()
    plt.plot(y,preds,'.r',ms=3)
    plt.title('%s %s: MAE = %.2f eV'%(model,tt,np.abs(y-preds).mean()))

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    
    plt.plot(xlim,xlim,'-k',lw=1)
    plt.gca().set_xlim(xlim)

    plt.xlabel('$\Delta E$ (eV)')
    plt.ylabel('$\Delta E$ Predicted (eV)')

    plt.savefig(model+'_'+tt+'_parity.pdf')

    plt.figure()
    plt.hist(preds-y,bins=100)
    plt.xlabel('$\hat{y} - y$ (eV)')
    plt.title('%s %s: MAE = %.2f eV'%(model,tt,np.abs(y-preds).mean()))

    xlim = np.array(plt.gca().get_xlim())
    xmax = np.abs(xlim).max()
    plt.gca().set_xlim([-xmax,xmax])

    plt.savefig(model+'_'+tt+'_hist.pdf')

    return

if __name__ == '__main__':
    key = ['comp', 'bulk', 'facet', 'coord', 'site_b', 'ads_a', 'ads_b', 'comp_g', 'dE']
    df = pickle.load(open('pairs_pdos.pkl'))

    models = ['RF','LinReg','GP','NN']
    features = 'pdos' #'moments'
    split = 'random'

    #Feature Selection
    if features == 'moments':
        X,y = train_prep_moms(df)
    elif features == 'pdos':
        X,y = train_prep_pdos(df)
    
    np.save('X',X)
    np.save('y',y)

    #Train/test split
    if split == 'none':
        X_train = X
        X_test = X
        y_train = y
        y_test = y
    elif split=='random':
        X_train,X_test,y_train,y_test = train_test_split(X,y)
    elif split=='pairs':
        X_train,X_test,y_train,y_test = is_fs_split(df,X,y)
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
            model = ensemble.RandomForestRegressor()
        elif m == 'LinReg':
            model = linear_model.LinearRegression()
        elif m == 'GP':
            model = gaussian_process.GaussianProcessRegressor()
        elif m == 'NN':
            model = build_nn()

        print "Training Model"
        if m == 'NN':
            model.fit(X_train,y_train,epochs=500)
        else:
            model.fit(X_train,y_train)

        print "Testing Model"
        preds = model.predict(X_train).flatten()
        evaluate(y_train,preds,m,tt='train')

        preds = model.predict(X_test).flatten()
        evaluate(y_test,preds,m,tt='test')
