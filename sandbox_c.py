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

def train_prep(df,scale_zeroth_mom=True):
    y = df['dE'].values#.reshape(-1,1)
    
    moms = [i for i in df.columns if 'mom' in i]
    #moms.append('WF_a')
    #moms.append('WF_b')
    #moms.append('coord')
    
    #df['moment_1_a'] -= df['WF_a']
    #df['moment_1_g'] -= df['WF_g']
    
    X = df[moms].values
    
    #multiply zeroth moment by coord
    if scale_zeroth_mom:
        X[:,0] = df.apply(lambda x: x.moment_0_a * x.coord_b if x.site_a == 's' else x.moment_0_a, axis=1).values

    return X,y

def train_prep_pdos(df,include_WF=True,stack=True):
    y = df['dE'].values#.reshape(-1,1)

    e_a_min = df.apply(lambda x: x.engs_a[0],axis=1).min()
    e_a_max = df.apply(lambda x: x.engs_a[-1],axis=1).max()
    e_g_min = df.apply(lambda x: x.engs_g[0],axis=1).min()
    e_g_max = df.apply(lambda x: x.engs_g[-1],axis=1).max()
    
    e_base = np.arange(min(e_a_min,e_g_min),max(e_g_max,e_a_max),0.01)
    
    if stack:
        X = np.zeros((df.shape[0],2,len(e_base)))
    else:
        X = np.zeros((df.shape[0],len(e_base)))

    
    for i in range(X.shape[0]):
        pdos_a = interp1d(df.iloc[i].engs_a,df.iloc[i].pdos_a,bounds_error=False,fill_value=0)(e_base)
        if df.iloc[i].site_a == 's':
            pdos_a *= df.iloc[i].coord_b
        pdos_g = interp1d(df.iloc[i].engs_g,df.iloc[i].pdos_g,bounds_error=False,fill_value=0)(e_base)
        
        if stack:
            X[i,0,:] = pdos_a
            X[i,1,:] = pdos_g
        else:
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

key = ['comp', 'bulk', 'facet', 'coord', 'site_b', 'ads_a', 'ads_b', 'comp_g', 'dE']
df = pickle.load(open('pairs_pdos.pkl'))

#apply filters
#df = df[df.coord_b == 1]
#df = df[df.ads_a == 'C']
#df = df[df.ads_b == 'CO']
#df = df[df.bulk == 'fcc']

#df['moment_0_a'] = df.apply(multiply_zeroth_mom,axis=1)

#X,y = train_prep_pdos(df,include_WF=False,stack=True)
X,y = train_prep(df)
#np.savetxt('X.csv',X)
#np.savetxt('y.csv',y)
np.save('data/X.npy',X)
np.save('data/y.npy',y)
#X,y = train_prep(df)

if True:
    X_train = X
    X_test = X
    y_train = y
    y_test = y
elif False:
    X_train,X_test,y_train,y_test = train_test_split(X,y)
elif False:
    X_train,X_test,y_train,y_test = is_fs_split(df,X,y)
    print "%d training examples, %d test examples"%(len(y_train),len(y_test))
else:
    no_Au = (df.comp != 'Au').values
    Au = (df.comp == 'Au').values
    X_train = X[no_Au,:]
    X_test = X[Au,:]
    y_train = y[no_Au]
    y_test = y[Au]

#model = linear_model.LinearRegression()
model = ensemble.RandomForestRegressor()
#model = gaussian_process.GaussianProcessRegressor()
#model = build_nn()

print "Training Model"

model.fit(X_train,y_train,epochs=500)

print "Testing Model"
preds = model.predict(X_test).flatten()
plt.plot(y_test,preds,'.r',ms=3)
plt.title('MAE = %.2f eV'%(np.abs(y_test-preds).mean()))

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

plt.plot(xlim,xlim,'-k',lw=1)
plt.gca().set_xlim(xlim)


plt.xlabel('$\Delta E$ (eV)')
plt.ylabel('$\Delta E$ Predicted (eV)')

plt.figure()
plt.hist(preds-y_test,bins=100)
plt.xlabel('$\hat{y} - y$ (eV)')
plt.title('MAE = %.2f eV'%(np.abs(y_test-preds).mean()))

xlim = np.array(plt.gca().get_xlim())
xmax = np.abs(xlim).max()
plt.gca().set_xlim([-xmax,xmax])

plt.show()
