import pickle
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

def add_noise(X,y,std_X = 0.1, std_y = 0.1,mult=2):
    if len(X.shape)==2:
        X_new = X.copy()
        y_new = y.copy()
        for i in range(mult-1):
            noise_X = np.random.normal(0,std_X,X_new.shape[1])
            X_add = X_new.copy()
            X_add+=noise_X
            X = np.append(X,X_add,axis=0)
            noise_y = np.random.normal(0,std_y,y_new.shape[-1])
            y_add = y_new.copy()
            y_add +=noise_y
            y = np.append(y,y_add)
        return X,y
    else:
        print("Warning: Not yet implemented for stacked X")
        return X,y
    #elif len(X.shape)==3:

def train_prep(df,scale_zeroth_mom=True,include_WF=True):
    y = df['dE'].values#.reshape(-1,1)
    
    moms = [i for i in df.columns if 'mom' in i]
    if include_WF:
        moms.append('WF_a')
        moms.append('WF_b')
    
    X = df[moms].values
    
    #multiply zeroth moment by coord
    if scale_zeroth_mom:
        X[:,0] = df.apply(lambda x: x.moment_0_a * x.coord_b if x.site_a == 's' else x.moment_0_a, axis=1).values

    return X,y

def train_prep_pdos(df,include_WF=False,stack=False,dE=0.1):
    y = df['dE'].values#.reshape(-1,1)

    e_a_min = df.apply(lambda x: x.engs_a[0],axis=1).min()
    e_a_max = df.apply(lambda x: x.engs_a[-1],axis=1).max()
    e_g_min = df.apply(lambda x: x.engs_g[0],axis=1).min()
    e_g_max = df.apply(lambda x: x.engs_g[-1],axis=1).max()
    
    e_base = np.arange(min(e_a_min,e_g_min),max(e_g_max,e_a_max),dE)
    fermi = np.argmin(np.abs(e_base))
    
    if stack:
        X = np.zeros((df.shape[0],2,len(e_base)))
    else:
        X = np.zeros((df.shape[0],2*len(e_base)))

    
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

def split_by_cols(df,X,y,cols,f_train=0.7,f_dev=0.15,ret_groups=False):
    """
    Perform a train-dev-test split that is disjoint with respect to cols.
    i.e. df_train[i][cols] != df_val[i][cols] != df_test[i][cols] for all i

    if ret_groups, also return a list containing group indices for each training
    example, which can be used with GroupKFold.
    """
    df2 = df[cols].drop_duplicates().sample(frac=1)
    split1 = int(df2.shape[0] * f_train)
    split2 = split1 + int(df2.shape[0] * f_dev)
    df2_train = df2.iloc[:split1]
    df2_dev = df2.iloc[split1:split2]
    df2_test = df2.iloc[split2:]
    
    X_train = np.empty(X.shape[1:])[np.newaxis,...]
    X_dev = np.empty(X.shape[1:])[np.newaxis,...]
    X_test = np.empty(X.shape[1:])[np.newaxis,...]
    y_train = []
    y_dev = []
    y_test = []
    
    g_train = []

    for i in range(df.shape[0]):
        s = df.iloc[i]
        train_bool = (df2_train[cols] == s[cols]).all(axis=1)
        if train_bool.sum() > 0:
            X_train = np.vstack((X_train, X[i][np.newaxis,...]))
            y_train.append(y[i])
            g_train.append(train_bool.idxmax())
        elif (df2_dev[cols] == s[cols]).all(axis=1).sum() > 0:
            X_dev = np.vstack((X_dev, X[i][np.newaxis,...]))
            y_dev.append(y[i])
        elif (df2_test[cols] == s[cols]).all(axis=1).sum() > 0:
            X_test = np.vstack((X_test, X[i][np.newaxis,...]))
            y_test.append(y[i])
        else:
            raise Exception()

    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    y_test = np.array(y_test)
    
    ret = [X_train[1:], X_dev[1:], X_test[1:], y_train, y_dev, y_test]
    if ret_groups:
        ret.append(g_train)

    return ret




def load_data(type='moments'):
    X_train = np.load('data/X_train_%s.npy'%type)
    X_dev = np.load('data/X_dev_%s.npy'%type)
    X_test = np.load('data/X_test_%s.npy'%type)
    y_train = np.load('data/y_train_%s.npy'%type)
    y_dev = np.load('data/y_dev_%s.npy'%type)
    y_test = np.load('data/y_test_%s.npy'%type)

    return X_train,X_dev,X_test,y_train,y_dev,y_test



if __name__ == "__main__":
    import sys
    import numpy as np
    np.random.seed(100)
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        pdos = True
        xtype = 'pdos'
        stack = 'stacked'
    else:
        pdos = False
        xtype = 'moments'
        stack = 'flat'

    if pdos:
        df = pickle.load(open('data/pairs_pdos.pkl'))
        X,y = train_prep_pdos(df,include_WF=False,stack=True)
    else:
        df = pickle.load(open('data/pairs.pkl'))
        X,y = train_prep(df,scale_zeroth_mom=True)

    X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
    np.save('data/X_train_%s_%s.npy'%(xtype,stack),X_train)
    np.save('data/X_dev_%s_%s.npy'%(xtype,stack),X_dev)
    np.save('data/X_test_%s_%s.npy'%(xtype,stack),X_test)
    np.save('data/y_train_%s_%s.npy'%(xtype,stack),y_train)
    np.save('data/y_dev_%s_%s.npy'%(xtype,stack),y_dev)
    np.save('data/y_test_%s_%s.npy'%(xtype,stack),y_test)

    np.save('data/X_%s_%s.npy'%(xtype,stack),X)
    np.save('data/y_%s_%s.npy'%(xtype,stack),y)
    
