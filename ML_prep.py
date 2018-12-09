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

def train_prep_pdos(df,include_WF=True,stack=True,dE=0.1):
    y = df['dE'].values#.reshape(-1,1)

    e_a_min = df.apply(lambda x: x.engs_a[0],axis=1).min()
    e_a_max = df.apply(lambda x: x.engs_a[-1],axis=1).max()
    e_g_min = df.apply(lambda x: x.engs_g[0],axis=1).min()
    e_g_max = df.apply(lambda x: x.engs_g[-1],axis=1).max()
    
    e_base = np.arange(min(e_a_min,e_g_min),max(e_g_max,e_a_max),dE)
    
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


def is_fs_split(df,X,y,f_train=0.7,f_dev=0.15):
    """
    Perform a train-dev-test split such that there are no identical reactions
    in both the train and test sets from a composition stantdpoint. 
        e.g. Au + CO --> Au-CO can exist in the training set many times 
            (different facets, sites), but nowhere in the test set.
    """
    comp_a_b = df[['comp','ads_a','ads_b']].drop_duplicates()
    split1 = int(comp_a_b.shape[0] * f_train)
    split2 = split1 + int(comp_a_b.shape[0] * f_dev)
    comp_a_b_train = comp_a_b.iloc[:split1]
    comp_a_b_dev = comp_a_b.iloc[split1:split2]
    comp_a_b_test = comp_a_b.iloc[split2:]
    
    X_train = np.empty(X.shape[1:])[np.newaxis,...]
    X_dev = np.empty(X.shape[1:])[np.newaxis,...]
    X_test = np.empty(X.shape[1:])[np.newaxis,...]
    y_train = []
    y_dev = []
    y_test = []

    for i in range(df.shape[0]):
        s = df.iloc[i]
        if len(comp_a_b_train[(comp_a_b_train.comp == s.comp) & (comp_a_b_train.ads_a == s.ads_a) \
                & (comp_a_b_train.ads_b == s.ads_b)]) > 0:
            X_train = np.vstack((X_train, X[i][np.newaxis,...]))
            y_train.append(y[i])
        elif len(comp_a_b_dev[(comp_a_b_dev.comp == s.comp) & (comp_a_b_dev.ads_a == s.ads_a) \
                & (comp_a_b_dev.ads_b == s.ads_b)]) > 0:
            X_dev = np.vstack((X_dev, X[i][np.newaxis,...]))
            y_dev.append(y[i])
        elif len(comp_a_b_test[(comp_a_b_test.comp == s.comp) & (comp_a_b_test.ads_a == s.ads_a) \
                & (comp_a_b_test.ads_b == s.ads_b)]) > 0:
            X_test = np.vstack((X_test, X[i][np.newaxis,...]))
            y_test.append(y[i])
        else:
            raise Exception()

    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    y_test = np.array(y_test)

    return X_train[1:], X_dev[1:], X_test[1:], y_train, y_dev, y_test


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        pdos = True
        xtype = 'pdos'
    else:
        pdos = False
        xtype = 'moments'

    if pdos:
        df = pickle.load(open('data/pairs_pdos.pkl'))
        X,y = train_prep_pdos(df,include_WF=False,stack=True)
    else:
        df = pickle.load(open('data/pairs.pkl'))
        X,y = train_prep(df,scale_zeroth_mom=True)

    X_train,X_dev,X_test,y_train,y_dev,y_test = is_fs_split(df,X,y)
    np.save('data/X_train_%s.npy'%(xtype),X_train)
    np.save('data/X_dev_%s.npy'%(xtype),X_dev)
    np.save('data/X_test_%s.npy'%(xtype),X_test)
    np.save('data/y_train_%s.npy'%(xtype),y_train)
    np.save('data/y_dev_%s.npy'%(xtype),y_dev)
    np.save('data/y_test_%s.npy'%(xtype),y_test)

    np.save('data/X_%s.npy'%(xtype),X)
    np.save('data/y_%s.npy'%(xtype),y)
