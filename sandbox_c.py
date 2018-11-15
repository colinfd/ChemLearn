import pickle
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def get_coord(row):
    site = row['site_b']
    bulk = row['bulk']
    facet = row['facet']
    
    if 'top' in site:
        return 1
    
    elif 'bridge' in site:
        return 2
    
    elif 'hcp' in site:
        return 3
    
    elif 'hollow' in site:
        if bulk == 'bcc' and facet == '110':
            return 3
        else:
            return 4
    
    else:
        raise Exception(row)

def train_prep(df):
    y = df['dE'].values.reshape(-1,1)
    #moms = [i for i in df.columns if 'moment_1' in i]
    moms = [i for i in df.columns if 'mom' in i]
    
    #moms.append('WF_a')
    #moms.append('WF_b')
    #moms.append('coord')
    
    #df['moment_1_a'] -= df['WF_a']
    #df['moment_1_g'] -= df['WF_g']
    
    X = df[moms].values

    return X,y


key = ['comp', 'bulk', 'facet', 'coord', 'site_b', 'ads_a', 'ads_b', 'comp_g', 'dE']

df = pickle.load(open('pairs.pkl'))
#apply filters
#df = df[df.coord_b == 1]
df = df[df.ads_a == 'C']
df = df[df.ads_b == 'CO']
#df = df[df.bulk == 'fcc']

def multiply_zeroth_mom(row):
    if row['site_a'] == 's':
        return row['moment_0_a'] * row['coord_b']
    else:
        return row['moment_0_a']

df['moment_0_a'] = df.apply(multiply_zeroth_mom,axis=1)

X,y = train_prep(df)
X_train,X_test,y_train,y_test = train_test_split(X,y)

lm = linear_model.LinearRegression()

if False:
    lm.fit(X_train,y_train)
    preds = lm.predict(X_test)
    plt.plot(y_test,preds,'.r',ms=6)
    plt.title('MAE = %.2f eV'%(np.abs(y_test-preds).mean()))
else:
    lm.fit(X,y)
    preds = lm.predict(X)
    plt.plot(y,preds,'.r',ms=6)
    plt.title('MAE = %.2f eV'%(np.abs(y-preds).mean()))

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

plt.plot(xlim,xlim,'-k',lw=1)
plt.gca().set_xlim(xlim)


plt.xlabel('$\Delta E$ (eV)')
plt.ylabel('$\Delta E$ Predicted (eV)')

plt.show()
