import pickle
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def check_diss(row,cutoff):
    atoms = row['atoms']
    ads_ind = row['ads_indices']
    if row['bulk'] == 'gas':
        return True
    if len(ads_ind) < 2:
        return True
    for ai in ads_ind[1:]:
        dist = np.linalg.norm(atoms[ai].position - atoms[ads_ind[0]].position)
        if dist > cutoff:
            return False
    return True

def longest_bl(row):
    atoms = row['atoms']
    bls = []
    if row['bulk'] == 'gas':
        if len(atoms)==1:
            return np.NaN
        for atom in atoms[1:]:
            dist = np.linalg.norm(atom.position - atoms[0].position)
            bls.append(dist)
        return max(bls)
    ads_ind = row['ads_indices']
    if len(ads_ind) > 2:
        for ai in ads_ind[1:]:
            dist = np.linalg.norm(atoms[ai].position - atoms[ads_ind[0]].position)
            bls.append(dist)
        return max(bls)
    else:
        return np.NaN


pkey = ['comp', 'bulk', 'facet', 'site', 'ads_a', 'ads_b', 'gas', 'dE']
skey = ['comp', 'bulk', 'facet', 'site', 'ads']

pdf = pickle.load(open('pairs.pkl'))
df = pickle.load(open('surfDB.pkl'))

features = []
for n in range(10):
    features.append('moment_%s_a'%(n))
    features.append('moment_%s_g'%(n))

X_train,X_test,y_train,y_test = train_test_split(pdf[features],pdf['dE'])

df['bl'] = df.apply(longest_bl,axis=1)

lens=[]
cutoffs=np.arange(1,3,0.01)
for cutoff in cutoffs:
    num = len(df[df.apply(check_diss,axis=1,args=(cutoff,))])
    lens.append(num)

plt.plot(cutoffs,lens,'ok')
plt.ylim((0,2750))
plt.savefig('hist-mat-inc.pdf')

plt.clf()

df.hist(column = 'bl',bins=500)
plt.savefig('hist-maxbl.pdf')
