import numpy as np
from ase.io import read,write
import pandas as pd
from ase.atoms import string2symbols
from collections import Counter
import pickle


def valid_pair(row):
    ads_a = row['ads_a']
    ads_b = row['ads_b']
    gas = row['comp_g']

    atoms_a = row['atoms_a']
    atoms_b = row['atoms_b']
    
    #a must be shorter than b, no duplicates
    if len(atoms_a) >= len(atoms_b):
        return False
    
    if ads_a == 's':
        if gas == ads_b:
            return True
        else:
            return False
    else:
        #If first element (binding element) equal, may be good
        if row['site_a']!= row['site_b']:
            return False
        if ads_a[0] == ads_b[0]:
            comp_a = Counter(string2symbols(ads_a))
            comp_b = Counter(string2symbols(ads_b))
            diff = comp_b-comp_a
            dcomp = list(diff.elements()) 
            if len(dcomp)==1 and gas == dcomp[0]:
                return True
            else:
                return False
        else:
            return False

if __name__=='__main__':

    df = pickle.load(open('surfDB.pkl'))
    
    #Create new DF with gases only 
    gases = df[df['bulk']=='gas'].copy()
    gas_list = [gas for gas in gases['comp']]
    gases.drop(['bulk','facet','cell_size','site','ads'],axis=1,inplace=True)

    #Remove gas entries from df
    df = df[df['bulk']!='gas']
    
    #Cross join on full DF & clean up
    df['dummy'] = 'foo'
    gases['dummy']='foo'
    gdf = df.join(gases.set_index('dummy'),rsuffix='_g',on='dummy')
    gases.drop(['dummy'],axis=1,inplace=True)
    gdf.drop(['dummy'],axis=1,inplace=True)
    df.drop(['dummy'],axis=1,inplace=True)
    
    #Join gdf on df again for pair finding
    key = ['comp','bulk','facet','cell_size']
    gdf.set_index(key)
    gdf = gdf.join(df.set_index(key),lsuffix='_a',rsuffix='_b',on=key)
    
    #Can drop moments for adsorbate_b bc will always be final state
    for i in range(10):
        gdf.drop(['moment_%s_b'%(i)],axis=1,inplace=True)
    
    gdf = gdf[gdf.apply(valid_pair,axis=1)]

    gdf['dE'] = gdf['eng_b']-gdf['eng_a'] - gdf['eng_g']

    key = ['comp','bulk','facet','cell_size','site_b','ads_a','ads_b']
    gdf.reset_index(inplace=True)
    gdf.drop(['index'],inplace=True,axis=1)

    gdf.sort_values(key,inplace=True)

    gdf.to_pickle('pairs.pkl')
