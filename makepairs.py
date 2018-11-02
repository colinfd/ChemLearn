import numpy as np
from ase.io import read,write
import pandas as pd
from ase.atoms import string2symbols
from collections import Counter

def json2atoms(json):
    f = open('temp.json','w')
    print >> f, json
    f.close()
    atoms = read('temp.json')
    return atoms

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

    #Read in rawdata as pd df
    filename ='rawdata.txt'
    df = pd.read_csv(filename, sep='\t',dtype={'facet':str})
    
    #Drop clunky pdos, eng_vec, shouldn't need unless training NNs
    df.drop([u'pdos',u'engs'],axis=1,inplace=True)
    
    #Read atoms objects and remove json columns
    df['atoms'] = df['atoms_rel_json'].apply(json2atoms)
    df['atoms_init'] = df['atoms_init_json'].apply(json2atoms)
    df.drop(['atoms_rel_json','atoms_init_json'],axis=1,inplace=True)
    
    #Create new DF with gases only 
    gases = df[df['bulk']=='gas'].copy()
    gas_list = [gas for gas in gases['comp']]
    gases.drop(['bulk','facet','cell_size','site','ads'],axis=1,inplace=True)
    
    #Cross join on full DF & clean up
    df['dummy'] = 'foo'
    gases['dummy']='foo'
    gdf = df.join(gases.set_index('dummy'),rsuffix='_g',on='dummy')
    gases.drop(['dummy'],axis=1,inplace=True)
    gdf.drop(['dummy'],axis=1,inplace=True)
    df.drop(['dummy'],axis=1,inplace=True)
    #print gdf.info()
    #exit()
    #gdf.reset_index(inplace=True)
    #gases.reset_index(inplace=True)
    #gdf.drop(['dummy'],axis=1,inplace=True)
    #gdf.drop(['index'],axis=1,inplace=True)  
    
    #Join gdf on df again for pair finding
    key = ['comp','bulk','facet','cell_size','site']
    gdf.set_index(key)
    gdf = gdf.join(df.set_index(key),lsuffix='_a',rsuffix='_b',on=key)
    
    #Can drop moments for adsorbate_b bc will always be final state
    for i in range(10):
        gdf.drop(['moment_%s_b'%(i)],axis=1,inplace=True)
    
    gdf.dropna(inplace=True)
    
    gdf = gdf[gdf.apply(valid_pair,axis=1)]

    gdf['dE'] = gdf['eng_b']-gdf['eng_a'] - gdf['eng_g']

    key = ['comp','bulk','facet','cell_size','site','ads_a','ads_b']
    gdf.reset_index(inplace=True)
    gdf.drop(['index'],inplace=True,axis=1)

    gdf.to_pickle('pairs.pkl')

    #Filter out RMSD?



