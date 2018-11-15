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

def get_ads_indices(row):
    atoms = row['atoms']
    ads = row['ads']
    if row['bulk']=='gas':
        return np.NaN
    binding_atom = ads[0]
    indices = []
    for atom in atoms:
        if atom.symbol in ['C','H','O','N']:
            if atom.symbol == binding_atom:
                indices.insert(0,atom.index)
            else:
                indices.append(atom.index)
    return indices

if __name__=='__main__':

    #Read in rawdata as pd df
    filename ='rawdata.txt'
    df = pd.read_csv(filename, sep='\t',dtype={'facet':str})
    
    #Drop clunky pdos, eng_vec, shouldn't need unless training NNs
    df.drop(['pdos','engs'],axis=1,inplace=True)
    
    #Read atoms objects and remove json columns
    df['atoms'] = df['atoms_rel_json'].apply(json2atoms)
    df['atoms_init'] = df['atoms_init_json'].apply(json2atoms)
    df.drop(['atoms_rel_json','atoms_init_json'],axis=1,inplace=True)

    df['ads_indices'] = df.apply(get_ads_indices,axis=1)
    
    df.to_pickle('surfDB.pkl')


