import numpy as np
import cPickle as pickle
from ase.io import read,write
import pandas as pd
from ase.atoms import string2symbols
from collections import Counter
import sys
    

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

def get_coord(row):
    site = row['site']
    bulk = row['bulk']
    facet = row['facet']

    if type(site) != str and np.isnan(site):
        return np.nan
    
    if 's' in site:
        return 0

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

def get_RMSD(row,norm=True,slab_only=True):
    a = row['atoms_init']
    b = row['atoms']
    if row['bulk'] == 'gas':
        return np.NaN
    #need to include functionality later
    #if slab_only ==True:
    d = np.linalg.norm(a.positions-b.positions,axis=1)
    if norm==True:
        lattice = np.min(a.get_distances(0,range(1,len(a))))
        d/=lattice
    rmsd = np.sqrt(np.mean(d**2))
    return rmsd

if __name__=='__main__':
    #Read in rawdata as pd df
    if len(sys.argv) > 1 and sys.argv[1] == '-r':
        filename ='rawdata.txt'
        df = pd.read_csv(filename, sep='\t',dtype={'facet':str})
        
        #Drop clunky pdos, eng_vec, shouldn't need unless training NNs
        df.drop(['pdos','engs'],axis=1,inplace=True)
        
        #Read atoms objects and remove json columns
        df['atoms'] = df['atoms_rel_json'].apply(json2atoms)
        df['atoms_init'] = df['atoms_init_json'].apply(json2atoms)
        df.drop(['atoms_rel_json','atoms_init_json'],axis=1,inplace=True)

    else:
        df = pickle.load(open('surfDB.pkl'))

    df['ads_indices'] = df.apply(get_ads_indices,axis=1)
    df['coord'] = df.apply(get_coord,axis=1)
    df['rmsd_norm'] = df.apply(get_RMSD,axis=1)
    
    df.to_pickle('surfDB.pkl')
