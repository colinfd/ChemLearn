import numpy as np
from ase.io import read,write
import pandas as pd

gases = 'CH3 CH2 CH C NO CO OC ON NH3 NH2 NH N OH2 OH O'.split(' ')

def json2atoms(json):
    f = open('temp.json','w')
    print >> f, json
    f.close()
    atoms = read('temp.json')
    return atoms

def sub_atoms(self,atoms):
    """
    Subtract PD2 atoms from self.
    Returns a dictionary with atom symbols as keys.
    """
    delta_dict = {}
    syms1 = atoms.get_chemical_symbols()
    syms2 = atoms.get_chemical_symbols()
    for sym in set(syms1 + syms2):
        delta_dict[sym] = syms1.count(sym) - syms2.count(sym)
    return delta_dict

def valid_pair(row):
    ads_a = row['adsorbate_a']
    ads_b = row['adsorbate_b']

    atoms_a = row['atoms_a']
    atoms_b = row['atoms_b']
    if len(row['atoms_a']) >= len(row['atoms_b']):
        return False
    
    if ads_a == 's':
        if ads_b in gases:
            return True
        else:
            return False
    else:
        if ads_a[0] == ads_b[0]:
            delta_comp = atoms_b.sub(atoms_a)
            s=0
            for key in delta_comp :
                s +=delta_comp[key]
            if s==1:
                return True


if __name__=='__main__':
    
    filename ='rawdata.txt'
    df = pd.read_csv(filename, sep='\t')
    df.drop([u'pdos',u'eng_vec'],axis=1,inplace=True)
    df['atoms'] = df['atoms_rel_json'].apply(json2atoms)
    df['atoms_init'] = df['atoms_init_json'].apply(json2atoms)
    df.drop(['atoms_rel_json','atoms_init_json'],axis=1,inplace=True)
    key = ['composition','bulk_structure','facet','cell_size']
    df.set_index(key)
    jdf = df.join(df.set_index(key),lsuffix='_a',rsuffix='_b',on=key)
    for i in range(10):
        jdf.drop(['moment_%s_b'%(i)],axis=1,inplace=True)
    
    #filtering
   #jdf = jdf.ix[jdf['atoms_a'].apply(len) < jdf['atoms_b'].apply(len) ]

    jdf = jdf[jdf.apply(valid_pair,axis=1)]





