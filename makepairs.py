import numpy as np
from ase.io import read,write
import pandas as pd


def json2atoms(json):
    f = open('temp.json','w')
    print >> f, json
    f.close()
    atoms = read('temp.json')
    return atoms

def sub_atoms(atoms_b,atoms_a):
    """
    Subtract PD2 atoms from self.
    Returns a dictionary with atom symbols as keys.
    """
    delta_dict = {}
    syms1 = atoms_b.get_chemical_symbols()
    syms2 = atoms_a.get_chemical_symbols()
    for sym in set(syms1 + syms2):
        delta_dict[sym] = syms1.count(sym) - syms2.count(sym)
    return delta_dict

def valid_pair(row):
    ads_a = row['adsorbate_a']
    ads_b = row['adsorbate_b']
    gas = row['composition_g']

    atoms_a = row['atoms_a']
    atoms_b = row['atoms_b']
    
    #a must be shorter than b, no duplicates
    if len(atoms_a) >= len(atoms_b):
        return False
    
    if ads_a == 's':
        if ads_b in gas_list:
            print gas,ads_a,ads_b
            if gas == ads_b:
                print 'gas=ads_b'
                return True
            else:
                return False
        else:
            return False
    else:
        if ads_a[0] == ads_b[0]:
            delta_comp = sub_atoms(atoms_b,atoms_a)
            s=0
            for key in delta_comp :
                s +=delta_comp[key]
            #really should be checking if fragment is a calculated gas
            if s==1:
                if gas=='H':
                    return True
                else:
                    #print gas
                    return False
            else:
                return False
        else:
            return False



if __name__=='__main__':


    gas_list = 'CH3 CH2 CH C NO CO OC ON NH3 NH2 NH N OH2 OH O'.split(' ')
    
    filename ='rawdata.txt'
    df = pd.read_csv(filename, sep='\t')
    df.drop([u'pdos',u'eng_vec'],axis=1,inplace=True)
    df['atoms'] = df['atoms_rel_json'].apply(json2atoms)
    df['atoms_init'] = df['atoms_init_json'].apply(json2atoms)
    df.drop(['atoms_rel_json','atoms_init_json'],axis=1,inplace=True)

    key = ['composition','bulk_structure','facet','cell_size']
    df.set_index(key)
    
    gases = df[df['facet'].isnull()].copy()
    gases.drop(['bulk_structure','facet','cell_size','site','adsorbate'],axis=1,inplace=True)

    jdf = df.join(df.set_index(key),lsuffix='_a',rsuffix='_b',on=key)

    jdf['dummy'] = 'foo'
    gases['dummy']='foo'
    
    gdf = jdf.join(gases.set_index('dummy'),rsuffix='_g',on='dummy')

    gdf.drop(['dummy'],axis=1,inplace=True)
    
    for i in range(10):
        gdf.drop(['moment_%s_b'%(i)],axis=1,inplace=True)
    
    gdf.dropna(inplace=True)
    
    gdf = gdf[gdf.apply(valid_pair,axis=1)]

    gdf['dE'] = gdf['energy_b']-gdf['energy_a'] - gdf['energy']



