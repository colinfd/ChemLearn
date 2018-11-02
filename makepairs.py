import numpy as np
from ase.io import read,write
import pandas as pd

def json2atoms(json):
    f = open('temp.json','w')
    print >> f, json
    f.close()
    atoms = read('temp.json')
    return atoms

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
    
    #filtering
    #jdf = jdf.ix[len(jdf['atoms_a']) >= len(jdf['atoms_a']) ]




