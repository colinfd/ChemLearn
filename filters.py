import numpy as np

def check_coord(row,u_cut=0.05,l_cut=1.3):
    """
    Identifies if coordination number of binding atom relative to metal atoms matches coord.
        u_cut - maxmimum allowed relative difference between nearest and furthest NN
        l_cut - minimum allowed ratio of furthest NN to nearest 2nd NN.
    """
    atoms = row['atoms']
    inds = row['ads_indices']
    coord = row['coord']

    if coord == 0 or np.isnan(coord):
        return True
    else:
        coord = int(row['coord'])

    M_inds = [atom.index for atom in atoms if atom.index not in inds]
    
    d = atoms.get_distances(inds[0],M_inds,mic=True)
    d_tup = zip(d,M_inds)
    d_tup.sort(key=lambda x: x[0])
    d,M_inds = zip(*d_tup)
    
    if (d[coord-1] - d[0])/d[0] > u_cut:  
        return False

    if d[coord]/d[coord-1] < l_cut: 
        return False
    
    return True

def check_diss(row,cutoff=1.25):
    """
    Checks to see if adsorbate has dissociated based on a cutoff bond length requirement.
        cutoff - bond length cutoff in Angstroms. 
            - adsorbates with any bond length above this cutoff will return False
        Returns False if:
            - any bond length in adsorbate > cutoff
        Return True if:
            - any bond length in adsorbate < cutoff
            - row corresponds to gas, clean surface, or single atom adsorbate
    """
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

if __name__ == '__main__':
    import pickle
    df = pickle.load(open('surfDB.pkl','r'))
    fdf = df[df.apply(check_diss,axis=1)]
    gdf = fdf[fdf.apply(check_coord,axis=1)]
