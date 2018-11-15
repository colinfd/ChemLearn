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

