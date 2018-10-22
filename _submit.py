from ase.constraints import FixAtoms,FixedLine
from ase.io import read
from ase.build import surface
from ase.visualize import view
import os
import glob

submit = False
bulk_dir = '/home/users/alatimer/work_dir/cs229-PDOS-ML/bulk/'
ads_prototype_dir = '/home/users/alatimer/work_dir/cs229-PDOS-ML/adsorbates/'
vac = 7
adsorbates = ['CH3','CH2','CH','C','NH','NH2','OH','O','CO','OC','NO','ON']
cell_sizes = {
        'fcc':{
            '100':['3x3'],
            '111':['3x3'],
            '211':['3x3'],
            },
        'bcc':{
            '110':['3x3'],
            '100':['3x3'],
            },
        'hcp':{
            '0001':['3x3'],
            '10m10':['3x4'],
            },
        }

sites = {
        'fcc':{
            '100':['ontop','bridge','hollow'],
            '111':['ontop','bridge','hcp'],
            '211':['bridge','ontop'], #bridge, ontop not supported by ase
            },
        'bcc':{
            '100':['ontop','bridge','hollow'],
            '110':['ontop','shortbridge','longbridge','hollow'],
            },
        'hcp':{
            '0001':['ontop','bridge','hcp'],
            '10m10':['bridge','ontop'], #bridge not supported by ase
            },
        }

bond_lengths = {
        'ontop':1.5,
        'bridge':1.2,
        'longbridge':1.2,
        'shortbridge':1.2,
        'hcp':1.,
        'hollow':1.,
        }
##########################################################

assert set(sites.keys()) == set(cell_sizes.keys())
structs = sites.keys()
bulks = [d.split('/')[-1] for d in glob.glob(bulk_dir + '/*')]
comps = {}
for struct in structs:
    comps[struct] = [b.split('_')[1] for b in bulks if b.split('_')[0] == struct]

home = os.getcwd()


def new_dir(d):
    d = os.getcwd() + '/' + d
    if not os.path.isdir(d):
        os.mkdir(d)
    os.chdir(d)
    return

def build_surf(struct,comp,facet,cell_size):
    #first check if lattice opt terminated successfully
    f = open(bulk_dir + '/%s_%s/nospin/myjob.out'%(struct,comp))
    lines = f.readlines()
    f.close()
    if len(lines) <= 1:
        return None
 
    f = open(bulk_dir + '/%s_%s/nospin/lattice_opt.log'%(struct,comp))
    lines = f.readlines()
    f.close()
   
    lines[-1].split()
    a = float(lines[1])
    if len(lines) == 3:
        c = float(lines[2])
    else:
        c = None
    if c == None:
        exec 'surf = surface.' + struct + facet + '(' + comp + ',' + \
                '(%s,%s,%s),'%(cell_size[0],cell_size[1],cell_size[2]) + \
                'a=%s,vacuum=%s'%(a,vac)
    else:
        exec 'surf = surface.' + struct + facet + '(' + comp + ',' + \
                '(%s,%s,%s),'%(cell_size[0],cell_size[1],cell_size[2]) + \
                'a=%s,c=%s,vacuum=%s'%(a,c,vac)

    surf = surf.repeat((cell_size[0],cell_size[1],1))
    return surf


def constrain_surf(surf):
    """
    Constrain lower layers of atoms to their bulk positions. 
    Will constrain at least 50% of the atoms in the slab.
    """
    z_set = sorted(set([round(atom.z,4) for atom in atoms]))
    n_layers = len(z_set)
    cut_layer = int(round(0.51*n_layers,0))
    inds = [atom.index for atom in atoms if atom.z < z_set[cut_layer-1]]
    surf.constraints = [FixAtoms(inds)]
    return

def add_ads(surf,facet,ads,site):
    """
    Add ads (string) to surf. Bonding atom is assumed to be the first in the ads string.
    e.g. CO means binding through C, OC means binding through O.
    """
    if ads in ['OC','ON']:
        ads_atoms = read('%s/%s.traj'%(ads_prototype_dir,ads[1] + ads[0]))
    else:
        ads_atoms = read('%s/%s.traj'%(ads_prototype_dir,ads))
    bonding_atoms = [atom for atom in atoms if atom.symbol == string2symbols(ads)[0]]
    assert len(bonding_atoms) == 1
    bonding_atom = bonding_atoms[0] 
    bonding_atom_ind = bonding_atoms[0].index + len(surf)
    
    surf += ads_atoms

    #figure out bonding atom position based n facet, site#
    try: #use ase builtin method if possible
        surface.add_adsorbate(surf,'X',bond_lengths[site],site)
        bonding_atom_pos = surf[-1].position
        del surf[-1]
    except TypeError:
        #find top layer
        top_z = sorted(set([round(atom.z,4) for atom in atoms]))[-1]
        #find first atom in top layer
        top_atom_inds = [atom.index for atom in atoms if atom.z == top_z]
        top_atom = surf[min(top_atom_inds)] #surface atom in top layer w/ lowest index
        top_atom2 = surf[top_atom_inds[np.argmin(surf.get_distances(top_atom.index,top_atom_inds))]] #1st NN to top_atom
        if facet == '211' and site == 'ontop':
            bonding_atom_pos = top_atom.position + bond_lengths[site]
        elif facet == '211' and site == 'bridge':
            bonding_atom_pos = (top_atom.position + top_atom2.position)/2. + bond_lengths[site]
        elif facet == '10m10' and site == 'bridge':
            bonding_atom_pos = (top_atom.position + top_atom2.position)/2. + bond_lengths[site]
        else:
            raise Exception("Cannot add adsorbate to %s-%s (not in ase.build.surface.add_adsorbate or custom definitions)"%(facet-site))

    for i in range(bonding_atom_ind,len(surf)):
        surf[i].position += bonding_atom_pos - surf[bonding_atom_ind].position

    return surf


def constrain_ads(surf,facet,ads,site):
    """
    Add appropriate FixedLine constraints to adsorbates on surf
    """
    return


def submit(submit=False):
    """
    Submit job if not already finished or currently queued.
    """
    #check if job has completed successfully
    if os.path.isfile('out.WF'):
        return
    
    #check if job is already in the queue or is running
    for user in ['colinfd','alatimer']:
        current_jobs = os.popen('squeue -u %s -o "%Z"'%user).readlines()[1:]
        for job in current_jobs:
            if job.strip() == os.getcwd():
                return
   
    os.system('sbatch relax_auto.py')


for struct in comps:
    new_dir(struct)
    for comp in comps[struct]:
        new_dir(comp)
        for facet in cell_sizes[struct]:
            new_dir(facet)
            for cell_size in cell_sizes[struct][facet]:
                new_dir(cell_size)
                for ads in adsorbates:
                    new_dir(ads)
                    for site in sites[struct][facet]:
                        new_dir(site)
                        surf = build_surf(struct,comp,facet,cell_size)
                        if surf == None: 
                            os.chdir(home); continue
                        constrain_surf(surf)
                        add_ads(surf,ads,site)
                        #constrain_ads(surf,ads)
                        
                        #surf.write('init.traj')
                        view(surf)
                        input('%s %s %s %s %s %s'%(struct,comp,facet,cell_size,ads,site))
                        submit(submit=submit)

                        os.chdir(home)
