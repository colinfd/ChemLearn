#!/usr/bin/env python
#SBATCH -p dev
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err
#SBATCH --time=00:45:00

from ase.constraints import FixAtoms,FixedLine
from ase.io import read
from ase.build.surface import fcc100, fcc111, fcc211, bcc110, bcc100, hcp0001, hcp10m10, add_adsorbate
from ase.atoms import string2symbols
from ase.visualize import view
import os
import glob
import numpy as np

os.chdir('surfaces')
bulk_dir = '/home/users/alatimer/work_dir/cs229-PDOS-ML/bulk/'
ads_prototype_dir = '/home/users/alatimer/work_dir/cs229-PDOS-ML/adsorbates/'
vac = 7
adsorbates = ['s','CH3','CH2','CH','C','NH','NH2','OH','O','CO','OC','NO','ON']
cell_sizes = {
        'fcc':{
            '100':['3x3x4'],
            '111':['3x3x4'],
            '211':['3x3x4'],
            },
        'bcc':{
            '110':['3x3x4'],
            '100':['3x3x4'],
            },
        'hcp':{
            '0001':['3x3x4'],
            '10m10':['3x4x4'],
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
        'ontop':2,
        'bridge':1.2,
        'longbridge':1.2,
        'shortbridge':1.2,
        'hcp':1.2,
        'hollow':1.2,
        }
##########################################################

assert set(sites.keys()) == set(cell_sizes.keys())
for struct in sites:
    for facet in sites[struct]:
        sites[struct][facet].append('s')
structs = sites.keys()
bulks = [d.split('/')[-1] for d in glob.glob(bulk_dir + '/*')]
comps = {}
for struct in structs:
    comps[struct] = [b.split('_')[1] for b in bulks if b.split('_')[0] == struct]

home = os.getcwd()


def new_dir(d):
    d = os.getcwd() + '/' + d
    if not os.path.isdir(d):
        os.system('mkdir -p ' + d)
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
   
    build_fun = eval(struct + facet)
    cell_size = [int(i) for i in cell_size.split('x')]
    line = lines[-1].split()
    if len(line) == 3:
        a = float(line[2][:-1])
        surf = build_fun(comp,cell_size,a=a,vacuum=vac)
    else:
        a = float(line[2][:-1])
        c = float(line[3][:-1])*a
        surf = build_fun(comp,cell_size,a=a,c=c,vacuum=vac)
    return surf


def constrain_surf(surf):
    """
    Constrain lower layers of atoms to their bulk positions. 
    Will constrain at least 50% of the atoms in the slab.
    """
    z_set = sorted(set([atom.z for atom in surf]))
    n_layers = len(z_set)
    cut_layer = int(round(0.51*n_layers,0))
    inds = [atom.index for atom in surf if atom.z <= z_set[cut_layer-1]]
    surf.constraints = [FixAtoms(inds)]
    return

def add_ads(surf,facet,ads,site,fixed_line=True):
    """
    Add ads (string) to surf. Bonding atom is assumed to be the first in the ads string.
    e.g. CO means binding through C, OC means binding through O.
    if fixed_line == True: add fixed_line constraint to binding atom.
    """
    ads_atoms = read('%s/%s.traj'%(ads_prototype_dir,ads))
    bonding_atoms = [atom for atom in ads_atoms if atom.symbol == string2symbols(ads)[0]]
    assert len(bonding_atoms) == 1, ""
    bonding_atom = bonding_atoms[0] 
    bonding_atom_ind = bonding_atoms[0].index + len(surf)
    
    #figure out bonding atom position based n facet, site#
    try: #use ase builtin method if possible
        add_adsorbate(surf,'X',bond_lengths[site],site)
        bonding_atom_pos = surf[-1].position
        del surf[-1]
    except TypeError:
        #find top layer
        top_z = sorted(set([atom.z for atom in surf]))[-1]
        #find first atom in top layer
        top_atom_inds = [atom.index for atom in surf if atom.z == top_z]
        top_atom = surf[top_atom_inds.pop(np.argmin(top_atom_inds))] #surface atom in top layer w/ lowest index
        top_atom2 = surf[top_atom_inds[np.argmin(surf.get_distances(top_atom.index,top_atom_inds))]] #1st NN to top_atom
        if facet == '211' and site == 'ontop':
            bonding_atom_pos = top_atom.position + np.array([0,0,bond_lengths[site]])
        elif facet == '211' and site == 'bridge':
            bonding_atom_pos = (top_atom.position + top_atom2.position)/2. + np.array([0,0,bond_lengths[site]])
        elif facet == '10m10' and site == 'bridge':
            bonding_atom_pos = (top_atom.position + top_atom2.position)/2. + np.array([0,0,bond_lengths[site]])
        else:
            raise Exception("Cannot add adsorbate to %s-%s (not in ase.build.surface.add_adsorbate or custom definitions)"%(facet-site))

    surf += ads_atoms
    delta_vec = bonding_atom_pos - surf[bonding_atom_ind].position
    for i in range(len(surf)-len(ads_atoms),len(surf)):
        surf[i].position += delta_vec

    if fixed_line:
        surf.constraints.append(FixedLine(bonding_atom_ind,[0,0,1]))
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
        current_jobs = os.popen('squeue -u %s -o "%%Z"'%user).readlines()[1:]
        for job in current_jobs:
            if job.strip() == os.getcwd():
                return
   
    os.system('cp %s/../relax_auto.py .'%home)
    if submit:
        os.system('sbatch relax_auto.py')


for struct in comps:
    for comp in comps[struct]:
        for facet in cell_sizes[struct]:
            for cell_size in cell_sizes[struct][facet]:
                for ads in adsorbates:
                    for site in sites[struct][facet]:
                        if ads == 's' and site != 's': continue
                        if ads != 's' and site == 's': continue
                        
                        new_dir('%s/%s/%s/%s/%s/%s'%(struct,comp,facet,cell_size,ads,site))
                        print '%s %s %s %s %s %s'%(struct,comp,facet,cell_size,ads,site)
                        surf = build_surf(struct,comp,facet,cell_size)
                        if surf == None: 
                            os.chdir(home)
                            continue

                        constrain_surf(surf)

                        if ads != 's':
                            add_ads(surf,facet,ads,site,fixed_line=True)
                        
                        surf.write('init.traj')
                        submit(submit=False)

                        os.chdir(home)
