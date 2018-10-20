from ase.constraints import FixAtoms,FixedLine
import os
import glob

submit = False
bulk_dir = '??'
adsorbates = ['CH3','CH2','CH','C','NH','NH2','OH','O','CO','NO']
cell_sizes = {
        'FCC':{
            '111':['3x3'],
            '211':['3x3'],
            },
        'BCC':{
            '110':['3x3'],
            '100':['3x3'],
            },
        'HCP':{
            '0001':['3x3'],
            '10m10':['3x4'],
            },
        }

sites = {
        'FCC':{
            '111':['top','bridge','hcp'],
            '211':['edge-bridge','edge-top'],
            },
        'BCC':{
            '100':['top','bridge','hollow'],
            '110':['top','short-bridge','long-bridge'],
            },
        'HCP':{
            '0001':['top','bridge','hcp'],
            '10m10':['edge-bridge','edge-top'],
            },
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
    bulk = read(bulk_dir + '/%s_%s'%(struct,comp))
    surf = surface(bulk,(facet[0],facet[1],facet[2]),cell_size[2],vacuum=vac)
    surf = surf.repeat((cell_size[0],cell_size[1],1))
    return surf


def constrain_surf(surf):
    z_set = sorted(set([round(atom.z,4) for atom in atoms]))
    n_layers = len(z_set)
    cut_layer = int(round(0.51*n_layers,0))
    inds = [atom.index for atom in atoms if atom.z < z_set[cut_layer-1]]
    surf.constraints = [FixAtoms(inds)]
    return


def add_ads(surf,facet,ads,site):
    """
    Add ads (string) to surf
    """


def constrain_ads(surf,facet,ads,site):
    """
    Add appropriate FixedLine constraints to adsorbates on surf
    """


def submit(submit=False):
    #check if job has completed successfully
    if os.path.isfile('out.WF'):
        return
    
    #check if job is already in the queue or is running
    for user in ['colinfd','alatimer']:
        current_jobs = os.popen('squeue -u %s -o "%Z"'%user).readlines()[1:]
        for job in current_jobs:
            if job.strip() == os.getcwd():
                return
   
    os.system('sbatch relaxSP_auto.py')


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
                        constrain_surf(surf)
                        add_ads(surf,ads,site)
                        constrain_ads(surf,ads)

                        surf.write('init.traj')
                        os.system('cp %s/relaxSP_auto.py .'%home)
                        submit(submit=submit)

                        os.chdir(home)
