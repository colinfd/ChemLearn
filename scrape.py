#!/usr/bin/env python
#SBATCH -p owners,iric,normal
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err
#SBATCH --time=03:00:00

import glob
from ase.io import read,write
import pickle
import json
from StringIO import StringIO
import numpy as np
import os

id_dict = {
        'fcc_100':[27,36],
        'fcc_211':[0,36],
        'fcc_111':[27,36],
        'hcp_0001':[27,36],
        'hcp_10m10':[36,48],
        'bcc_100':[27,36],
        'bcc_110':[27,36],
        }

def get_RMSD(a,b):
    d = np.linalg.norm(a.positions-b.positions,axis=1)
    rmsd = np.sqrt(np.mean(d**2))
    return rmsd

def get_fermi(log):
    f = open(log,'r')
    lines = f.readlines()
    f.close()

    n = len(lines)

    for i in range(n):
        line = lines[n-1-i]
        if len(line.split()) == 6:
            if line.split()[1] == 'Fermi':
                return float(line.split()[-2])

def get_moments(e,pdos,n):
    moments = []
    for i in range(n):
        if i==0:
            moment = np.trapz(pdos,x=e)
        elif i==1:
            moment = np.trapz(pdos*e,x=e)/moments[0]
        elif i>1:
            moment = np.trapz((pdos-moments[1])*e**i,x=e)/moments[0]
        moments.append(moment)
    return moments

def get_pdos(directory,atoms):
    pdos = pickle.load(open(directory+'/pdos.pkl','r'))
    eng = pdos[0]
    index = None
    
    if 'surfaces' in directory:
        attrs = directory.split('/')
        facet = attrs[-4]
        bulk_structure = attrs[-6]
        ads = attrs[-2]
        if ads == 's':
            index = id_dict[bulk_structure+'_'+facet][0]
        elif ads in ['CH3','OH','NH3','CO','NO','OC','ON']:
            return 'NULL','NULL'
        else:
            index = id_dict[bulk_structure+'_'+facet][1]
    elif 'gases' in directory:
        gas = directory.split('/')[-2]
        for atom in atoms:
            if atom.symbol == gas[0]:
                index = atom.index

    total_pdos = np.zeros((len(eng),))
    for key in pdos[2][index]:
        total_pdos +=np.array(pdos[2][index][key][0])
    
    #print index,atoms[index].symbol
    return eng, total_pdos

def add_line(directory,txt,delim,n=0,lines=None):

    if 'gases' in directory:
        composition = directory.split('/')[-2]
        bulk_structure = 'gas'
        facet = 'NULL'
        cell_size = 'NULL'
        adsorbate = 'NULL'
        site = 'NULL'
        WF = -get_fermi(directory+'/calcdir/log')
    
    else:
        if os.path.exists(directory+'/out.WF')==False  or os.stat(directory+"/out.WF").st_size == 0:
            return
        attrs = directory.split('/')
        composition = attrs[-5]
        bulk_structure = attrs[-6]
        facet = attrs[-4]
        cell_size = attrs[-3]
        adsorbate = attrs[-2]
        site = attrs[-1]
        WF = eval(open(directory+'/out.WF','r').read())[0]
    
    #Check if already in DB
    if lines != None:
        for line in lines:
            key = delim.join([composition, bulk_structure, facet, cell_size, adsorbate, site])
            if line.startswith(key):
                print "Already in DB: ",key
                return 
    
    print "Adding:",directory
    atoms_init = read(directory+'/init.traj')
    atoms_rel = read(directory+'/qn.traj')
    rmsd = get_RMSD(atoms_init,atoms_rel)
    eng_vec,pdos = get_pdos(directory,atoms_rel)
    if n>0 and str(pdos)!='NULL':
        moments = get_moments(eng_vec,pdos,n)
        strmom = ''
        for m in moments:
            if m==moments[-1]:    
                strmom+='%.6e'%(m)
            else:
                strmom+='%.6e%s'%(m,delim)
            #strmom+='%.6e,'%(m)
    else:
        strmom = ''
        for i in range(n):
            if i==n-1:
                strmom += 'NULL'
            else:
                strmom += 'NULL%s'%(delim)
            #strmom = 'NULL'
    energy = atoms_rel.get_potential_energy()

    atoms_rel.write('temp.json',append=False)
    atoms_rel_json = open('temp.json','r').read().replace('\n',' ')
     
    atoms_init.write('temp.json',append=False)
    atoms_init_json = open('temp.json','r').read().replace('\n',' ')
   
    line = [composition, bulk_structure, facet, cell_size, adsorbate, site, 
        str(energy), str(WF), atoms_init_json, atoms_rel_json, str(rmsd), 
        strmom, str(list(eng_vec)), str(list(pdos)) ]
    #print line[0:-2]
    print >> txt, delim.join(line)

    return 

if __name__ == '__main__':   
    #if making db for first time
    initialize=False
    roots = ['../gases/*/nospin']
    #roots.append('../surfaces/*/*/*/*/*/*')
    roots.append('/home/users/colinfd/scratch/CS229_project/ChemLearn/surfaces/*/*/*/*/*/*')
    delim='\t'
    n=10
    if initialize == True:
        txt = open('rawdata.txt','w') #w if delete all
        strmom = ' '.join(['moment_%i'%(i) for i in range(n)])
        #headers = 'composition bulk_structure facet cell_size adsorbate site energy WF atoms_init_json atoms_rel_json rmsd moments eng_vec pdos'.replace(' ',delim)
        #headers = ('composition bulk_structure facet cell_size adsorbate site energy WF atoms_init_json atoms_rel_json rmsd %s eng_vec pdos'%(strmom)).replace(' ',delim)
        headers = ('comp bulk facet cell_size ads site eng WF atoms_init_json atoms_rel_json rmsd %s engs pdos'%(strmom)).replace(' ',delim)
        print >> txt, headers
        lines = None
    else:
        txt = open('rawdata.txt','a')
        lines = open('rawdata.txt','r').readlines()

    for root in roots:
        #print root
        for direc in glob.glob(root):
            add_line(direc,txt,delim,n,lines)

