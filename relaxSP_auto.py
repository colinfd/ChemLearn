#!/usr/bin/env python
#above line selects special python interpreter which knows all the paths
#SBATCH -p owners,iric,normal
#################
#set a job name
#SBATCH --job-name=
#################
#a file for job output, you can check job progress
#SBATCH --output=myjob.out
#################
# a file for errors from the job
#SBATCH --error=myjob.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=48:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem-per-cpu=4000
#you could use --mem-per-cpu; they mean what we are calling cores
#################
#get emailed about job BEGIN, END, and FAIL
#SBATCH --mail-type=ALL
#################
#who to send email to; please change to your email
#SBATCH  --mail-user=alatimer@stanford.edu
#################
#task to run per node; each node has 16 cores
#SBATCH --ntasks-per-node=16
#################

from ase.io import read,Trajectory
from espresso import espresso
from ase.optimize import QuasiNewton
from ase.constraints import FixAtoms
from ase.parallel import rank
import time
import os
import sys
import numpy as np
import bader,pdos
from wf import calc_wf

#Calculation paramaters
kpts = (4,4,1)
dipole = {'status':True}
pw = 600
dw = 6000
xc = 'RPBE'
psppath = '/scratch/users/colinfd/psp/gbrv'

#Auto restart parameters
avg_iter = 5. #Hours/iteration, too high and you may miss an ionic step, too low and the restart won't work
maxtime = 48. #Time limit on job in hours
restart_command = "/usr/bin/sbatch relaxSP_auto.py"

#Output options
save_pdos_pkl = True
save_cube = False
save_cd = False
         
#Continue previous calculation or start from init.traj
try:
    lines = open('qn.log','r').readlines()
    print "Found qn.log"
    run = int([line for line in lines if 'run' in line][-1].split(';')[1].split('=')[-1].strip()[-1]) + 1
    print "Last run = " + str(run - 1)
    atoms = read('qn.traj')
    atoms_list = read('qn.traj',index=':')
    traj = Trajectory('qn.traj','a',atoms)
    print "Successfully loaded qn.traj"

    open('qn.log','a').write('kpts = (%i,%i,%i); run = %i; image = %d\n' %(kpts[0],kpts[1],kpts[2],run,len(atoms_list)))
except:
    print "Fresh relaxation from init.traj"
    atoms = read('init.traj')
    traj = Trajectory('qn.traj','w',atoms)
    run = 0

    open('qn.log','w').write('kpts = (%i,%i,%i); run = %i; image = 0\n' %(kpts[0],kpts[1],kpts[2],run))

    atoms.rattle(stdev=0.05)
    
    if [atom.magmom for atom in atoms] == [0]*len(atoms):
        ##Set initial magmom here##
        for atom in atoms:
            if atom.symbol == 'H':
                atom.magmom = 1
            else: 
                atom.magmom = 2

calc = espresso(pw=pw,
                dw=dw,
                kpts=kpts,
                xc = xc,
                psppath = psppath,
                spinpol=True,
                outdir='outdir',
                convergence = {'energy':1e-5,
                               'mixing':0.1,
                               'nmix':10,
                               'mixing_mode':'local-TF',
                               'maxsteps':200,
                               'diag':'david'},
                output = {'removesave':True},
                dipole=dipole)

calc.nbands = int(-1*calc.get_nvalence()[0].sum()/5.)
val_dict = calc.get_nvalence()[1]

for atom in atoms:
    if val_dict[atom.symbol] < 2.5:
        atom.magmom = val_dict[atom.symbol]
    else:
        atom.magmom = 2.5

def reduce_magmoms(atoms,ntypx = 10):
    """
    Reduce the number of unique magnetic moments by combining those that are 
    most similar among atoms with the same atomic symbol. This is necessary for
    atoms objects with more than 10 types of magmom/symbol pairs because QE only
    accepts a maximum of 10 types of atoms.
    """
    syms = set(atoms.get_chemical_symbols())

    master_dict = {}
    for sym in syms:
        master_dict[sym] = {}

    for atom in atoms:
        if atom.magmom in master_dict[atom.symbol]:
            master_dict[atom.symbol][atom.magmom].append(atom.index)
        else:
            master_dict[atom.symbol][atom.magmom] = [atom.index]

    ntyp = 0
    for sym in syms:
        ntyp += len(master_dict[sym].keys())

    while ntyp > ntypx:
        magmom_pairs = {}
        for sym in syms:
            magmoms = master_dict[sym].keys()
            if not len(magmoms) > 1: continue
            min_delta = 1e6
            min_pair = ()
            for i,magmom1 in enumerate(magmoms):
                for j,magmom2 in enumerate(magmoms):
                    if not i < j: continue
                    delta = np.abs(magmom1 - magmom2)
                    if delta < min_delta:
                        min_delta = delta
                        min_pair = (magmom1,magmom2)
            
            assert min_delta != 1e6
            assert min_pair != ()
            magmom_pairs[sym] = min_pair
        
        min_delta = 1e6
        min_sym = ""
        for sym in magmom_pairs:
            delta = np.abs(magmom_pairs[sym][0] - magmom_pairs[sym][1])
            if delta < min_delta:
                min_delta = delta
                min_sym = sym

        assert min_delta != 1e6
        assert min_sym != ""
        if min_delta > 0.5:
            print "WARNING, reducing pair of magmoms whose difference is %.2f"%min_delta

        if np.abs(magmom_pairs[min_sym][0]) > np.abs(magmom_pairs[min_sym][1]):
            master_dict[min_sym][magmom_pairs[min_sym][0]].extend(
                    master_dict[min_sym][magmom_pairs[min_sym][1]])
            del master_dict[min_sym][magmom_pairs[min_sym][1]]
        else:
            master_dict[min_sym][magmom_pairs[min_sym][1]].extend(
                    master_dict[min_sym][magmom_pairs[min_sym][0]])
            del master_dict[min_sym][magmom_pairs[min_sym][0]]

        ntyp = 0
        for sym in syms:
            ntyp += len(master_dict[sym].keys())

    #reassign magmoms
    for sym in syms:
        for magmom in master_dict[sym]:
            for index in master_dict[sym][magmom]:
                atoms[index].magmom = magmom

reduce_magmoms(atoms)

atoms.set_calculator(calc)
qn = QuasiNewton(atoms,logfile = 'qn.log',force_consistent=False)

####################################
##Functions to attach to optimizer##
####################################

def estimate_magmom():
    """
    Estimate magmom from log file (based on charge spheres centered on atoms) and assign to 
    atoms object to assist with calculation restart upon unexpected interruption
    """
    f = open('outdir/log')
    lines = f.readlines()
    f.close()

    i = len(lines) - 1
    while True:
        if i == 0: raise IOError("Could not identify espresso magmoms")
        line = lines[i].split()
        if len(line) > 3:
            if line[0] == "absolute":
                abs_magmom = float(line[3])
        if len(line) > 6:
            if line[4] == "magn:":
                i -= len(atoms) - 1
                break
        i -= 1
    
    if abs_magmom < 1e-3:
        for atom in atoms:
            atom.magmom = 0
    else:
        total_esp_magmom = 0
        for j in range(len(atoms)):
            total_esp_magmom += np.abs(float(lines[i+j].split()[5]))

        for j in range(len(atoms)):
            atoms[j].magmom = float(lines[i+j].split()[5])*abs_magmom/total_esp_magmom

def StopCalc():
    now = time.time()
    if maxtime - (now - starttime)/3600. < avg_iter:
        pdos.pdos(atoms,outdir='outdir',spinpol=True) #update magmom from pdos for helping in calc restart
        traj.write()
        if rank == 0:
            open('auto_restart','a').write("Run %i ended on " %(run) + time.strftime('%m/%d/%Y\t%H:%M:%S') + " after %4.2f hours" %((now-starttime)/3600.) + '\n')
            os.system(restart_command)
            sys.exit()

####################################

qn.attach(estimate_magmom)
qn.attach(traj)
qn.attach(StopCalc)
starttime = time.time()

if run > 0:
    qn.replay_trajectory('qn.traj')
qn.run(fmax=0.05)

pdos.pdos(atoms,outdir='outdir',spinpol=True,save_pkl=save_pdos_pkl,
        Emin=-20,Emax=20,kpts=(kpts[0]*3,kpts[1]*3,1),DeltaE=0.01,
        nscf=True)

bader.bader(atoms,outdir='outdir',spinpol=True,save_cube=save_cube,save_cd=save_cd)

#Calculate Work Function
wf = calc.get_work_function(pot_filename="pot.xsf", edir=3)
fw = open('AEout.WF','w')
fw.write(str(wf))
fw.close()
os.system('rm pot.xsf')

#Calculate with karen's script
wf = calc_wf(atoms,'outdir')
fw = open('out.WF','w')
fw.write(str(wf))
fw.close()
