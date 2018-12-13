import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

"""
Generates Figure 3 of the text which verifies previous d-band and O 2p states model results
"""

df = pickle.load(open('data/pairs_pdos.pkl'))

colors = ['rebeccapurple','plum','darkorange','darkcyan','cornflowerblue']

ctr =0

def plot(df,xlim,label):
    global ctr
    x = df.moment_1_a
    y = df.dE

    fit = linregress(x,y)
    
    plt.plot(xlim,fit[0]*xlim + fit[1],'-',color=colors[ctr],lw=1)
    plt.plot(x,y,'o',color=colors[ctr],ms=3,label=label)

    ctr+=1

figsize=(3,3)

#d-band model
df_dO1 = df[df.bulk == 'bcc'][df.facet == '100'][df.ads_b == 'O'][df.site_b == 'ontop'][df.comp != 'Rb'][df.comp != 'Ba']
df_dO2 = df[df.bulk == 'bcc'][df.facet == '100'][df.ads_b == 'O'][df.site_b == 'bridge'][df.comp != 'Rb'][df.comp != 'Ba']
df_dC1 = df[df.bulk == 'bcc'][df.facet == '100'][df.ads_b == 'C'][df.site_b == 'bridge'][df.comp != 'Rb'][df.comp != 'Ba']
xlim = np.array([-0.8,0.2])
ylim = np.array([-10,-7])


plt.figure(figsize=(6,3))

plot(df_dO1,xlim,'O_ontop')
plot(df_dO2,xlim,'O_bridge')
plot(df_dC1,xlim,'C_bridge')

plt.legend()

plt.xlabel('Surface PDOS 1st Moment (eV)')
plt.ylabel('$\Delta E$ (eV)')

plt.gca().set_xlim(xlim)
plt.gca().set_ylim(ylim)
plt.tight_layout()

plt.savefig('motivation/dband.pdf')

#O --> OH
xlim = np.array([-9,-3])
plt.figure(figsize=figsize)
df_O_OH = df[df.ads_a == 'O'][df.ads_b == 'OH']

plot(df_O_OH,xlim,'black')

plt.gca().set_xlim(xlim)

plt.xlabel('O* PDOS 1st Moment (eV)')
plt.ylabel('$\Delta E$ (eV)')

plt.tight_layout()

plt.savefig('motivation/O-OH.pdf')
#C --> CH
xlim = np.array([0,-5])
plt.figure(figsize=figsize)
df_O_OH = df[df.ads_a == 'C'][df.ads_b == 'CH']

plot(df_O_OH,xlim,'black')

plt.gca().set_xlim(xlim)

plt.xlabel('C* PDOS 1st Moment (eV)')
plt.ylabel('$\Delta E$ (eV)')

plt.tight_layout()

plt.savefig('motivation/C-CH.pdf')
