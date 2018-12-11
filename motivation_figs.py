import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

df = pickle.load(open('data/pairs_pdos.pkl'))

def plot(df,xlim,color):
    x = df.moment_1_a
    y = df.dE

    fit = linregress(x,y)
    
    plt.plot(xlim,fit[0]*xlim + fit[1],'-',color=color,lw=1)
    plt.plot(x,y,'o',color=color,ms=7)


#d-band model
df_dO1 = df[df.bulk == 'bcc'][df.facet == '100'][df.ads_b == 'O'][df.site_b == 'ontop'][df.comp != 'Rb'][df.comp != 'Ba']
df_dO2 = df[df.bulk == 'bcc'][df.facet == '100'][df.ads_b == 'O'][df.site_b == 'bridge'][df.comp != 'Rb'][df.comp != 'Ba']
df_dC1 = df[df.bulk == 'bcc'][df.facet == '100'][df.ads_b == 'C'][df.site_b == 'bridge'][df.comp != 'Rb'][df.comp != 'Ba']
xlim = np.array([-0.8,0.2])
ylim = np.array([-10,-7])


plot(df_dO1,xlim,'black')
plot(df_dO2,xlim,'red')
plot(df_dC1,xlim,'blue')

plt.xlabel('Surface PDOS 1st Moment (eV)')
plt.ylabel('$\Delta E$ (eV)')

plt.gca().set_xlim(xlim)
plt.gca().set_ylim(ylim)


#O --> OH
xlim = np.array([-9,-3])
plt.figure()
df_O_OH = df[df.ads_a == 'O'][df.ads_b == 'OH']

plot(df_O_OH,xlim,'black')

plt.gca().set_xlim(xlim)

plt.xlabel('O* PDOS 1st Moment (eV)')
plt.ylabel('$\Delta E$ (eV)')


#C --> CH
xlim = np.array([0,-5])
plt.figure()
df_O_OH = df[df.ads_a == 'C'][df.ads_b == 'CH']

plot(df_O_OH,xlim,'black')

plt.gca().set_xlim(xlim)

plt.xlabel('C* PDOS 1st Moment (eV)')
plt.ylabel('$\Delta E$ (eV)')

plt.show()
