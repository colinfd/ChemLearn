import pickle
import matplotlib.pyplot as plt
from ML_prep import load_data, split_by_cols, train_prep_pdos, train_prep
import numpy as np

df = pickle.load(open('data/pairs_pdos.pkl'))
X,y = train_prep_pdos(df,include_WF=True,dE=0.1)
X1,y1 = train_prep(df,include_WF=True)

i = 1094
#i = 889

n = 2
figsize = (8,8)

if n == 2:
    fig1 = plt.figure(facecolor=(0.75,)*3,figsize=figsize)
    fig2 = plt.figure(facecolor=(0.75,)*3,figsize=figsize)
    axs = [fig1.add_subplot(111),fig2.add_subplot(111)]
else:
    fig = plt.figure(figsize=figsize)
    axs = [fig.add_subplot(111)]

while True:
    print i
    Xi = X[i]

    axs[0].fill_between(range(200),[0]*200,Xi[:200],facecolor='deepskyblue',edgecolor='none')
    axs[0].fill_between(range(199,352),[0]*(352-199),Xi[199:352],facecolor='deepskyblue',edgecolor='none',alpha=0.3)

    if n == 2:
        ax = axs[1]
    else:
        ax = axs[0]

    ax.fill_between(range(352,552),[0]*200,Xi[352:552],facecolor='darkgoldenrod',edgecolor='none')
    ax.fill_between(range(351,X.shape[1]-2),[0]*(X.shape[1]-2-351),Xi[351:-2],facecolor='darkgoldenrod',edgecolor='none',alpha=0.5)


    for ax in axs:
        ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
                labelbottom=False,labelright=False,labelleft=False)
        #ax.set_xlim([100,700])
        ax.set_xlabel('Energy w.r.t Fermi Level',family='sans-serif')
        ax.set_ylabel('PDOS',family='sans-serif')
    
    plt.savefig('feature_example.png',dpi=300)
    plt.show()
    
    #i = np.random.randint(X.shape[0])
