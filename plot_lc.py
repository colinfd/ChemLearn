import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model,ensemble
from ML_prep import train_prep_pdos,split_by_cols,train_prep,add_noise
from matplotlib.pyplot import cm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
from scipy.stats import randint as sp_randint

features = ['moments']#,'pdos']# #pdos,'moments'
models_labels = ['Linear Regression','Random Forest','Boosting','Kernel RR']
models = ['lr','rf','boost','krr']

colors = cm.rainbow(np.linspace(0,1,6))
i = 0

percent = range(10,101,10)

for feature in features:
    for m_lab,m in zip(models_labels,models):
        i+=1
        if (m == 'lr' or m == 'boost' )and feature == 'pdos':
            continue
             
        mae_train = np.load('learning_curve/mae_train_%s_%s.npy'%(m, feature))
        mae_dev = np.load('learning_curve/mae_dev_%s_%s.npy'%(m, feature))
        
        plt.plot(percent,mae_dev,'--',label=m_lab+' Dev',color=colors[i])
        plt.plot(percent,mae_train,'-',label=m_lab+' Train',color=colors[i])

    plt.title(feature)
plt.legend()
plt.ylim(0,1.5)
plt.ylabel('MAE (eV)')
plt.xlabel('% training data included')
plt.savefig('learning_curve/learning_curve_%s.pdf'%(feature))
