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
import PlotFormat

"""
Plots learning curves for all models, need to specify 
whether looking at moments or pdos as features
"""

features = 'moments'

if features == 'moments':
    models_labels = ['Linear Regression','Random Forest','Gradient Boosting','Kernel Ridge Regression']
    models = ['lr','rf','boost','krr']
elif features == 'pdos':
    models_labels = ['Random Forest','Conv Neural Net','Gradient Boosting']
    models = ['rf','cnn','boost']

colors = PlotFormat.muted_colors
i = 0

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

num_train_ex = 1408
percent = np.arange(.1,1.01,.1)
size_train = percent*num_train_ex

for m_lab,m in zip(models_labels,models):
    i+=1
    if m == 'lr' and features == 'pdos':
        continue
         
    mae_train = np.load('learning_curve/mae_train_%s_%s.npy'%(m, features))
    mae_dev = np.load('learning_curve/mae_dev_%s_%s.npy'%(m, features))
    
    print(mae_dev.shape,mae_train.shape,m_lab)
    plt.plot(size_train,mae_dev,'--',label=m_lab+' Dev',color=colors[m])
    plt.plot(size_train,mae_train,'-',label=m_lab+' Train',color=colors[m])

    #plt.title(features)

plt.legend(fontsize=8)
plt.ylim(0,1.5)
plt.ylabel('MAE (eV)')
plt.xlabel('# training examples included')
plt.xlim((size_train[0],size_train[-1]))
plt.savefig('learning_curve/learning_curve_%s.pdf'%(features))
