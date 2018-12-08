import pickle
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
import seaborn as sns
from sklearn import linear_model,ensemble,gaussian_process
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from ML_prep import train_prep_pdos,is_fs_split,train_prep

def evaluate(y_train,train_preds,y_test,test_preds,model,features):
    
    plt.figure(figsize=(4,3))
    plt.plot(y_train,train_preds,'.b',ms=3,label='train',alpha=0.5)
    plt.plot(y_test,test_preds,'.r',ms=3,label='test',alpha=1)
    plt.title('%s: Train MAE = %.2f eV; Test MAE = %.2f eV'%(model,np.abs(y_train-train_preds).mean(),np.abs(y_test-test_preds).mean()),fontsize=10)

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    
    alim = (min(xlim[0],ylim[0]),max(xlim[1],ylim[1]))

    plt.plot(alim,alim,'-k',lw=1)
    plt.gca().set_ylim(alim)
    plt.gca().set_xlim(alim)

    plt.xlabel('$\Delta E$ (eV)')
    plt.ylabel('$\Delta E$ Predicted (eV)')

    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('./output/'+model+'_'+features+'_'+split+'_parity.pdf')

    #plt.figure()
    #plt.hist(preds-y,bins=100)
    #plt.xlabel('$\hat{y} - y$ (eV)')
    #plt.title('%s %s: MAE = %.2f eV'%(model,tt,np.abs(y-preds).mean()))

    #xlim = np.array(plt.gca().get_xlim())
    #xmax = np.abs(xlim).max()
    #plt.gca().set_xlim([-xmax,xmax])

    #plt.savefig('./output/'+model+'_'+features+'_'+split+'_'+tt+'_hist.pdf')

    return

if __name__ == '__main__':
    key = ['comp', 'bulk', 'facet', 'coord', 'site_b', 'ads_a', 'ads_b', 'comp_g', 'dE']
    df = pickle.load(open('pairs_pdos.pkl'))

    m = 'RF'#,'LinReg','GP']#,'NN']
    features = 'pdos'# #pdos,'moments'
    split = 'pairs'#'random',pairs

    #Feature Selection
    if features == 'moments':
        X,y = train_prep(df)
    elif features == 'pdos':
        X,y = train_prep_pdos(df,stack=False,include_WF=False)
    
    np.save('X',X)
    np.save('y',y)

    #Train/test split
    if split == 'none':
        X_train = X
        X_test = X
        y_train = y
        y_test = y
    elif split=='random':
        X_train,X_test,y_train,y_test = train_test_split(X,y) #need to add dev
    elif split=='pairs':
        X_train,X_dev,X_test,y_train,y_dev,y_test = is_fs_split(df,X,y)
        print "%d training examples, %d test examples"%(len(y_train),len(y_test))
    elif split == 'au':
        no_Au = (df.comp != 'Au').values
        Au = (df.comp == 'Au').values
        X_train = X[no_Au,:]
        X_test = X[Au,:]
        y_train = y[no_Au]
        y_test = y[Au]
    
    if m == 'RF':
        ntrees = np.arange(1,40,2)
        n = X.shape[1]
        print(X.shape)
        npreds = np.array([n-1,n/2.,np.round(np.sqrt(n)),10,5])
        print(npreds)
        for j,npred in enumerate(npreds): 
            mae_test = np.zeros(ntrees.shape)
            mae_train = np.zeros(ntrees.shape)
            for i,nt in enumerate(ntrees):
                print npred
                model = ensemble.RandomForestRegressor(n_estimators=nt,max_features=int(npred))
                model.fit(X_train,y_train)
                train_preds = model.predict(X_train).flatten()
                test_preds = model.predict(X_test).flatten()
                mae_train[i] = np.abs(y_train-train_preds).mean()
                mae_test[i] = np.abs(y_test-test_preds).mean()
                print(m,nt,mae_test[i],mae_train[i])
            plt.plot(ntrees,mae_test,'-',label='test_'+str(npred))
            plt.plot(ntrees,mae_train,'-',label='train_'+str(npred))
        plt.legend(loc='best')
        plt.savefig('./output/'+m+'_'+features+'_'+split+'_tuningAll.pdf')
