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
from matplotlib.patches import Rectangle

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mae = np.mean(errors)
    #print('Model Performance')
    #print('Average Error: {:0.4f} eV.'.format(mae))
    return mae


df = pickle.load(open('data/pairs_pdos.pkl'))

features = ['pdos']#,'pdos']# #pdos,'moments'
models = ['rf']#['lr','rf']

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)


for feature in features:
    for m in models:
        if m == 'lr' and feature == 'pdos':
            continue
        
        if feature == 'moments':
            X,y = train_prep(df)
        elif feature == 'pdos':
            X,y = train_prep_pdos(df,stack=False,include_WF=True,dE=0.1)
        
        np.random.seed(42)
        X_train,X_dev,X_test,y_train,y_dev,y_test = split_by_cols(df,X,y,['comp','ads_a','ads_b'])
        if m == 'rf':
            if feature == 'pdos':	    
                model = ensemble.RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=18,
                  max_features='sqrt', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=5,
                  min_weight_fraction_leaf=0.0, n_estimators=26, n_jobs=None,
                  oob_score=False, random_state=None, verbose=0, warm_start=False)
            elif feature == 'moments':
                model = ensemble.RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=18,
                    max_features=11, max_leaf_nodes=None, min_impurity_decrease=0.0,
                    min_impurity_split=None, min_samples_leaf=2,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=None, oob_score=False,
                    random_state=None, verbose=0, warm_start=False)
        elif m == 'lr':
            model = linear_model.LinearRegression()

        #ax2.plot(X_train.mean(axis=0),lw=.1)
        ax.set_ylim((0,7))
        #ax.fill_between(range(X_train.shape[1]),[0]*X_train.shape[1],X_train.mean(axis=0),facecolor='deepskyblue',edgecolor=None)
        ax.fill_between(range(200),[0]*200,X_train.mean(axis=0)[:200],facecolor='deepskyblue',edgecolor='none')
        ax.fill_between(range(352,552),[0]*200,X_train.mean(axis=0)[352:552],facecolor='darkgoldenrod',edgecolor='none')

        ax.fill_between(range(199,352),[0]*(352-199),X_train.mean(axis=0)[199:352],facecolor='deepskyblue',edgecolor='none',alpha=0.3)
        ax.fill_between(range(351,X_train.shape[1]-2),[0]*(X_train.shape[1]-2-351),X_train.mean(axis=0)[351:-2],facecolor='darkgoldenrod',edgecolor='none',alpha=0.5)

        #ax.plot(X_train.mean(axis=0)[:352],'-k',lw=0.5)
        #ax.plot(range(352,X_train.shape[1]-2),X_train.mean(axis=0)[352:X_train.shape[1]-2],'-k',lw=0.5)


        num_examples = X_train.shape[0]
        
        model.fit(X_train, y_train)
        mae_dev = evaluate(model, X_dev, y_dev)
        mae_train = evaluate(model, X_train, y_train)
        
        ylim = [0,0.04]
        ax2 = ax.twinx()
        #ax2.add_patch(Rectangle((0,0),352,0.04,fc='darkgoldenrod',ec=None,alpha=0.3))
        #ax2.add_patch(Rectangle((352,0),352,0.04,fc='darksalmon',ec=None,alpha=0.3))
        #ax2.plot([200]*2,ylim,'--k',lw=1)
        #ax2.plot([552]*2,ylim,'--k',lw=1)
        for i in range(len(model.feature_importances_[:-2])):
            ax2.plot([i]*2,[0,model.feature_importances_[i]],'-k',lw=1)
        
        ax2.add_patch(Rectangle((352*2,0),10,model.feature_importances_[-2],fc='deepskyblue',ec='white'))#,alpha=0.3))
        ax2.add_patch(Rectangle((352*2+10,0),10,model.feature_importances_[-1],fc='darkgoldenrod',ec='white'))#,alpha=0.3))
        #ax.bar(range(len(model.feature_importances_)),model.feature_importances_)
        ax2.set_ylim(ylim)
        ax.set_xlim([0,352*2 + 50])
        
        #plt.bar(range(len(model.feature_importances_)),model.feature_importances_)


        #np.save('learning_curve/mae_train_%s_%s.npy'%(m, feature),mae_train)
        #np.save('learning_curve/mae_dev_%s_%s.npy'%(m, feature),mae_dev)
        
        #plt.plot(mae_dev,label=m+' '+feature+' dev')
        #plt.plot(mae_train,label=m+' '+feature+' train')
#plt.legend()
#plt.ylim(0,1.5)
#plt.savefig('feature_importance/feature_importance_%s.pdf'%(feature))

#plt.clf()


for axi in [ax,ax2]:
    axi.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
            labelbottom=False,labelright=False,labelleft=False)

#ax.set_xlabel('Feature',family='sans-serif')

plt.savefig('feature_importance/X1_%s.png'%(feature),dpi=300)
#plt.show()
