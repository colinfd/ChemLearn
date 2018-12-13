import torch
from dataset import PdosDataset
import numpy as np
import matplotlib.pyplot as plt
import PlotFormat 

colors=PlotFormat.muted_colors

def parity(y_train,train_preds,y_test,test_preds,model,features):
    
    plt.figure(figsize=(3.5,3))
    train_mae = np.abs(y_train-train_preds).mean()
    test_mae = np.abs(y_test-test_preds).mean()
    print('Test MAE: %4.2f'%(test_mae))
    print('Train MAE: %4.2f'%(train_mae))
    plt.plot(y_train,train_preds,'.',c='grey',ms=3,label='Train MAE: %4.2f'%train_mae,alpha=1)
    plt.plot(y_test,test_preds,'.',c=colors[model],ms=3,label='Test MAE: %4.2f'%test_mae,alpha=1)

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
    plt.savefig('../parity/'+model+'_'+features+'_'+split_type+'_parity.pdf')

#Change train test split 
split_type = 'comp_rxn'#comp_rxn,random,comp or rxn

test_dataset = PdosDataset(data_subset = 'test')
train_dataset = PdosDataset(data_subset = 'train')
model = torch.load('model_%s.pt'%split_type)
model.eval()

X_test, y_test = test_dataset[range(test_dataset.X_data.shape[0])]
X_train, y_train = train_dataset[range(train_dataset.X_data.shape[0])]

y_pred,asdf,asdf = model(X_test)
y_train_pred,asdf,asdf = model(X_train)
MAE = np.abs(y_pred.detach().numpy()[:,0]-y_test.detach().numpy()[:]).mean()

print("Test error for %s split = %.3f"%(split_type,MAE))

#Make parity plot for comp_rxn only, could change
if split_type=='comp_rxn':
    y_train = y_train.detach().numpy()[0,:]
    y_pred = y_pred.detach().numpy()[:,0]
    y_train_pred = y_train_pred.detach().numpy()[:,0]
    y_test = y_test.detach().numpy()[0,:]
    parity(y_train,y_train_pred,y_test,y_pred,'cnn','pdos')
