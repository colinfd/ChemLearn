import torch
from torch.utils.data import DataLoader
from dataset import PdosDataset
from model import PdosModel, VariationalModel
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_

lr = 1e-3
weight_decay = 0.
n_epochs = 150
batch_size = 8
variational = True
std_dev_multiplier = 0.2
variational_loss_weight = 0.2
use_moments = False

split_type = 'comp_rxn'


dataset = PdosDataset(data_subset = 'train', split_type=split_type)
dataloader = DataLoader(dataset, batch_size = batch_size)
loss_fn = torch.nn.MSELoss()
val_dataset = PdosDataset(data_subset='dev')
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))


if variational == False:
        model = PdosModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


        losses = np.zeros(n_epochs)
        val_losses = np.zeros(n_epochs)
        for epoch in range(n_epochs):
                epoch_loss = np.zeros(len(dataloader))
                model.train()
                for i, batch in enumerate(dataloader):
                        X, y = batch
                        prediction = model(X)
                        loss = loss_fn(prediction, y)
                        optimizer.zero_grad()
                        loss.backward()
                        # clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

                        epoch_loss[i] = loss.detach().cpu().numpy()
                losses[epoch] = np.mean(epoch_loss)

                model.eval()
                for i, batch in enumerate(val_loader):
                        assert i==0
                        X, y = batch
                        prediction = model(X)
                        MSE = loss_fn(prediction, y).detach().cpu().numpy()
                        MAE = torch.mean(torch.abs((prediction-y))).detach().cpu().numpy()

                val_losses[epoch] = MSE
                print('Begin Epoch {}'.format(epoch+1))
                print('train: ', losses[epoch])
                print('val: ', MSE, MAE)
else:
        model = VariationalModel(std_dev_multiplier=std_dev_multiplier, use_moments=use_moments)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = np.zeros(n_epochs)
        val_losses = np.zeros(n_epochs)
        for epoch in range(n_epochs):
                if epoch == 2: torch.save(model,'test.pt')
                epoch_loss = np.zeros(len(dataloader))
                model.train()
                for i, batch in enumerate(dataloader):
                        X, y = batch
                        prediction, mus, logvars = model(X)
                        MSE = loss_fn(prediction, y)
                        KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
                        loss = MSE + KLD*variational_loss_weight
                        optimizer.zero_grad()
                        loss.backward()
                        # clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

                        epoch_loss[i] = MSE.detach().cpu().numpy()
                losses[epoch] = np.mean(epoch_loss)

                model.eval()
                for i, batch in enumerate(val_loader):
                        assert i==0
                        X, y = batch
                        prediction, mus, logvars = model(X)
                        MSE = loss_fn(prediction, y).detach().cpu().numpy()
                        MAE = torch.mean(torch.abs((prediction-y))).detach().cpu().numpy()

                val_losses[epoch] = MSE
                print('Begin Epoch {}'.format(epoch+1))
                print('train: ', losses[epoch])
                print('val: ', MSE, MAE)


x_plot = np.arange(n_epochs) + 1
plt.plot(x_plot, losses, label='Train Loss')
plt.plot(x_plot, val_losses, '--', label='Validation Loss')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error (eV^2)')

torch.save(model,'model_%s.pt'%split_type)
plt.show()




































