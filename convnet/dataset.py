import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
#from scipy.stats import moment

class PdosDataset(Dataset):
        def __init__(self, data_path = '.', data_subset = 'train', noise_scale=0.05, n_moments=3,split_type='comp_rxn'):
                super(PdosDataset, self).__init__()
                self.data_subset = data_subset
                self.noise_scale = noise_scale
                self.dtype = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'
                X_data = np.load('data/X_%s_%s.npy'%(split_type,data_subset))
                y_data = np.load('data/y_%s_%s.npy'%(split_type,data_subset))
                #For learning curves
                i = 10
                if data_subset=='train':
                    self.X_data = X_data[0:X_data.shape[0]//10*i]
                    self.y_data = y_data[0:X_data.shape[0]//10*i]
                else:
                    self.X_data = X_data
                    self.y_data = y_data
                #self.X_data = np.load('data/X_{}.npy'.format(data_subset)) # [:, :-2]
                #load = np.load('data/X_{}.npy'.format(data_subset)) # [:, :-2]
                #self.X_data = np.zeros((load.shape[0], 4, load.shape[2]))
                #self.X_data[:,:2,:] = load
                #self.X_data[:,2] = self.X_data[:,0] + self.X_data[:,1]
                #self.X_data[:,3] = self.X_data[:,0] * self.X_data[:,1]
                # ncols = self.X_data.shape[1]
                # self.X_data = self.X_data[:,np.arange(0, ncols, 6)]
                #self.y_data = np.load('data/y_{}.npy'.format(data_subset))



                # self.moments = np.zeros[(self.X_data.shape[0], n_moments+1)]
                # for i in range(n_moments):
                #       self.moments[:,i] = moment(self.X_data, axis=1, moment=i)
                # print(np.max(self.y_data))
                # print(np.min(self.y_data))
                # print(np.max(self.X_data))
                # print(np.min(self.X_data))
                # asd0


        def __len__(self):
                assert self.y_data.shape[0] == self.X_data.shape[0]
                return self.X_data.shape[0]


        def __getitem__(self, idx):
                X = self.X_data[idx, :]
                #if self.data_subset == 'train':
                #       noise = np.random.normal(scale=self.noise_scale, size=X.shape)
                #       X += noise
                X = torch.from_numpy(X).type(self.dtype)
                y = torch.from_numpy(np.array([self.y_data[idx]])).type(self.dtype)
                return (X, y)





if __name__ == '__main__':
        dataset = PdosDataset()
        a = dataset[0]
        print(a[0].shape)
        print(a[1].shape)
