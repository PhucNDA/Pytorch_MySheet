"""
epoch=1 forward and backward pass of all training samples
batch_size=number of training samples in one forward and backward pass
number of iterations=number of passes, each pass using [batch_size] number of samples

e.g 100 samples, batch_size=20 --> 100/20=5 iterations for 1 epoch
"""
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self,transform=None):
        #dataloading
        xy=np.loadtxt('wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x=xy[:,1:]
        self.y=xy[:,0]
        self.n_samples=xy.shape[0]

        self.transform=transform

    def __getitem__(self,index):
        #dataset[0]
        sample = self.x[index],self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return sample

    def __len__(self):
        #len(dataset)
        return self.n_samples


class ToTensor:
    def __call__(self,sample):
        inputs, targets=sample
        return torch.from_numpy(np.array(inputs)), torch.from_numpy(np.array(targets))

dataset=WineDataset(transform=ToTensor())
first_data=dataset[0]
features,labels=first_data
print(type(features),type(labels))
