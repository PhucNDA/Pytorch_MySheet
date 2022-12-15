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
    def __init__(self):
        #dataloading
        xy=np.loadtxt('wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x=torch.Tensor(xy[:,1:])
        self.y=torch.Tensor(xy[:,0])
        self.n_samples=xy.shape[0]
    def __getitem__(self,index):
        #dataset[0]
        return self.x[index],self.y[index]
    def __len__(self):
        #len(dataset)
        return self.n_samples

datasett=WineDataset()
print(datasett[0])
dataloader=DataLoader(dataset=datasett,batch_size=4,shuffle=True)

# training loop
num_epochs=2
total_samples=len(datasett)
n_iterations=math.ceil(total_samples/4)
print(total_samples, n_iterations)


for epoch in range(num_epochs):
    for i, (inputs,labels) in enumerate(dataloader):

        #forward and backward, update
        print('Epoch ',epoch,'..Iter: ',i,'/',n_iterations)
        print(inputs.shape)
