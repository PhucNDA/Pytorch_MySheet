"""
epoch=1 forward and backward pass of all training samples
batch_size=number of training samples in one forward and backward pass
number of iterations=number of passes, each pass using [batch_size] number of samples

e.g 100 samples, batch_size=20 --> 100/20=5 iterations for 1 epoch
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        #dataloading
        xy=np.loadtxt('wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x=torch.Tensor(xy[:,1:])
        self.y=torch.Tensor(xy[:,0])
        self.y-=1#OHE (0,1,2)
        self.y=self.y.type(torch.LongTensor)
        self.n_samples=xy.shape[0]
    def __getitem__(self,index):
        #dataset[0]
        return self.x[index],self.y[index]
    def __len__(self):
        #len(dataset)
        return self.n_samples

class NeuralNet(nn.Module):
    def __init__(self,n_input_features):
        super(NeuralNet,self).__init__()
        self.linear=nn.Linear(n_input_features,10)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(10,3)# wine dataset has 3 classes
    def forward(self,x):
        out=self.linear(x)
        out=self.relu(out)
        out=self.linear2(out)
        return out


datasett=WineDataset()
# print(datasett[0])
dataloader=DataLoader(dataset=datasett,batch_size=4,shuffle=True)

# training loop
num_epochs=100
batch_size=64
learning_rate=0.001

total_samples=len(datasett)
n_iterations=math.ceil(total_samples/64)
print(total_samples, n_iterations)
n_samples, n_features=datasett.x.shape
print(n_features)
#model
model=NeuralNet(n_features)
criterion=nn.CrossEntropyLoss() # applies softmax
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for i, (inputs,labels) in enumerate(dataloader):
        # forward
        y_predicted=model(inputs)
        # backward
        loss=criterion(y_predicted,labels)
        loss.backward()
        # update
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch ',epoch,'..Iter: ',i,'/',n_iterations,' LOSS: ',loss.item())
        
