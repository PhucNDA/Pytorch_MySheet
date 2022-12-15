import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device config
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper params
input_size=784 # 28x28
hidden_size=20
num_classes=10 #0-9
num_epochs=14
batch_size=100
learning_rate=0.001

#MNIST
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),)

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

examples=iter(train_loader)
samples,labels=examples.next()
print(samples.shape,labels.shape)

''' Showing samples
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
plt.show()
'''

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNetwork,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        return out

model=NeuralNetwork(input_size,hidden_size,num_classes).to(device)

#loss and optimizer
criterion=nn.CrossEntropyLoss() # already applied softmax
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#training procedure
n_total_step=len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        #forward
        outputs=model(images)
        loss=criterion(outputs,labels)
        #backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i%100 == 0:
            print('Epoch ',epoch,'..Iter: ',i,'/',n_total_step,' LOSS: ',loss.item())


#Testing procedure
with torch.no_grad():
    n_correct=0
    n_samples=0
    for i, (images,labels) in enumerate(test_loader):
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        outputs=model(images)
        
        #Retrieving output
        rst=[]
        for tsr in outputs:
            #Softmaxing
            softmax=torch.softmax(tsr,axis=0)
            rst.append(torch.argmax(softmax))
        rst=torch.from_numpy(np.array(rst))

        n_samples+=labels.shape[0]
        n_correct+=(rst==labels).sum().item()

    acc=100 * (n_correct)/n_samples
    print('ACC: ',acc)

