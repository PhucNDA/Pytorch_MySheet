# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#  - forward pass: compute prediction
#  - backward pass: gradients
#  - update weights
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) dataprep
bc=datasets.load_breast_cancer()
x,y=bc.data, bc.target
n_samples, n_features = x.shape
print('Samples and features: ', n_samples, n_features)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Zero mean and Unit variance
sc=StandardScaler()
sc=sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)

x_train=torch.Tensor(x_train.astype(np.float32))
x_test=torch.Tensor(x_test.astype(np.float32))
y_train=torch.Tensor(y_train.astype(np.float32))
y_test=torch.Tensor(y_test.astype(np.float32))

# flatten out
y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)
# 1) model
class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
    def forward(self,x):
        y_predicted=torch.sigmoid(self.linear(x))
        return y_predicted
model=LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate=0.01
criterion=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

# 3) training loop
num_epochs=100
for epoch in range(num_epochs):
    #forward pass & loss
    y_predicted=model(x_train)
    loss=criterion(y_predicted,y_train)

    #backward pass
    loss.backward()

    #updates
    optimizer.step()
    #zero gradients
    optimizer.zero_grad()
    
    if(epoch)%10==0:
        print('Epoch: ',epoch,' loss: ',loss.item())

#Evaluation
with torch.no_grad():
    y_predicted=model(x_test)
    y_predicted_cls=y_predicted.round()
    acc=y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print('Accuracy: ',acc.item())