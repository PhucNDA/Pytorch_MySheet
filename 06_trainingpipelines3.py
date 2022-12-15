# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#  - forward pass: compute prediction
#  - backward pass: gradients
#  - update weights
import torch
import torch.nn as nn
# Linear Regression
# f=w*x [f=2*x]
x=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

n_samples, n_features=x.shape
print(n_samples, n_features)

input_size=n_features
output_size=n_features

class LinearRegression(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(LinearRegression,self).__init__()
        #define layers
        self.lin=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        return self.lin(x)

model=LinearRegression(input_size,output_size)

# Training
learning_rate=0.01
n_iters=100
loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=model(x)
    #loss
    l=loss(y,y_pred)
    #gradient=backward pass
    l.backward() # gradient of loss dl/dw
    #update weight
    optimizer.step()
    #empty gradients (l.backward accumulate into w.grad)
    optimizer.zero_grad()
    if epoch%10==0:
        [w,b]=model.parameters()
        print('epoch ',epoch,' w: ',w.item(), 'loss: ',l.item())
