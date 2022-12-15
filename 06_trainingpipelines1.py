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
x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# model prediction
def forward(x):
    return w*x

# Training
learning_rate=0.01
n_iters=100
loss=nn.MSELoss()
optimizer=torch.optim.SGD([w], lr=learning_rate)
for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    #loss
    l=loss(y,y_pred)
    #gradient=backward pass
    l.backward() # gradient of loss dl/dw
    #update weight
    optimizer.step()
    #empty gradients (l.backward accumulate into w.grad)
    optimizer.zero_grad()
    if epoch%10==0:
        print('epoch ',epoch,' w: ',w, 'loss: ',l)
