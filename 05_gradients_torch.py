import torch

# Linear Regression
# f=w*x [f=2*x]
x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# model prediction
def forward(x):
    return w*x

# loss MSE
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

# Training
learning_rate=0.001
n_iters=100
for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    #loss
    l=loss(y,y_pred)
    #gradient=backward pass
    l.backward() # gradient of loss dl/dw
    #update weight
    with torch.no_grad():
        w-=learning_rate*w.grad
    #empty gradients (l.backward accumulate into w.grad)
    w.grad.zero_()
    if epoch%10==0:
        print('epoch ',epoch,' w: ',w, 'loss: ',l)
