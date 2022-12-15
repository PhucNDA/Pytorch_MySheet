import numpy as np

# Linear Regression
# f=w*x [f=2*x]
x=np.array([1,2,3,4],dtype=np.float32)
y=np.array([2,4,6,8],dtype=np.float32)
w=0.0

# model prediction
def forward(x):
    return w*x

# loss MSE
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x-y)**2
# dJ/dw=1/N * 2*x(w*x-y)
def gradient(x,y,y_predict):
    return np.dot(2*x,y_predict-y).mean()

# Training
learning_rate=0.001
n_iters=20
for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    #loss
    l=loss(y,y_pred)
    #gradient
    dw=gradient(x,y,y_pred)
    #update weight
    w-=learning_rate*dw

    print('epoch ',epoch,' w: ',w, 'loss: ',l)
