import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual,predicted):
    loss=-np.sum(actual*np.log(predicted))
    return loss
    #normalize by dividing len vec

# y must be OHE
# 1 = [1,0,0]
# 2 = [0,1,0]
# 3 = [0,0,1]
Y=np.array([1,0,0])
Y_pred_good=np.array([0.7,0.2,0.1]) # softmax value
Y_pred_bad=np.array([0.1,0.3,0.6])
l1=cross_entropy(Y,Y_pred_good)
l2=cross_entropy(Y,Y_pred_bad)
print(l1,l2)

loss=nn.CrossEntropyLoss()
Y=torch.tensor([0])
# nsamples x nclasses || 2D arr
Y_pred_good=torch.tensor([[2.0,1.0,0.1]]) # raw values not applied Softmax
Y_pred_bad=torch.tensor([[0.5,2.0,0.3]])

l1=loss(Y_pred_good,Y)
l2=loss(Y_pred_bad,Y)
print(l1.item(),l2.item())