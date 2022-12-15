import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x=np.array([1.0,2.0,0.5])
outputs=softmax(x)
print(outputs)

x=torch.Tensor([1.0,2.0,0.5])
outputs=torch.softmax(x,axis=0)
print(outputs)