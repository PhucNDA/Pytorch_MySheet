import torch

x=torch.randn(5, requires_grad=True)
print(x)

y=x+2
print(y)

z=y*y*2
# z=z.mean()
print(z)

z.backward(torch.tensor([0.1]*5,dtype=torch.float32))
print(x.grad)