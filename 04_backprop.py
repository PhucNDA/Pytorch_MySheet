import torch

x=torch.tensor(1.0)
y=torch.tensor(2.0)
w=torch.tensor(1.0,requires_grad=True)

# Forward pass
y_hat=w*x
loss=(y_hat-y)**2
print(loss)

# Backward pass
loss.backward()
print(w.grad)

# Update weights
# Next forward & backward pass