import torch
import torch.nn as nn #Neural Network+Autograd engine
from time import perf_counter
#from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
#import requests
#import os

## check if GPU is available and use it; otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# N is a Neural Network - This is exactly the network used by Lagaris et al. 1997
N = nn.Sequential(nn.Linear(1, 50), nn.Sigmoid(), nn.Linear(50,1, bias=False))
# In layer 1, it takes an input scalr x and returns:
# z^{1}=W^{1}x+b^{1}. W^{1}∈R^{50×1}, 𝑏^^{1}∈𝑅^{50}
# Then it passes it through a sigmoid: a^{1} = σ(z^{1})
# Then it passes it through a second layer with no bias:
# N(x)=W^{2}a^{1}, where 𝑊^{2}∈𝑅^{1x50}
# Thus the NN is of the form: N(x)=W^{2}⋅σ(W^{1}x+b^{1})
# Input: scalar 𝑥
# Hidden layer: 50 neurons
# Activation: sigmoid
# Output: scalar

# Initial condition
A = 0

# The Psi_t function
Psi_t = lambda x: A + x * N(x)

# The right hand side function
f = lambda x, Psi: torch.exp(-x / 5.0) * torch.cos(x) - Psi / 5.0

# The loss function
def loss(x):
        x.requires_grad = True #enables autograd.
        outputs = Psi_t(x) # PyTorch builds a computational graph : x → N(x) → x * N(x) → Psi_t(x)
        Psi_t_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
                  # d/dx command  #fn is Psi_t #wrt x 
                  # Autograd computes:
                  # 𝑑/dx (∑_i y_i.v_i), where v = grad_outputs. So by setting v_i=1,
                  # You get:  𝑑/dx (∑_i y_i), 👉 i.e., sum of gradients across batch
        return torch.mean((Psi_t_x - f(x, outputs)) ** 2)


# Optimize (same algorithm as in Lagaris)
optimizer = torch.optim.LBFGS(N.parameters())

# The collocation points used by Lagaris
x = torch.Tensor(np.linspace(0, 2, 100)[:, None])

# Run the optimizer
def closure():
    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l
    
for i in range(10):
    optimizer.step(closure)

# Let's compare the result to the true solution
xx = np.linspace(0, 2, 100)[:, None]
with torch.no_grad():
    yy = Psi_t(torch.Tensor(xx)).numpy()
yt = np.exp(-xx / 5.0) * np.sin(xx)

fig, ax = plt.subplots(dpi=100)
ax.plot(xx, yt, label='True')
ax.plot(xx, yy, '--', label='Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$\Psi(x)$')
plt.legend(loc='best');

# We need to reinitialize the network
N = nn.Sequential(nn.Linear(1, 50), nn.Sigmoid(), nn.Linear(50,1, bias=False))

# Let's see now if a stochastic optimizer makes a difference
adam = torch.optim.Adam(N.parameters(), lr=0.01)

# The batch size you want to use (how many points to use per iteration)
n_batch = 5

# The maximum number of iterations to do
max_it = 1000

for i in range(max_it):
    # Randomly pick n_batch random x's:
    x = 2 * torch.rand(n_batch, 1)
    # Zero-out the gradient buffers
    adam.zero_grad()
    # Evaluate the loss
    l = loss(x)
    # Calculate the gradients
    l.backward()
    # Update the network
    adam.step()
    # Print the iteration number
    if i % 100 == 99:
        print(i+1)

# Let's compare the result to the true solution
xx = np.linspace(0, 2, 100)[:, None]
with torch.no_grad():
    yy = Psi_t(torch.Tensor(xx)).numpy()
yt = np.exp(-xx / 5.0) * np.sin(xx)

fig, ax = plt.subplots(dpi=100)
ax.plot(xx, yt, label='True')
ax.plot(xx, yy, '--', label='Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$\Psi(x)$')
plt.legend(loc='best');