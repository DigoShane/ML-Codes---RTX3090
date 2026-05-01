# We want to solve the problem for a damped oscillator ODE
# m y''(t) + \mu y'(t) + ky = 0
# y(0) = 1, y'(0) = 0

from PIL import Image

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Multi Layer Perceptron
class MLP(nn.Module):

    def __init__(self, arch):
        super(MLP, self).__init__()

        #Interpret arch(itecture) input. 
        if isinstance(arch, tuple):
            # Case 1: (num_layers, width)
            num_layers, width = arch
            assert num_layers >= 2, "Need at least input and output layer"
            
            # Example: (4, 32) → [1, 32, 32, 32, 1]
            layers = [1] + [width]*(num_layers-2) + [1]

        elif isinstance(arch, list):
            # Case 2: explicit list
            layers = arch
            assert len(layers) >= 2, "Need at least input and output size"

        else:
            raise ValueError("arch must be either tuple (a,b) or list [a1,...,an]")

        # Build network
        self.activation = nn.Tanh()

        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1])
            for i in range(len(layers)-1)
        ])

    def forward(self, t):
        for i in range(len(self.linears)-1):
            t = self.activation(self.linears[i](t))
        t = self.linears[-1](t)
        return t


def residual(model, t, params): # t: #collocation pts x 1. The last dim must match the 

    m, mu, k = params

    y = model(t)

    # print(t.shape)

    dydt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    # We could create a list of derivatives, dydt, dydx. the [0] says we are only picking the 1st entry in the list.
    # dydx1, dydx2, dydx3 = torch.autograd.grad(y, (x1,x2,x3), grad_outputs=torch.ones_like(y), create_graph=True)
    # create_graph : help wiht backpropagation. It tells me the sequence of turns i took to get to a point. 
    # Important for taking higher order derivatives.
    # grad_outputs: we have N collocation t = [t1,t2,..., tN] and y = [y1, y2, ..., yN].
    # Thus dy/dt is a jacobian matrix. BUT since y is a vector, PyTorch computes: grad_outputs^T⋅ ∂y/∂t 
	# since grad_outputs=torch.ones_like(y) = [1, 1, ..., 1] N times. Thus ​grad_outputs^T⋅ ∂y/∂t = [y'(t1), y'(t2), ..., y'(tN)]

    d2ydt2 = torch.autograd.grad(dydt, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    return m * d2ydt2 + mu * dydt + k * y

def boundary(model, tbc): # size nbc x 1
    ybc = model(tbc)

    dydt = torch.autograd.grad(ybc, tbc, grad_outputs=torch.ones_like(ybc), create_graph=True)[0]

    return ybc - 1, dydt # y(0) - 1 ->0, y'(0) -> 0.

def datapoints():
    
    tbc = torch.zeros(1, 1, device = device, requires_grad = True)
    tcol = torch.linspace(0, T, 50, device = device).reshape(50, 1)
    tcol.requires_grad_(True)

    return tbc, tcol

def train(tbc, tcol, params, model, optimizer):

    #set training mode
    model.train()

    # zero out gradients
    optimizer.zero_grad()

    # BC loss
    bc1, bc2 = boundary(model, tbc)
    lossbc = torch.mean(bc1**2) + torch.mean(bc2**2)

    # physics loss
    lossphy = torch.mean(residual(model, tcol, params)**2)

    # total loss
    lambda1 = 1e-1
    loss = lossphy + lambda1 * lossbc

    #back prop 
    loss.backward()

    #update parameters
    optimizer.step()

    return loss.item()

def exact(t, params):
    m, mu, k = params
    
    delta = mu/(2*m)
    omega0 = torch.sqrt(torch.tensor(k/m, device=t.device))

    omega = torch.sqrt(omega0**2 - delta**2)
    phi = torch.arctan(-delta/omega)
    A = 1/(2*torch.cos(phi))
    y = torch.exp(-delta*t)*2*A*torch.cos(omega*t + phi)
    return y

if __name__ == "__main__":
    m, mu, k = 1, 4, 50
    params = (m, mu, k)
    T = 5

    arch = (4, 32)
    model = MLP(arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, )

    #generate eval points. 
    tbc, tcol = datapoints()

    epochs = 10000
    train_losses = np.zeros(epochs)

    for t in range(epochs):
        train_losses[t] = train(tbc, tcol, params, model, optimizer)

        if t % 100 == 0:
            print(train_losses[t])
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(epochs), train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")

    t_test = torch.linspace(0, T, 200).reshape(-1,1).to(device)
    y_hat = model(t_test).detach().cpu()
    y_exact = exact(t_test, params)

    plt.figure()
    plt.plot(t_test.cpu(), y_hat.cpu(), "--")
    plt.plot(t_test.cpu(), y_exact.cpu())
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.legend(["PINN Prediction", "Exact Soln"])

    plt.show()

