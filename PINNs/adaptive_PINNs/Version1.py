# ============================================================
# VERSION 1:
# FIXED DOMAIN-DECOMPOSED PINN
#
# PDE:
#     -u''-k^2u = f(x)
#
# Exact solution:
#     u(x) = sin(k*pi*x)
#
# Domain:
#     [0,1] = [0,a] U [a,1]
#
# One neural network per subdomain.
# Continuity enforced weakly at interface.
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os
import shutil

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# OUTPUT FOLDER
# ============================================================

output_folder = "Version1_evals"

# Delete old folder if it exists
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# Create fresh folder
os.makedirs(output_folder)

# ============================================================
# PROBLEM SETUP
# ============================================================

omega = 15
helmholtz_k = 20

def exact_solution(x):
    return torch.sin(omega * np.pi * x)

def forcing(x):

    return ( (omega * np.pi)**2 - helmholtz_k**2 ) * torch.sin(omega * np.pi * x)

# ============================================================
# DOMAIN DECOMPOSITION
# ============================================================

a = 0.5
# Left domain: [0,a]
# Right domain: [a,1]

N_f = 200

x_left = torch.linspace(0, a, N_f).view(-1,1).to(device)
x_right = torch.linspace(a, 1, N_f).view(-1,1).to(device)

# Boundary points
x0 = torch.tensor([[0.0]], device=device)
x1 = torch.tensor([[1.0]], device=device)

# Interface point
xa = torch.tensor([[a]], device=device)

# ============================================================
# PINN MODEL
# ============================================================

class PINN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( nn.Linear(1,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1) )

    def forward(self, x):
        return self.net(x)

# ============================================================
# TWO SUBDOMAIN NETWORKS
# ============================================================

model_left = PINN().to(device)
model_right = PINN().to(device)

# ============================================================
# DERIVATIVES
# ============================================================

def second_derivative(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_xx

# ============================================================
# PDE RESIDUAL
# ============================================================

def pde_residual(model, x):

    u = model(x)
    u_xx = second_derivative(model, x)

    return -u_xx - helmholtz_k**2 * u - forcing(x)

# ============================================================
# LOSS FUNCTION
# ============================================================

def loss_function():

    # PDE LOSS
    loss_pde_left = torch.mean(pde_residual(model_left, x_left)**2)
    loss_pde_right = torch.mean(pde_residual(model_right, x_right)**2)
    loss_pde = loss_pde_left + loss_pde_right

    # BOUNDARY CONDITIONS
    # u(0)=0, u(1)=0
    loss_bc = (torch.mean(model_left(x0)**2) + torch.mean(model_right(x1)**2))

    # INTERFACE CONTINUITY
    # u_left(a) = u_right(a)
    loss_interface = torch.mean((model_left(xa) - model_right(xa))**2)

    # TOTAL LOSS
    loss = (loss_pde + loss_bc + 100000*loss_interface)

    return loss

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam( list(model_left.parameters()) + list(model_right.parameters()), lr=1e-3)

# ============================================================
# SAVE CURRENT SOLUTION
# ============================================================

def save_current_solution(epoch):
    x_test = torch.linspace(0,1,1000).view(-1,1).to(device)
    u_pred = []
    with torch.no_grad():
        for x in x_test:
            if x <= a:
                val = model_left(x.view(1,1))
            else:
                val = model_right(x.view(1,1))
            u_pred.append(val.item())
    u_pred = np.array(u_pred)
    u_exact = exact_solution(x_test).cpu().numpy()
    plt.figure(figsize=(8,4))
    plt.plot( x_test.cpu().numpy(), u_exact, label="Exact")
    plt.plot( x_test.cpu().numpy(), u_pred, '--', label="PINN Approximation")
    plt.axvline( a, color='black', linestyle=':')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"Epoch {epoch}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_folder}/epoch_{epoch:05d}.png", dpi=150, bbox_inches='tight')

    plt.close()


# ============================================================
# TRAINING
# ============================================================

epochs = 5000

loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_function()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d} | Loss = {loss.item():.4e}")
        save_current_solution(epoch)

# ============================================================
# EVALUATION
# ============================================================

x_test = torch.linspace(0,1,1000).view(-1,1).to(device)
u_pred = []

with torch.no_grad():
    for x in x_test:
        if x <= a:
            val = model_left(x.view(1,1))
        else:
            val = model_right(x.view(1,1))
        u_pred.append(val.item())

u_pred = np.array(u_pred)
u_exact = exact_solution(x_test).cpu().numpy()

# ============================================================
# RELATIVE L2 ERROR
# ============================================================

error = np.linalg.norm(u_exact.flatten() - u_pred)
error /= np.linalg.norm(u_exact.flatten())

print("\nRelative L2 Error =", error)

# ============================================================
# PLOT SOLUTION
# ============================================================

plt.figure(figsize=(8,4))
plt.plot(x_test.cpu().numpy(), u_exact, label="Exact")
plt.plot(x_test.cpu().numpy(), u_pred, '--', label="Domain-Decomposed PINN")
plt.axvline(a, color='black', linestyle=':', label='Interface')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Version 1: Helmholtz PINN")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.show()