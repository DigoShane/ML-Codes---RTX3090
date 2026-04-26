import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use("TkAgg")   # must come BEFORE pyplot import

import matplotlib.pyplot as plt

# ------------------------
# Device (GPU if available)
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# Problem setup
# ------------------------
k = 7  # try k = 1 and k = 10
N_f = 1000  # IMPORTANT: reduced from 100000 (too large for local GPU/CPU)

def f(x):
    return (k*np.pi)**2 * torch.sin(k*np.pi*x)

def exact_solution(x):
    return torch.sin(k*np.pi*x)

# ------------------------
# PINN Model
# ------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PINN().to(device)

# ------------------------
# Collocation + BC points (FIXED BUG HERE)
# ------------------------
x_f = torch.linspace(0, 1, N_f).view(-1,1).to(device)

x_b = torch.tensor([[0.0],[1.0]], device=device)
u_b = torch.tensor([[0.0],[0.0]], device=device)

# ------------------------
# PDE residual
# ------------------------
def pde_residual(x):
    x.requires_grad_(True)

    u = model(x)

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    return -u_xx - f(x)

# ------------------------
# Loss function
# ------------------------
def loss_fn():
    res = pde_residual(x_f)
    loss_pde = torch.mean(res**2)

    u_pred = model(x_b)
    loss_bc = torch.mean((u_pred - u_b)**2)

    return loss_pde + loss_bc

# ------------------------
# Optimizer
# ------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------
# Training
# ------------------------
epochs = 5000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()

    loss = loss_fn()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4e}")

# ------------------------
# Evaluation
# ------------------------
x_test = torch.linspace(0,1,200).view(-1,1).to(device)

with torch.no_grad():
    u_pred = model(x_test).cpu().numpy()

u_exact = exact_solution(x_test).cpu().numpy()

# ------------------------
# Plot solution
# ------------------------
plt.figure(figsize=(6,4))
plt.plot(x_test.cpu(), u_exact, label="Exact")
plt.plot(x_test.cpu(), u_pred, '--', label="PINN")
plt.title(f"k = {k}")
plt.legend()
plt.grid()
plt.show()

# ------------------------
# Error
# ------------------------
error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
print(f"Relative L2 Error: {error:.2e}")

# ------------------------
# Plot loss
# ------------------------
plt.figure()
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.show()