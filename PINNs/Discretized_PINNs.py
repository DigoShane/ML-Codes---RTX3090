import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DEVICE SETUP (GPU if available)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# PROBLEM SETUP
# PDE: -u'' = f(x), exact solution u = sin(k*pi*x)
# ============================================================
k = 7

def f(x):
    return (k*np.pi)**2 * torch.sin(k*np.pi*x)

def exact_solution(x):
    return torch.sin(k*np.pi*x)

# ============================================================
# DOMAIN DECOMPOSITION
# Split [0,1] → [0,a], [a,b], [b,1]
# ============================================================
a = 0.33
b = 0.66

# Number of collocation points per subdomain
N = 100

x1 = torch.linspace(0, a, N).view(-1,1).to(device)
x2 = torch.linspace(a, b, N).view(-1,1).to(device)
x3 = torch.linspace(b, 1, N).view(-1,1).to(device)

# Interface points
xa = torch.tensor([[a]], device=device)
xb = torch.tensor([[b]], device=device)

# Boundary points
x0 = torch.tensor([[0.0]], device=device)
x1_end = torch.tensor([[1.0]], device=device)

# ============================================================
# NEURAL NETWORK MODEL (used for each subdomain)
# ============================================================
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

# Create 3 independent networks
model1 = PINN().to(device)
model2 = PINN().to(device)
model3 = PINN().to(device)

# ============================================================
# DERIVATIVE HELPERS (for PDE + interface conditions)
# ============================================================
def grad(model, x):
    """Compute first derivative du/dx"""
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return u_x

def pde_residual(model, x):
    """Compute PDE residual: -u'' - f(x)"""
    x.requires_grad_(True)
    u = model(x)

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return -u_xx - f(x)

# ============================================================
# LOSS FUNCTION
# ============================================================
def loss_fn():
    # ------------------------
    # PDE residual (each subdomain)
    # ------------------------
    loss_pde = (
        torch.mean(pde_residual(model1, x1)**2) +
        torch.mean(pde_residual(model2, x2)**2) +
        torch.mean(pde_residual(model3, x3)**2)
    )

    # ------------------------
    # Boundary conditions
    # u(0)=0, u(1)=0
    # ------------------------
    loss_bc = torch.mean(model1(x0)**2) + torch.mean(model3(x1_end)**2)

    # ------------------------
    # Interface continuity (C0)
    # u1(a) = u2(a), u2(b) = u3(b)
    # ------------------------
    loss_interface = (
        torch.mean((model1(xa) - model2(xa))**2) +
        torch.mean((model2(xb) - model3(xb))**2)
    )

    ## ------------------------
    ## Optional: derivative continuity (C1)
    ## improves smoothness
    ## ------------------------
    #loss_grad = (
    #    torch.mean((grad(model1, xa) - grad(model2, xa))**2) +
    #    torch.mean((grad(model2, xb) - grad(model3, xb))**2)
    #)

    # ------------------------
    # Total loss
    # (interface terms weighted higher)
    # ------------------------
    loss = loss_pde + loss_bc + 10*loss_interface + loss_grad

    return loss

# ============================================================
# OPTIMIZER
# ============================================================
optimizer = torch.optim.Adam(
    list(model1.parameters()) +
    list(model2.parameters()) +
    list(model3.parameters()),
    lr=1e-3
)

# ============================================================
# TRAINING LOOP
# ============================================================
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

# ============================================================
# EVALUATION
# Stitch solution across domains
# ============================================================
x_test = torch.linspace(0,1,500).view(-1,1).to(device)

u_pred = []

with torch.no_grad():
    for x in x_test:
        if x <= a:
            u_pred.append(model1(x.view(1,1)).item())
        elif x <= b:
            u_pred.append(model2(x.view(1,1)).item())
        else:
            u_pred.append(model3(x.view(1,1)).item())

u_pred = np.array(u_pred)

u_exact = exact_solution(x_test).cpu().numpy()

# ============================================================
# PLOT SOLUTION
# ============================================================
plt.figure(figsize=(6,4))
plt.plot(x_test.cpu(), u_exact, label="Exact")
plt.plot(x_test.cpu(), u_pred, '--', label="Multi-domain PINN")
plt.axvline(a, color='gray', linestyle=':')
plt.axvline(b, color='gray', linestyle=':')
plt.title(f"Domain-Decomposed PINN (k={k})")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# ERROR
# ============================================================
error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
print(f"Relative L2 Error: {error:.2e}")

# ============================================================
# PLOT LOSS
# ============================================================
plt.figure()
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.show()