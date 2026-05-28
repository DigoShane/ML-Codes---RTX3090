# ============================================================
# VERSION 2:
# ADAPTIVE DOMAIN-DECOMPOSED PINN
#
# Start with ONE network on [0,1]
#
# If optimization stagnates:
#     split into:
#         [0,a] and [a,1]
#
# Child networks inherit parent weights.
#
# PDE:
#     -u'' - k^2 u = f
#
# Exact:
#     u = sin(omega*pi*x)
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
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

output_folder = "Version2_evals"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

# ============================================================
# PDE SETUP
# ============================================================

omega = 15
helmholtz_k = 20

def exact_solution(x):
    return torch.sin(omega * np.pi * x)

def forcing(x):

    return ((omega*np.pi)**2 - helmholtz_k**2)*torch.sin(omega*np.pi*x)

# ============================================================
# GLOBAL DOMAIN
# ============================================================

N_f = 400
x_global = torch.linspace(0,1,N_f).view(-1,1).to(device)

x0 = torch.tensor([[0.0]], device=device)
x1 = torch.tensor([[1.0]], device=device)

# ============================================================
# PINN MODEL
# ============================================================

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# DERIVATIVES
# ============================================================

def first_derivative(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    return u_x

def second_derivative(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad( u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad( u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_xx

# ============================================================
# PDE RESIDUAL
# ============================================================

def pde_residual(model, x):
    u = model(x)
    u_xx = second_derivative(model, x)
    return -u_xx - helmholtz_k**2 * u - forcing(x)

# ============================================================
# SAVE CURRENT SOLUTION
# ============================================================

def save_solution(epoch, models, split=False):
    x_test = torch.linspace(0,1,1000).view(-1,1).to(device)
    u_pred = []
    with torch.no_grad():
        if not split:
            model = models[0]
            u_pred = model(x_test).cpu().numpy()
        else:
            model_left, model_right = models
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
    plt.plot( x_test.cpu().numpy(), u_pred, '--', label="PINN")
    if split:
        plt.axvline(a, color='black', linestyle=':')
    plt.legend()
    plt.grid()
    plt.title(f"Epoch {epoch}")
    plt.savefig(f"{output_folder}/epoch_{epoch:05d}.png", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# INITIAL GLOBAL MODEL
# ============================================================

global_model = PINN().to(device)
optimizer = torch.optim.Adam( global_model.parameters(), lr=1e-3)

# ============================================================
# GLOBAL LOSS
# ============================================================

def global_loss():
    loss_pde = torch.mean( pde_residual(global_model, x_global)**2 )
    loss_bc = ( torch.mean(global_model(x0)**2) + torch.mean(global_model(x1)**2) )

    return loss_pde + loss_bc

# ============================================================
# TRAIN GLOBAL MODEL
# ============================================================
epochs_stage1 = 5000 #we allow atmost 5000 training steps just on 1 nn.
loss_history = []
window = 500 # we check avg loss over 'window' iterations; 500 here.
stagnation_tol = 1e-3
split_triggered = False

for epoch in range(epochs_stage1):
    optimizer.zero_grad()
    loss = global_loss()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(f"[GLOBAL] Epoch {epoch:5d} | Loss = {loss.item():.4e}")
        save_solution( epoch, [global_model], split=False)

    # --------------------------------------------------------
    # STAGNATION CHECK
    # --------------------------------------------------------

    if epoch > 2*window:
        old_loss = np.mean(loss_history[-2*window:-window])
        new_loss = np.mean(loss_history[-window:])
        relative_improvement = (old_loss - new_loss) / old_loss

        if (relative_improvement < stagnation_tol and new_loss > 1e-4):
            print("\n================================================")
            print("STAGNATION DETECTED")
            print("TRIGGERING DOMAIN SPLIT")
            print("================================================\n")
            split_triggered = True
            break

# SPLIT DOMAIN
a = 0.5
x_left = torch.linspace(0,a,N_f//2).view(-1,1).to(device)
x_right = torch.linspace(a,1,N_f//2).view(-1,1).to(device)
xa = torch.tensor([[a]], device=device)

# CREATE CHILD NETWORKS
model_left = PINN().to(device)
model_right = PINN().to(device)

# PARENT-TO-CHILD INHERITANCE
model_left.load_state_dict( copy.deepcopy(global_model.state_dict()))
model_right.load_state_dict( copy.deepcopy(global_model.state_dict()))

# NEW OPTIMIZER
optimizer = torch.optim.Adam(list(model_left.parameters()) + list(model_right.parameters()), lr=1e-3)

# DECOMPOSED LOSS
def decomposed_loss():
    # PDE
    loss_pde = ( torch.mean( pde_residual(model_left, x_left)**2) + torch.mean(pde_residual(model_right, x_right)**2))
    # BC
    loss_bc = ( torch.mean(model_left(x0)**2) + torch.mean(model_right(x1)**2))
    # INTERFACE CONTINUITY
    loss_interface = torch.mean( (model_left(xa) - model_right(xa))**2)
    # FLUX CONTINUITY
    loss_flux = torch.mean((first_derivative(model_left, xa) - first_derivative(model_right, xa))**2)

    return ( loss_pde + loss_bc + 1000.0 * loss_interface + 100.0 * loss_flux)

# ============================================================
# TRAIN DECOMPOSED SYSTEM
# ============================================================
epochs_stage2 = 10000
for epoch in range(epochs_stage2):
    optimizer.zero_grad()
    loss = decomposed_loss()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"[DECOMPOSED] Epoch {epoch:5d} | Loss = {loss.item():.4e}")
        save_solution( epoch + epochs_stage1, [model_left, model_right], split=True)

# ============================================================
# FINAL EVALUATION
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
error = np.linalg.norm( u_exact.flatten() - u_pred)
error /= np.linalg.norm(u_exact.flatten())

print("\n================================================")
print(f"FINAL RELATIVE L2 ERROR = {error:.4e}")
print("================================================")

# ============================================================
# FINAL PLOT
# ============================================================

plt.figure(figsize=(8,4))
plt.plot( x_test.cpu().numpy(), u_exact, label="Exact")
plt.plot( x_test.cpu().numpy(), u_pred, '--', label="Adaptive PINN")
plt.axvline( a, color='black', linestyle=':')
plt.legend()
plt.grid()
plt.title("Version 2: Adaptive Domain Decomposition")
plt.show()