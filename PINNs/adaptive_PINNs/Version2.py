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
# IMPORTANT SERVICE ANNOUNCEMENT
# ============================================================
print("This code isn't really complete yet. There seem to be some underlying issue which havent been addressed yet.")
print("It doesn't seem to have anything to do with the domain decomposition. If you look at the folder for version2_eval.")
print("You will notice that at around 2000 epochs (well before decomposition), the curve just stagnates. It doesnt change anymore.")
print("I am moving forward hoping that this issue, while not having to do with the domain decomposition will be addressed by it.")
print("The reason is that there is a decrease in loss after decomposition, and loss decreasing is the only thing that matters.")
print("Maybe more decompositions lead to better/more loss. Also the stagnation of the weights was the reason why the domain decomposed.")

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
# SAVE RESIDUAL PLOT
# ============================================================

def save_residual_plot(epoch, models, split=False):
    """
    Evaluate and plot the pointwise PDE residual across [0,1].
    During the global stage, models = [global_model].
    During the decomposed stage, models = [model_left, model_right].
    """
    x_test = torch.linspace(0, 1, 1000).view(-1, 1).to(device)
    x_test.requires_grad_(True)

    if not split:
        model = models[0]
        res = pde_residual(model, x_test)
        res_np = res.detach().cpu().numpy().flatten()
    else:
        model_left, model_right = models
        # Split the test points at the interface
        mask_left  = (x_test.detach() <= a).squeeze()
        mask_right = (x_test.detach()  > a).squeeze()

        x_l = x_test[mask_left]
        x_r = x_test[mask_right]

        res_l = pde_residual(model_left,  x_l).detach().cpu().numpy().flatten()
        res_r = pde_residual(model_right, x_r).detach().cpu().numpy().flatten()

        res_np = np.concatenate([res_l, res_r])

    x_np = x_test.detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 4))
    plt.plot(x_np, res_np, color='crimson', linewidth=0.8)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    if split:
        plt.axvline(a, color='black', linestyle=':', label=f'Interface x={a}')
        plt.legend()
    plt.xlabel("x")
    plt.ylabel("PDE Residual")
    plt.title(f"PDE Residual — Epoch {epoch}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/residual_{epoch:05d}.png", dpi=150, bbox_inches='tight')
    plt.close()

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
# optimizer = torch.optim.Adam( global_model.parameters(), lr=1e-3)
optimizer = torch.optim.LBFGS( global_model.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

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
global_loss_history = []
window = 500 # we check avg loss over 'window' iterations; 500 here.
stagnation_tol = 1e-3
split_triggered = False

for epoch in range(epochs_stage1):
    # L-BFGS requires a closure that clears gradients, computes and returns the loss
    def closure():
        optimizer.zero_grad()
        loss = global_loss()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    loss_history.append(loss.item())
    global_loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(f"[GLOBAL] Epoch {epoch:5d} | Loss = {loss.item():.4e}")
        save_solution(epoch, [global_model], split=False)
        save_residual_plot(epoch, [global_model], split=False)

    # --------------------------------------------------------
    # STAGNATION CHECK
    # --------------------------------------------------------

    if epoch > 2*window:
        old_loss = np.mean(loss_history[-2*window:-window])#mean loss over previous window.
        new_loss = np.mean(loss_history[-window:])#mean loss over current window.
        relative_improvement = (old_loss - new_loss) / old_loss

        if (relative_improvement < stagnation_tol and new_loss > 1e-4):
            #latter cond. prevents unnecessary decomposition after sufficient convergence.
            print("\n================================================")
            print("STAGNATION DETECTED")
            print("TRIGGERING DOMAIN SPLIT")
            print("================================================\n")
            split_triggered = True
            break

# SPLIT DOMAIN
a = 0.5
x_left = torch.linspace(0,a,N_f//2).view(-1,1).to(device) # N_f//2 = [N_f/2], integer part.
x_right = torch.linspace(a,1,N_f//2).view(-1,1).to(device)
xa = torch.tensor([[a]], device=device)

# CREATE CHILD NETWORKS
model_left = PINN().to(device)
model_right = PINN().to(device)

# PARENT-TO-CHILD INHERITANCE. Use parent weights.
model_left.load_state_dict( copy.deepcopy(global_model.state_dict())) 
model_right.load_state_dict( copy.deepcopy(global_model.state_dict()))

# NEW OPTIMIZER
# optimizer = torch.optim.Adam(list(model_left.parameters()) + list(model_right.parameters()), lr=1e-4)
optimizer = torch.optim.LBFGS( list(model_left.parameters()) + list(model_right.parameters()), lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

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

    total_loss = loss_pde + loss_bc + 1000.0 * loss_interface + 100.0 * loss_flux

    return ( total_loss, loss_pde, loss_bc, loss_interface, loss_flux)

# ============================================================
# TRAIN DECOMPOSED SYSTEM
# ============================================================
decomposed_loss_history = []
epochs_stage2 = 100000
for epoch in range(epochs_stage2):
    # L-BFGS requires a closure
    def closure():
        optimizer.zero_grad()
        (loss, loss_pde, loss_bc, loss_interface, loss_flux) = decomposed_loss()
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    # Re-evaluate individual terms for logging (no grad needed)
    with torch.no_grad():
        (_, loss_pde, loss_bc, loss_interface, loss_flux) = decomposed_loss()

    decomposed_loss_history.append(loss.item())
    if epoch % 100 == 0:
        print(f"[DECOMPOSED] Epoch {epoch:5d} | " 
              f"Total={loss.item():.4e} | " 
              f"PDE={loss_pde.item():.4e} | " 
              f"BC={loss_bc.item():.4e} | "
              f"Interface={loss_interface.item():.4e} | " 
              f"Flux={loss_flux.item():.4e}")
        save_solution(epoch + epochs_stage1, [model_left, model_right], split=True)
        save_residual_plot(epoch + epochs_stage1, [model_left, model_right], split=True)

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

# ============================================================
# LOSS HISTORY PLOT
# ============================================================

plt.figure(figsize=(10,4))

# Global stage
plt.plot(np.arange(len(global_loss_history)), global_loss_history, label="Global PINN")
# Decomposed stage
plt.plot(np.arange(len(decomposed_loss_history)) + len(global_loss_history), decomposed_loss_history, label="Decomposed PINN")
# Split location
plt.axvline(len(global_loss_history), color='black', linestyle=':', label='Domain Split')
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid()
plt.savefig("loss_history.png", dpi=300, bbox_inches='tight')
plt.show()