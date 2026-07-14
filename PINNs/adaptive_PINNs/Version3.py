# ============================================================
# VERSION 3 PROTOTYPE
# RESIDUAL-DRIVEN ENRICHMENT
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

output_folder = "Version3_eval"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

# ============================================================
# PROBLEM
# ============================================================

omega = 15
helmholtz_k = 20

def exact_solution(x):
    return torch.sin(omega*np.pi*x)

def forcing(x):
    return ( (omega*np.pi)**2 - helmholtz_k**2)*torch.sin(omega*np.pi*x)

# ============================================================
# OPTIMIZER SETTINGS
# ============================================================

optimizer_type = "adam" 
#optimizer_type = "sgd" 
#optimizer_type = "gd"
learning_rate = 1e-4

# ============================================================
# OPTIMIZER FACTORY
# ============================================================

def create_optimizer(parameters):
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam( parameters, lr=learning_rate)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(  parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_type.lower() == "gd":
        return torch.optim.SGD( parameters, lr=learning_rate, momentum=0.0)
    else:
        raise ValueError( f"Unknown optimizer: {optimizer_type}")
 
print()
print("Optimizer:", optimizer_type)
print("Learning rate:", learning_rate)
print()

# ============================================================
# COLLOCATION POINTS
# ============================================================

N_f = 400
x_global = ( torch.linspace(0,1,N_f).view(-1,1).to(device))
x0 = torch.tensor([[0.0]], device=device)
x1 = torch.tensor([[1.0]], device=device)

# ============================================================
# NETWORK
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

    def forward(self,x):
        return self.net(x)

# ============================================================
# DERIVATIVES
# ============================================================

def first_derivative(model,x):

    x_req = (x.clone().detach().requires_grad_(True))
    u = model(x_req)
    ux = torch.autograd.grad( u, x_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    return ux

def second_derivative(model,x):

    x_req = ( x.clone().detach().requires_grad_(True)    )
    u = model(x_req)
    ux = torch.autograd.grad( u, x_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad( ux, x_req, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

    return uxx

# ============================================================
# GLOBAL RESIDUAL
# ============================================================

def global_residual(model,x):
    x_req = x.clone().detach().requires_grad_(True)
    u = model(x_req)
    ux = torch.autograd.grad( u, x_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad( ux, x_req, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

    return ( -uxx - helmholtz_k**2*u - forcing(x_req) )

# ============================================================
# WINDOW FUNCTION
# ============================================================

def window_function(x, xL, xR, delta=0.02):
    #return 0.5*( torch.tanh((x-xL)/delta) - torch.tanh((x-xR)/delta) )
    return ((x >= xL) & (x <= xR)).float() #characteristic function window.
                                #.float() is a pytorch method to case tensors to floating point nos.

def local_coordinate(x, xL, xR):
    center = 0.5*(xL+xR)
    half_width = 0.5*(xR-xL)
    return (x-center)/half_width

# ============================================================
# ENRICHED SOLUTION
# ============================================================

def enriched_solution( x, global_model, local_model, xL, xR):
    w = window_function( x, xL, xR)
    xi = local_coordinate( x, xL, xR)
    return ( global_model(x) + w*local_model(xi))

# ============================================================
# ENRICHED RESIDUAL
# ============================================================

def enriched_residual( x, global_model, local_model, xL, xR):

    x_req = ( x.clone().detach().requires_grad_(True))
    u = enriched_solution( x_req, global_model, local_model, xL, xR)
    ux = torch.autograd.grad( u, x_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x_req, grad_outputs=torch.ones_like(ux),create_graph=True)[0]

    return ( -uxx - helmholtz_k**2*u - forcing(x_req))

# ============================================================
# SAVE SOLUTION
# ============================================================

def save_solution( epoch, global_model, local_model=None, xL=None, xR=None, u_snapshot=None, x_snapshot_sol=None):

    x_test = ( torch.linspace(0,1,1000).view(-1,1).to(device))

    with torch.no_grad():
        if local_model is None:
            u_pred = global_model(x_test)
        else:
            u_pred = enriched_solution( x_test, global_model, local_model, xL, xR)

    u_exact = exact_solution(x_test)

    plt.figure(figsize=(8,4))
    plt.plot( x_test.cpu().numpy(), u_exact.cpu().numpy(), label="Exact")
    plt.plot( x_test.cpu().numpy(), u_pred.cpu().numpy(), '--', label="PINN")
    if u_snapshot is not None:
        plt.plot( x_snapshot_sol, u_snapshot, color='steelblue', linewidth=0.5,
                  linestyle='--', alpha=0.6, label="PINN at enrichment start")
    if xL is not None:# if window is active, then draw 2 vertical lines at xL and xR.
        plt.axvline(xL,color='k',linestyle=':')
        plt.axvline(xR,color='k',linestyle=':')
    plt.legend()
    plt.grid()
    plt.savefig( f"{output_folder}/epoch_{epoch:05d}.png", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# SAVE ETA PLOT
# ============================================================

def save_eta_plot(epoch, global_model, local_model=None, xL=None, xR=None, eta_snapshot=None, x_snapshot=None):
    """
    Plots eta(x) = log(r^2) across [0,1].
    Stage 1: uses global_residual on global_model.
    Stage 2: uses enriched_residual on global+local model.
    If eta_snapshot is provided, overlays it as a thin reference curve.
    """
    x_test = torch.linspace(0, 1, 1000).view(-1, 1).to(device)

    if local_model is None:
        r = global_residual(global_model, x_test)
    else:
        r = enriched_residual(x_test, global_model, local_model, xL, xR)

    eta = (r.detach().cpu().numpy().flatten())**2
    eta = np.log(eta + 1e-12)  # log(r^2), small epsilon avoids log(0)
    x_np = x_test.detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 4))
    plt.plot(x_np, eta, color='darkorange', linewidth=0.8, label=r'$\eta = \log(r^2)$')
    ##Displaying collocation points -START !! Comment out if you dont want to plot.
    #collocation_np = ( x_global.detach().cpu().numpy().flatten())
    #for xc in collocation_np:
    #    plt.axvline( xc, color='red', linewidth=0.2, alpha=0.4)
    ##Displaying collocation points -END
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    if xL is not None:
        plt.axvline(xL, color='k', linestyle=':', linewidth=0.8)
        plt.axvline(xR, color='k', linestyle=':', linewidth=0.8)
    if eta_snapshot is not None:
        plt.plot(x_snapshot, eta_snapshot, color='steelblue', linewidth=0.5,
                 linestyle='--', alpha=0.6, label=r'$\eta$ at enrichment start')
    plt.xlabel("x")
    plt.ylabel(r"$\eta(x) = \log(r^2)$")
    plt.title(f"$\\eta$ — Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/eta_{epoch:05d}.png", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# STAGE 1
# GLOBAL TRAINING
# ============================================================

global_model = PINN().to(device)
optimizer = create_optimizer( global_model.parameters()) # we defined create_optimizer in line #55.
loss_history = []
epochs_stage1 = 5000
window = 500
stagnation_tol = 1e-3

for epoch in range(epochs_stage1):
    #skiping model.tain()
    optimizer.zero_grad()
    r = global_residual( global_model, x_global)
    loss_pde = torch.mean(r**2)
    loss_bc = ( torch.mean(global_model(x0)**2) + torch.mean(global_model(x1)**2))
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(
            f"[GLOBAL] "
            f"{epoch:5d} "
            f"{loss.item():.4e}"
        )
        save_solution( epoch, global_model)
        save_eta_plot( epoch, global_model)

    if epoch > 2*window:
        old_loss = np.mean( loss_history[-2*window:-window])
        new_loss = np.mean( loss_history[-window:])
        improvement = (old_loss-new_loss)/old_loss
        if (improvement < stagnation_tol and new_loss > 1e-4):
            print("STAGNATION DETECTED")
            break

# ============================================================
# FIND HOTSPOT
# ============================================================

x_residual = ( torch.linspace(0,1,4000).view(-1,1).to(device))

eta = np.log((global_residual(global_model, x_residual).detach().cpu().numpy().flatten())**2 + 1e-12)

# Store snapshot of eta used for hotspot detection — overlaid on Stage 2 plots
eta_snapshot    = eta.copy()
x_snapshot      = x_residual.detach().cpu().numpy().flatten()

# Store snapshot of the solution at enrichment start — overlaid on Stage 2 solution plots
x_snapshot_sol = torch.linspace(0, 1, 1000).view(-1, 1).to(device)
with torch.no_grad():
    u_snapshot = global_model(x_snapshot_sol).cpu().numpy().flatten()
x_snapshot_sol = x_snapshot_sol.cpu().numpy().flatten()

# ============================================================
# HOTSPOT DETECTION
# ============================================================

tau = 0.5
mask = ( eta > tau*np.max(eta))

# longest run of True
best_start = 0
best_end = 0
best_len = 0

start = None

for i,val in enumerate(mask):
    if val and start is None:
        start = i
    if ( (not val or i==len(mask)-1) and start is not None):
        end = i
        if end-start > best_len:
            best_len = end-start
            best_start = start
            best_end = end
        start = None

xL = x_residual[best_start].item()
xR = x_residual[best_end].item()

print ("Enrichment pre-min width")
print("xL =",xL)
print("xR =",xR)

#enforcing minwidth 2% of domain length
min_width = 0.10
if (xR - xL) < min_width:
    center = 0.5*(xL + xR)
    xL = max( 0.0, center - min_width/2)
    xR = min(1.0, center + min_width/2)

print("ENRICHMENT REGION:")
print("xL =",xL)
print("xR =",xR)
print("WIDTH =",xR-xL)

x_plot = torch.linspace( 0, 1, 2000).view(-1,1).to(device)
w_plot = ( window_function( x_plot, xL, xR).cpu().numpy())
plt.figure(figsize=(8,4))
plt.plot( x_plot.cpu().numpy(), w_plot)
plt.grid()
plt.title("Characteristic Window")
plt.show()

# ============================================================
# LOCAL CORRECTION NETWORK
# ============================================================

local_model = PINN().to(device)
optimizer = create_optimizer( list(global_model.parameters()) + list(local_model.parameters()))

# ============================================================
# STAGE 2
# ENRICHED TRAINING
# ============================================================

epochs_stage2 = 10000
loss_history_stage2 = []

for epoch in range(epochs_stage2):

    optimizer.zero_grad()
    r = enriched_residual( x_global, global_model, local_model, xL, xR)
    loss_pde = torch.mean(r**2)
    u0 = enriched_solution( x0, global_model, local_model, xL, xR)
    u1 = enriched_solution( x1, global_model, local_model, xL, xR)
    loss_bc = ( torch.mean(u0**2) + torch.mean(u1**2))
    loss = loss_pde + loss_bc
    if torch.isnan(loss):
        print("NaN DETECTED")
        break
    loss.backward()
    optimizer.step()
    loss_history_stage2.append(loss.item())

    if epoch % 100 == 0:
        print(
            f"[ENRICHED] "
            f"{epoch:5d} "
            f"{loss.item():.4e}"
        )
        save_solution( epoch+epochs_stage1, global_model, local_model, xL, xR,
                       u_snapshot=u_snapshot, x_snapshot_sol=x_snapshot_sol)
        save_eta_plot( epoch+epochs_stage1, global_model, local_model, xL, xR,
                       eta_snapshot=eta_snapshot, x_snapshot=x_snapshot)

# ============================================================
# FINAL ERROR
# ============================================================

x_test = ( torch.linspace(0,1,2000).view(-1,1).to(device))

with torch.no_grad():
    u_pred = enriched_solution( x_test, global_model, local_model, xL, xR)

u_exact = exact_solution(x_test)

error = ( torch.norm(u_exact-u_pred)/torch.norm(u_exact))

print()
print("FINAL RELATIVE L2 ERROR")
print(error.item())

# ============================================================
# LOSS HISTORY PLOT
# ============================================================

n1 = len(loss_history)
n2 = len(loss_history_stage2)

plt.figure(figsize=(10, 4))
plt.plot(np.arange(n1), loss_history, label="Stage 1 (Global)")
plt.plot(np.arange(n1, n1 + n2), loss_history_stage2, label="Stage 2 (Enriched)")
plt.axvline(n1, color='black', linestyle=':', label='Enrichment Start')
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}/loss_history.png", dpi=150, bbox_inches='tight')
plt.show()