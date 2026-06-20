# ============================================================
# VERSION 1 MODIFIED:
# COMPARISON OF 1, 2, AND 4 DOMAIN-DECOMPOSED PINNS
#
# PDE:
#     -u'' - k^2 u = f(x)
#
# Exact solution:
#     u(x) = sin(omega*pi*x)
#
# Cases:
#     Case 1: 1 neural network on [0,1]
#     Case 2: 2 neural networks on [0,0.5] and [0.5,1]
#     Case 3: 4 neural networks on four equal subdomains
#
# Output:
#     Version1_eval/Case_1/
#     Version1_eval/Case_2/
#     Version1_eval/Case_3/
#
# Important warning:
#     ReLU has zero second derivative almost everywhere.
#     Since this PINN uses the strong-form residual involving u'',
#     ReLU is not ideal for this problem.
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os
import shutil


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# USER INPUT
N_f = int(input("Enter number of collocation points per subdomain: "))

print("Which case do you want to run?")
print("  1 - Case 1")
print("  2 - Case 2")
print("  3 - Case 3")
print("  4 - All cases")
case_input = int(input("Enter choice (1/2/3/4): "))

if case_input == 4:
    cases_to_run = [1, 2, 3]
elif case_input in [1, 2, 3]:
    cases_to_run = [case_input]
else:
    raise ValueError("Case choice must be 1, 2, 3, or 4.")

# RESTART CHOICE
print("Restart from saved checkpoint if available?")
print("  1 - Yes")
print("  2 - No")
restart_input = int(input("Enter choice (1/2): "))

if restart_input == 1:
    restart_from_checkpoint = True
elif restart_input == 2:
    restart_from_checkpoint = False
else:
    raise ValueError("Restart choice must be 1 or 2.")

# TRAINING CONTROL
print("Training mode?")
print("  1 - Fixed epochs")
print("  2 - Train until loss tolerance")
training_input = int(input("Enter choice (1/2): "))

if training_input == 1:
    training_mode = "fixed"
    additional_epochs = int(input("Enter number of additional epochs to train: "))
    loss_tolerance = None
    max_epochs = None
elif training_input == 2:
    training_mode = "tol"
    loss_tolerance = float(input("Enter loss tolerance, e.g. 1e-4: "))
    max_epochs = int(input("Enter maximum allowed epochs, e.g. 50000: "))
    additional_epochs = None
else:
    raise ValueError("Training mode must be 1 or 2.")

# OUTPUT FOLDER
output_root = "Version1_eval"
os.makedirs(output_root, exist_ok=True)

# PROBLEM SETUP
omega = 5
helmholtz_k = 20


def exact_solution(x):
    return torch.sin(omega * np.pi * x)

def forcing(x):
    return ((omega * np.pi)**2 - helmholtz_k**2) * torch.sin(omega * np.pi * x)


# PINN MODEL
class PINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# DERIVATIVES
def model_derivatives(model, x):
    x_req = x.clone().detach().requires_grad_(True)
    u = model(x_req)
    u_x = torch.autograd.grad( u, x_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad( u_x, x_req, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u, u_x, u_xx

def pde_residual(model, x):
    u, _, u_xx = model_derivatives(model, x)
    return -u_xx - helmholtz_k**2 * u - forcing(x)


# ============================================================
# BUILD SUBDOMAINS
# ============================================================
def build_subdomains(num_subdomains, N_f):
    endpoints = np.linspace(0.0, 1.0, num_subdomains + 1)

    subdomain_intervals = []
    collocation_points = []
    eps = 1e-6

    for i in range(num_subdomains):
        xL = endpoints[i]
        xR = endpoints[i + 1]

        subdomain_intervals.append((xL, xR))

        # Interior collocation points only.
        # Boundary/interface points are handled separately.
        x_sub = torch.linspace( xL + eps, xR - eps, N_f ).view(-1, 1).to(device)

        collocation_points.append(x_sub)

    interface_points = []

    for i in range(1, num_subdomains):
        xi = torch.tensor([[endpoints[i]]], device=device, dtype=torch.float32)
        interface_points.append(xi)

    return subdomain_intervals, collocation_points, interface_points


# ============================================================
# LOSS FUNCTION
# ============================================================

def loss_function(models, collocation_points, interface_points):
    num_subdomains = len(models)

    # PDE loss
    loss_pde = 0.0

    for i in range(num_subdomains):
        loss_pde = loss_pde + torch.mean( pde_residual(models[i], collocation_points[i])**2)

    # Boundary condition loss
    # u(0)=0 and u(1)=0
    x0 = torch.tensor([[0.0]], device=device)
    x1 = torch.tensor([[1.0]], device=device)

    loss_bc = torch.mean(models[0](x0)**2) + torch.mean(models[-1](x1)**2)

    # Interface continuity loss
    # u_i(interface) = u_{i+1}(interface)
    loss_interface_u = 0.0
    loss_interface_flux = 0.0

    for i, xi in enumerate(interface_points):
        u_left = models[i](xi)
        u_right = models[i + 1](xi)

        loss_interface_u = loss_interface_u + torch.mean((u_left - u_right)**2)

        _, ux_left, _ = model_derivatives(models[i], xi)
        _, ux_right, _ = model_derivatives(models[i + 1], xi)

        loss_interface_flux = loss_interface_flux + torch.mean((ux_left - ux_right)**2)

    # Total loss
    loss = ( loss_pde + loss_bc + 100.0*loss_interface_u + 10.0*loss_interface_flux)

    return loss, loss_pde, loss_bc, loss_interface_u, loss_interface_flux


# EVALUATE PIECEWISE MODEL
def evaluate_piecewise(models, subdomain_intervals, x_test):
    u_pred = torch.zeros_like(x_test)

    with torch.no_grad():
        for i, (xL, xR) in enumerate(subdomain_intervals):
            if i == len(subdomain_intervals) - 1:
                mask = (x_test >= xL) & (x_test <= xR)
            else:
                mask = (x_test >= xL) & (x_test < xR)

            x_local = x_test[mask.squeeze()]

            if len(x_local) > 0:
                u_pred[mask.squeeze()] = models[i](x_local)

    return u_pred


# SAVE CURRENT SOLUTION
def save_current_solution( models, subdomain_intervals, epoch, case_folder):
    x_test = torch.linspace(0, 1, 1000).view(-1, 1).to(device)

    u_pred = evaluate_piecewise(models, subdomain_intervals, x_test)
    u_exact = exact_solution(x_test)

    plt.figure(figsize=(8, 4))
    plt.plot( x_test.cpu().numpy(), u_exact.detach().cpu().numpy(), label="Exact")
    plt.plot( x_test.cpu().numpy(), u_pred.detach().cpu().numpy(), "--", label="PINN Approximation")

    for xL, xR in subdomain_intervals[1:]:
        plt.axvline(xL, color="black", linestyle=":", linewidth=0.8)

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"{case_folder}/solution_epoch_{epoch:05d}.png", dpi=150, bbox_inches="tight")
    plt.close()


# SAVE RESIDUAL PLOT
def save_residual_plot( models, subdomain_intervals, epoch, case_folder):
    x_test = torch.linspace(0, 1, 1000).view(-1, 1).to(device)

    x_all = []
    residual_all = []

    for i, (xL, xR) in enumerate(subdomain_intervals):
        if i == len(subdomain_intervals) - 1:
            mask = (x_test >= xL) & (x_test <= xR)
        else:
            mask = (x_test >= xL) & (x_test < xR)

        x_local = x_test[mask.squeeze()]

        if len(x_local) > 0:
            res_local = pde_residual(models[i], x_local)

            x_all.append(x_local.detach().cpu().numpy().flatten())
            residual_all.append(res_local.detach().cpu().numpy().flatten())

    x_np = np.concatenate(x_all)
    res_np = np.concatenate(residual_all)

    plt.figure(figsize=(8, 4))
    plt.plot(x_np, res_np, linewidth=0.8)
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")

    for xL, xR in subdomain_intervals[1:]:
        plt.axvline(xL, color="black", linestyle=":", linewidth=0.8)

    plt.xlabel("x")
    plt.ylabel("PDE Residual")
    plt.title(f"PDE Residual — Epoch {epoch}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"{case_folder}/residual_epoch_{epoch:05d}.png", dpi=150, bbox_inches="tight")
    plt.close()


# SAVE LOSS PLOT
def save_loss_plot(loss_history, case_folder):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig( f"{case_folder}/loss_history.png", dpi=150, bbox_inches="tight")
    plt.close()

# CHECKPOINT SAVE / LOAD
def save_checkpoint(checkpoint_path, models, optimizer, epoch, histories):
    checkpoint = { "epoch": epoch, "model_state_dicts": [model.state_dict() for model in models],
                   "optimizer_state_dict": optimizer.state_dict(), "histories": histories}
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, models, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for model, state_dict in zip(models, checkpoint["model_state_dicts"]):
        model.load_state_dict(state_dict)

    start_epoch = checkpoint["epoch"] + 1
    histories = checkpoint["histories"]

    return start_epoch, histories

# TRAIN ONE CASE
def train_case(case_number, num_subdomains):
    print("\n" + "="*60)
    print(f"Running Case {case_number}: {num_subdomains} subdomain(s)")
    print("="*60)

    case_folder = f"{output_root}/Case_{case_number}"
    checkpoint_path = f"{case_folder}/checkpoint.pt"

    if os.path.exists(case_folder) and not restart_from_checkpoint:
        shutil.rmtree(case_folder)
    os.makedirs(case_folder, exist_ok=True)

    subdomain_intervals, collocation_points, interface_points = build_subdomains( num_subdomains, N_f)
    models = [PINN().to(device) for _ in range(num_subdomains)]

    parameters = []
    for model in models:
        parameters += list(model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    # optimizer = torch.optim.SGD(parameters, lr=1e-7, momentum=0.0)

    save_every = 100
    histories = { "total": [], "pde": [], "bc": [], "interface_u": [], "interface_flux": []}
    start_epoch = 0

    if restart_from_checkpoint and os.path.exists(checkpoint_path):
        start_epoch, histories = load_checkpoint(checkpoint_path, models, optimizer)
        print(f"Restarted Case {case_number} from epoch {start_epoch}")
    elif restart_from_checkpoint:
        print(f"No checkpoint found for Case {case_number}. Starting from scratch.")

    epoch = start_epoch

    if training_mode == "fixed":
        final_epoch = start_epoch + additional_epochs

    while True:
        optimizer.zero_grad()
        loss, loss_pde, loss_bc, loss_interface_u, loss_interface_flux = loss_function( models, collocation_points, interface_points)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()

        histories["total"].append(current_loss)
        histories["pde"].append(loss_pde.item())
        histories["bc"].append(loss_bc.item())

        if isinstance(loss_interface_u, float):
            histories["interface_u"].append(loss_interface_u)
        else:
            histories["interface_u"].append(loss_interface_u.item())

        if isinstance(loss_interface_flux, float):
            histories["interface_flux"].append(loss_interface_flux)
        else:
            histories["interface_flux"].append(loss_interface_flux.item())

        if epoch % save_every == 0:
            print(
                f"Case {case_number} | "
                f"Epoch {epoch:6d} | "
                f"Total = {current_loss:.4e} | "
                f"PDE = {loss_pde.item():.4e} | "
                f"BC = {loss_bc.item():.4e} | "
                f"Interface = {histories['interface_u'][-1]:.4e} | "
                f"Flux = {histories['interface_flux'][-1]:.4e}"
            )

            save_current_solution(models, subdomain_intervals, epoch, case_folder)
            save_residual_plot(models, subdomain_intervals, epoch, case_folder)
            save_checkpoint(checkpoint_path, models, optimizer, epoch, histories)

        if training_mode == "fixed":
            if epoch >= final_epoch:
                print(
                    f"Stopping Case {case_number}: "
                    f"completed {additional_epochs} additional epochs."
                )
                break

        elif training_mode == "tol":
            if current_loss <= loss_tolerance:
                print(
                    f"Stopping Case {case_number}: "
                    f"loss {current_loss:.4e} reached tolerance {loss_tolerance:.4e}."
                )
                break

            if epoch >= max_epochs:
                print(
                    f"Stopping Case {case_number}: "
                    f"reached max_epochs = {max_epochs} before tolerance."
                )
                break

        epoch += 1

    # Save final checkpoint
    save_checkpoint(checkpoint_path, models, optimizer, epoch, histories)

    # Save loss histories
    np.savetxt(
        f"{case_folder}/loss_history.txt",
        np.column_stack([
            np.array(histories["total"]),
            np.array(histories["pde"]),
            np.array(histories["bc"]),
            np.array(histories["interface_u"]),
            np.array(histories["interface_flux"])
        ]),
        header="total_loss pde_loss bc_loss interface_u_loss interface_flux_loss"
    )

    save_loss_plot(histories["total"], case_folder)

    # Final evaluation
    x_test = torch.linspace(0, 1, 1000).view(-1, 1).to(device)

    u_pred = evaluate_piecewise(models, subdomain_intervals, x_test)
    u_exact = exact_solution(x_test)

    error = torch.norm(u_exact - u_pred) / torch.norm(u_exact)

    print(f"\nCase {case_number} Relative L2 Error = {error.item():.6e}")

    with open(f"{case_folder}/relative_L2_error.txt", "w") as f:
        f.write(f"Relative L2 Error = {error.item():.12e}\n")

    save_current_solution(models, subdomain_intervals, epoch, case_folder)
    save_residual_plot(models, subdomain_intervals, epoch, case_folder)

    return error.item()


# RUN SELECTED CASES

case_to_subdomains = { 1: 1, 2: 2, 3: 4}
errors = {}

for case_number in cases_to_run:
    num_subdomains = case_to_subdomains[case_number]
    error = train_case(case_number, num_subdomains)
    errors[case_number] = error


# SUMMARY
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for case_number, error in errors.items():
    print(f"Case {case_number}: Relative L2 Error = {error:.6e}")

summary_file = f"{output_root}/summary.txt"

with open(summary_file, "w") as f:
    for case_number, error in errors.items():
        f.write(f"Case {case_number}: Relative L2 Error = {error:.12e}\n")

print(f"\nSaved all outputs in: {output_root}")