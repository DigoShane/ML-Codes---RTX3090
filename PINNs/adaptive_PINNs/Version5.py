# VERSION 5
# Manual checkpoint-driven enrichment.
#
# Fresh run:
#   Train only the global PINN (Stage 1). When stagnation is detected,
#   save a checkpoint and stop.
#
# Restart:
#   Load the previous global/local models, report all existing windows
#   and their current xL/xR values, and ask whether new windows should
#   be added. The user supplies the initial xL/xR of each new window.
#   All window locations remain trainable.
#
# Enriched training:
#   Train global + all local models using physical loss + overlap loss.
#   When stagnation is detected, save a checkpoint and stop. No window
#   is ever introduced automatically.


import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# USER OPTIONS
# ============================================================
batch_choice = input( "Enter minibatch size (or press Enter for full-batch): ").strip()
batch_size = int(batch_choice) if batch_choice else None

output_folder = "Version5_manual_enrichment"

restart_choice = input("Restart from checkpoint? Enter yes or no: ").lower()

if restart_choice in ["yes", "y"]:
    restart_from_checkpoint = True
elif restart_choice in ["no", "n"]:
    restart_from_checkpoint = False
else:
    raise ValueError("Restart choice must be yes or no.")


if restart_from_checkpoint:
    load_optimizer_choice = input( "Load optimizer state from checkpoint? Enter yes or no: ").lower()

    if load_optimizer_choice in ["yes", "y"]:
        load_optimizer_state = True
    elif load_optimizer_choice in ["no", "n"]:
        load_optimizer_state = False
    else:
        raise ValueError("Optimizer-state choice must be yes or no.")
else:
    load_optimizer_state = False


optimizer_type = input("Choose optimizer: adam, sgd, or gd: ").lower()

if optimizer_type not in ["adam", "sgd", "gd"]:
    raise ValueError("Optimizer must be adam, sgd, or gd.")

learning_rate = float(input("Enter learning rate, e.g. 1e-4: "))

if restart_from_checkpoint:
    epochs_stage1 = None
    additional_stage2_epochs = int(
        input("Enter maximum number of epochs for this restarted training run: ")
    )
else:
    epochs_stage1 = int(input("Enter maximum number of Stage 1 epochs: "))
    additional_stage2_epochs = 0

stagnation_window = int(input("Enter Stagnation check window size:"))

# ============================================================
# OUTPUT FOLDERS
# ============================================================
if os.path.exists(output_folder) and not restart_from_checkpoint:
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

solution_folder = os.path.join(output_folder, "solution_plots")
residual_folder = os.path.join(output_folder, "residual_plots")
window_folder = os.path.join(output_folder, "window_plots")

for folder in [solution_folder, residual_folder, window_folder]:
    os.makedirs(folder, exist_ok=True)

checkpoint_path = os.path.join(output_folder, "checkpoint.pt")


# ============================================================
# TEXT OUTPUT LOG
# ============================================================
# From this point onward, every existing print() in your code goes both to the terminal and to
log_file_path = os.path.join(output_folder, "training_output.txt")
log_mode = "a" if restart_from_checkpoint else "w"# Fresh run -> create a new file. Restart -> append to the existing file.
log_file = open(log_file_path, log_mode, buffering=1)


class Tee:
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


sys.stdout = Tee(sys.stdout, log_file)


print()
print("=" * 80)

if restart_from_checkpoint:
    print("RESTART FROM CHECKPOINT")
else:
    print("RUN FROM SCRATCH")

print("=" * 80)
print()


# ============================================================
# PROBLEM DEFINITION
# ============================================================
omega = 15
helmholtz_k = 20


def exact_solution(x):
    return torch.sin(omega * np.pi * x)


def forcing(x):
    return ( (omega * np.pi) ** 2 - helmholtz_k**2 ) * torch.sin(omega * np.pi * x)

print("--------------------------------------------------------------------")
print("                  CURRENTLY IN TEST MODE. TRAIN VALUES COMMENTED.   ")
print("--------------------------------------------------------------------")


# ============================================================
# TRAINING PARAMETERS
# ============================================================
N_f = int(input("Enter no. of Collocation points: "))
save_every = int(input("Save plots every: "))

# Compare the current loss with the loss stagnation_window epochs earlier.
stagnation_tol = float( input("Enter stagnation tolerance, e.g. 1e-3: ") )
stagnation_loss_threshold = 1e-4

# VERSION 5:
# beta remains fixed, while xL and xR remain trainable.
beta_init = 100.0

# This controls how strongly overlap between trainable windows is discouraged.
overlap_weight = float(
    input("Enter the weight associated with the window overlap function:")
)

# ============================================================
# COLLOCATION AND BOUNDARY POINTS
# ============================================================
x_global = torch.linspace(0.0, 1.0, N_f).view(-1, 1).to(device)
x0 = torch.tensor([[0.0]], device=device)
x1 = torch.tensor([[1.0]], device=device)


def iter_collocation_batches(x_pool, batch_size, shuffle=True):
    if batch_size is None or batch_size >= x_pool.shape[0]:
        yield x_pool
        return

    if shuffle:
        indices = torch.randperm(x_pool.shape[0], device=x_pool.device)
    else:
        indices = torch.arange(x_pool.shape[0], device=x_pool.device)

    for start in range(0, x_pool.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x_pool[batch_indices]


# ============================================================
# GLOBAL PINN
# ============================================================
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
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# LOCAL WINDOWED PINN
# ============================================================
class WindowedLocalPINN(nn.Module):
    """
    One trainable local correction:
        u_local_windowed(x) = W(x) * NN_local(xi)
    where
        W(x) = sigmoid(beta*(x - xL)) * sigmoid(beta*(xR - x))
    and
        xi = (x - center) / half_width.
    """

    def __init__(self, xL_init, xR_init, beta_init=100.0):
        super().__init__()

        xL_init = float(xL_init)
        xR_init = float(xR_init)

        if xR_init <= xL_init:
            raise ValueError("xR_init must be greater than xL_init.")

        xL_init = min(max(xL_init, 1e-4), 1.0 - 2e-4)
        xR_init = min(max(xR_init, xL_init + 1e-4), 1.0 - 1e-4)

        width_fraction_init = (xR_init - xL_init) / (1.0 - xL_init)
        width_fraction_init = min( max(width_fraction_init, 1e-4), 1.0 - 1e-4)

        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # VERSION 5:
        # A newly added correction starts at exactly zero. Therefore, adding a
        # new window does not instantaneously change the current solution.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.xL = nn.Parameter( torch.tensor([[xL_init]], dtype=torch.float32) )
        self.width_fraction = nn.Parameter( torch.tensor([[width_fraction_init]], dtype=torch.float32))

        #xDx-> CHANGED: beta is fixed, not trainable
        self.register_buffer( "beta", torch.tensor([[beta_init]], dtype=torch.float32))

    def window_parameters(self):
        xR = self.xL + (1.0 - self.xL) * self.width_fraction
        return self.xL, xR, self.beta

    def window(self, x):
        xL, xR, beta = self.window_parameters()
        left_switch = torch.sigmoid(beta * (x - xL))
        right_switch = torch.sigmoid(beta * (xR - x))
        raw_window = left_switch * right_switch
        half_width = 0.5 * (xR - xL)
        peak_value = torch.sigmoid( beta * half_width ) ** 2

        return raw_window / peak_value

    def local_coordinate(self, x):
        xL, xR, _ = self.window_parameters()
        center = 0.5 * (xL + xR)
        half_width = 0.5 * (xR - xL)
        return (x - center) / half_width

    def forward(self, x):
        w = self.window(x)
        xi = self.local_coordinate(x)
        correction = self.net(xi)
        return w * correction


# ============================================================
# WINDOW UTILITIES
# ============================================================
def project_window_parameters(local_model):
    """Keep one trainable window inside [0,1] with positive width."""

    with torch.no_grad():
        xL_min = 0.05 + 1e-4
        xR_max = 0.95 - 1.e-4
        minimum_width = 1e-3
        xL_max = xR_max - minimum_width
        local_model.xL.clamp_(min=xL_min, max=xL_max) # underscore used to rewrite original value.

        minimum_fraction = minimum_width / (1.0 - local_model.xL)
        maximum_fraction = (xR_max - local_model.xL) / (1.0 - local_model.xL)

        local_model.width_fraction.copy_( torch.maximum( local_model.width_fraction, minimum_fraction))
        local_model.width_fraction.copy_( torch.minimum( local_model.width_fraction, maximum_fraction))


def project_all_window_parameters(local_models):
    for local_model in local_models:
        project_window_parameters(local_model)


# VERSION 5:
# xL_init and xR_init are supplied by the user when a new window is added.
# They are only INITIAL values. xL and width_fraction remain nn.Parameters,
# so xL and xR continue to move during training.
def create_new_local_model(xL_init, xR_init):
    local_model = WindowedLocalPINN(
        xL_init=xL_init,
        xR_init=xR_init,
        beta_init=beta_init
    ).to(device)

    project_window_parameters(local_model)
    return local_model


# ============================================================
# OPTIMIZER
# ============================================================
def create_optimizer(parameter_groups, optimizer_type, learning_rate):
    if optimizer_type == "adam":
        return torch.optim.Adam(parameter_groups, lr=learning_rate)

    if optimizer_type == "sgd":
        return torch.optim.SGD( parameter_groups, lr=learning_rate, momentum=0.9)

    if optimizer_type == "gd":
        return torch.optim.SGD( parameter_groups, lr=learning_rate, momentum=0.0)

    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def make_optimizer(global_model, local_models):
    """
    VERSION 4 ADDITION:
    Use one optimizer parameter group for the global model and one parameter
    group for each local model. This lets a newly created window be appended
    without deleting the optimizer states of the existing models.
    """
    parameter_groups = [ {"params": list(global_model.parameters())} ]

    for local_model in local_models:
        parameter_groups.append( {"params": list(local_model.parameters())})

    return create_optimizer( parameter_groups, optimizer_type, learning_rate )


def add_local_model_to_optimizer(optimizer, local_model):
    optimizer.add_param_group( {"params": list(local_model.parameters())})


print()
print("Optimizer:", optimizer_type)
print("Learning rate:", learning_rate)
print("Window enrichment mode: manual, only on restart")
print()


# ============================================================
# DERIVATIVES AND RESIDUALS
# ============================================================
def derivatives_from_output(u, x_req):
    u_x = torch.autograd.grad( u, x_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad( u_x, x_req, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_x, u_xx


def global_residual(global_model, x):
    x_req = x.clone().detach().requires_grad_(True)
    u = global_model(x_req)
    _, u_xx = derivatives_from_output(u, x_req)
    residual = -u_xx - helmholtz_k**2 * u - forcing(x_req)
    return residual, x_req


def enriched_solution(x, global_model, local_models):
    u = global_model(x)
    for local_model in local_models:
        u = u + local_model(x)

    return u


def enriched_residual(x, global_model, local_models):
    x_req = x.clone().detach().requires_grad_(True)
    u = enriched_solution(x_req, global_model, local_models)
    _, u_xx = derivatives_from_output(u, x_req)
    residual = -u_xx - helmholtz_k**2 * u - forcing(x_req)
    return residual, x_req


# ============================================================
# LOSS FUNCTIONS
# ============================================================
def global_loss(global_model, x_batch):
    residual, _ = global_residual(global_model, x_batch)
    loss_pde = torch.mean(residual**2)
    loss_bc = ( torch.mean(global_model(x0) ** 2) + torch.mean(global_model(x1) ** 2))
    loss = loss_pde + loss_bc
    return loss, loss_pde, loss_bc


# overlap Loss
def window_overlap_loss(local_models):
    if len(local_models) <= 1:
        return torch.tensor(0.0, device=device)

    x_overlap = torch.linspace( 0.0, 1.0, 500, device=device).view(-1, 1)
    overlap_loss = torch.tensor(0.0, device=device)
    number_of_pairs = 0

    for i in range(len(local_models)):
        W_i = local_models[i].window(x_overlap)
        for j in range(i + 1, len(local_models)):
            W_j = local_models[j].window(x_overlap)
            overlap_loss = ( overlap_loss + torch.mean(W_i * W_j))
            number_of_pairs += 1

    overlap_loss = overlap_loss / number_of_pairs

    return overlap_loss


def enriched_loss(global_model, local_models, x_batch):
    residual, _ = enriched_residual( x_batch, global_model, local_models)
    loss_pde = torch.mean(residual**2)
    u0 = enriched_solution(x0, global_model, local_models)
    u1 = enriched_solution(x1, global_model, local_models)
    loss_bc = torch.mean(u0**2) + torch.mean(u1**2)
    loss_physical = loss_pde + loss_bc #xCHNGDx -> used to detect stagnation.
    loss_overlap = window_overlap_loss(local_models)
    loss_train = loss_physical + overlap_weight * loss_overlap #xCHNGDx -> rewritten for loss_physical.
    return ( loss_train, loss_physical, loss_pde, loss_bc, loss_overlap )



# ============================================================
# RESIDUAL INDICATOR
# ============================================================
def residual_indicator_from_residual(residual, x_req):
    residual_sq = residual.detach() ** 2
    eta = residual_sq
    eta = eta.cpu().numpy().flatten()
    eta = np.log(eta + 1e-12)
    return eta


# ============================================================
# PLOTTING
# ============================================================
def save_global_solution(epoch, global_model):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)

    global_model.eval()
    with torch.inference_mode():
        u_pred = global_model(x_test)
        u_exact = exact_solution(x_test)

    plt.figure(figsize=(8, 4))
    plt.plot( x_test.cpu().numpy(), u_exact.cpu().numpy(), label="Exact")
    plt.plot( x_test.cpu().numpy(), u_pred.cpu().numpy(), "--", label="Global PINN")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"Global solution — Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( os.path.join( solution_folder, f"global_solution_epoch_{epoch:06d}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_global_eta_plot(epoch, global_model):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)

    residual, x_req = global_residual(global_model, x_test)
    eta = residual_indicator_from_residual(residual, x_req)
    x_np = x_test.detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 4))
    plt.plot( x_np, eta, linewidth=0.8, label=r"$\eta=\log(r^2)$")
    plt.axhline( 0.0, color="black", linestyle="--", linewidth=0.6)
    plt.xlabel("x")
    plt.ylabel(r"$\eta(x)$")
    plt.title(f"Global residual indicator — Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( os.path.join( residual_folder, f"global_eta_epoch_{epoch:06d}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def get_window_parameter_values(local_models):
    values = []
    for local_model in local_models:
        xL, xR, beta = local_model.window_parameters()
        values.append( (xL.item(), xR.item(), beta.item()) )

    return values


def save_all_windows_plot(local_models, epoch):
    x_plot = torch.linspace(0.0, 1.0, 2000).view(-1, 1).to(device)
    local_models.eval()
    plt.figure(figsize=(10, 5))

    with torch.inference_mode():
        for index, local_model in enumerate(local_models):
            w_plot = local_model.window(x_plot)
            xL, xR, beta = local_model.window_parameters()
            plt.plot( x_plot.cpu().numpy(), w_plot.cpu().numpy(), label=(
                    f"Window {index + 1}: "
                    f"xL={xL.item():.3f}, "
                    f"xR={xR.item():.3f}, "
                    f"beta={beta.item():.1f}"))
            plt.axvline( xL.item(), linestyle=":", linewidth=0.7, alpha=0.6)
            plt.axvline( xR.item(), linestyle="--", linewidth=0.7, alpha=0.6)

    plt.xlabel("x")
    plt.ylabel(r"$W_i(x)$")
    plt.title( f"All trainable windows — Epoch {epoch} "
               f"({len(local_models)} windows)")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( os.path.join( window_folder, f"all_windows_epoch_{epoch:06d}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_enriched_solution( epoch, global_model, local_models, u_snapshot=None, x_snapshot=None):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)
    global_model.eval()
    local_models.eval()

    with torch.inference_mode():
        u_pred = enriched_solution( x_test, global_model, local_models)
        u_exact = exact_solution(x_test)
        window_values = get_window_parameter_values(local_models)

    plt.figure(figsize=(9, 4))
    plt.plot( x_test.cpu().numpy(), u_exact.cpu().numpy(), label="Exact")
    plt.plot( x_test.cpu().numpy(), u_pred.cpu().numpy(), "--", label="Enriched PINN")

    if u_snapshot is not None and x_snapshot is not None:
        plt.plot( x_snapshot, u_snapshot, linestyle=":", linewidth=0.8, alpha=0.8, label="Solution before latest window")

    for xL, xR, _ in window_values:
        plt.axvline( xL, color="black", linestyle=":", linewidth=0.5, alpha=0.4)
        plt.axvline( xR, color="black", linestyle="--", linewidth=0.5, alpha=0.4)

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(
        f"Enriched solution — Epoch {epoch} "
        f"({len(local_models)} windows)"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( os.path.join( solution_folder, f"enriched_solution_epoch_{epoch:06d}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_enriched_eta_plot( epoch, global_model, local_models):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)
    residual, x_req = enriched_residual( x_test, global_model, local_models)
    eta = residual_indicator_from_residual(residual, x_req)
    x_np = x_test.detach().cpu().numpy().flatten()
    local_models.eval()

    with torch.inference_mode():
        window_values = get_window_parameter_values(local_models)

    plt.figure(figsize=(9, 4))
    plt.plot( x_np, eta, linewidth=0.8, label=r"Current $\eta=\log(r^2)$")
    plt.axhline( 0.0, color="black", linestyle="--", linewidth=0.6)

    for xL, xR, _ in window_values:
        plt.axvline( xL, color="black", linestyle=":", linewidth=0.5, alpha=0.4,)
        plt.axvline( xR, color="black", linestyle="--", linewidth=0.5, alpha=0.4)

    plt.xlabel("x")
    plt.ylabel(r"$\eta(x)$")
    plt.title(
        f"Enriched residual indicator — Epoch {epoch} "
        f"({len(local_models)} windows)"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( os.path.join( residual_folder, f"enriched_eta_epoch_{epoch:06d}.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# STAGNATION CHECK
# ============================================================
def check_stagnation( loss_history, stagnation_window, stagnation_tol, loss_threshold):
    if len(loss_history) < stagnation_window + 1:
        return False, None

    old_loss = loss_history[-stagnation_window - 1]
    new_loss = loss_history[-1]

    if not np.isfinite(old_loss) or not np.isfinite(new_loss):
        return False, None

    change = abs(old_loss - new_loss)

    stagnated = ( change < stagnation_tol and new_loss > loss_threshold )

    return stagnated, change


# ============================================================
# SNAPSHOT BEFORE A NEW ENRICHMENT
# ============================================================
def take_solution_snapshot(global_model, local_models):
    x_snapshot_torch = ( torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device) )

    global_model.eval()
    local_models.eval()

    with torch.inference_mode():
        u_snapshot = enriched_solution( x_snapshot_torch, global_model, local_models).cpu().numpy().flatten()
    x_snapshot = x_snapshot_torch.cpu().numpy().flatten()
    return x_snapshot, u_snapshot


# ============================================================
# CHECKPOINTING
# ============================================================
def save_checkpoint( checkpoint_path, global_model, local_models, optimizer, stage1_end_epoch, loss_history_stage1,
                     loss_history_stage2, stagnation_history, window_creation_epochs, x_snapshot_sol, u_snapshot):
    checkpoint = { "global_model_state_dict": global_model.state_dict(), 
                   "local_model_state_dicts": [ model.state_dict() for model in local_models],
                   "optimizer_state_dict": optimizer.state_dict(), "checkpoint_optimizer_type": optimizer_type,
                    "checkpoint_learning_rate": learning_rate, "stage1_end_epoch": stage1_end_epoch, 
                    "loss_history_stage1": loss_history_stage1, "loss_history_stage2": loss_history_stage2,
                    "stagnation_history": list(stagnation_history), "window_creation_epochs": window_creation_epochs,
                    "x_snapshot_sol": ( None if x_snapshot_sol is None else x_snapshot_sol.tolist()),
                    "component_history": component_history, 
                    "u_snapshot": ( None if u_snapshot is None else u_snapshot.tolist()) }

    torch.save(checkpoint, checkpoint_path)


def create_local_model_from_state_dict(state_dict):
    xL_loaded = state_dict["xL"].item()
    width_fraction_loaded = state_dict["width_fraction"].item()
    xR_loaded = ( xL_loaded + (1.0 - xL_loaded) * width_fraction_loaded)
    local_model = WindowedLocalPINN( xL_init=xL_loaded, xR_init=xR_loaded, beta_init=beta_init).to(device)
    local_model.load_state_dict(state_dict)
    project_window_parameters(local_model)
    return local_model


def load_checkpoint_models(checkpoint_path, global_model):
    checkpoint = torch.load( checkpoint_path, map_location=device)
    global_model.load_state_dict( checkpoint["global_model_state_dict"])
    state_dicts = checkpoint.get( "local_model_state_dicts", None)
    component_history = checkpoint.get( "component_history", { "stage1_pde": [], "stage1_bc": [], "stage2_physical": [],
                                                               "stage2_pde": [], "stage2_bc": [], "stage2_overlap": [],
                                                               "stage2_weighted_overlap": [] } )

    # Limited fallback for loading Version 3 model weights. The Version 3
    # optimizer state will generally not match the new parameter-group layout.
    if state_dicts is None:
        old_state_dict = checkpoint.get( "local_model_state_dict", None)
        state_dicts = ( [] if old_state_dict is None else [old_state_dict])
    local_models = nn.ModuleList().to(device)

    for state_dict in state_dicts:
        local_models.append( create_local_model_from_state_dict(state_dict))

    stage1_end_epoch = checkpoint.get( "stage1_end_epoch", 0)
    loss_history_stage1 = checkpoint.get( "loss_history_stage1", [])
    loss_history_stage2 = checkpoint.get( "loss_history_stage2", [])
    stagnation_history = deque( checkpoint.get("stagnation_history", []), maxlen=stagnation_window + 1 )

    window_creation_epochs = checkpoint.get( "window_creation_epochs", [len(loss_history_stage1)] if len(local_models) > 0 else [])

    x_snapshot_sol = checkpoint.get("x_snapshot_sol", None)
    u_snapshot = checkpoint.get("u_snapshot", None)

    if x_snapshot_sol is not None:
        x_snapshot_sol = np.array(x_snapshot_sol)

    if u_snapshot is not None:
        u_snapshot = np.array(u_snapshot)

    return ( checkpoint, local_models, stage1_end_epoch, loss_history_stage1, loss_history_stage2, stagnation_history,
             component_history, window_creation_epochs, x_snapshot_sol, u_snapshot)


def maybe_load_optimizer_state( checkpoint, optimizer, load_optimizer_state):
    if not load_optimizer_state:
        print( "Loaded model weights only. "
            "Optimizer state was not loaded."
        )
        return

    checkpoint_optimizer_type = checkpoint.get( "checkpoint_optimizer_type", None)
    checkpoint_learning_rate = checkpoint.get( "checkpoint_learning_rate", None)

    same_optimizer = checkpoint_optimizer_type == optimizer_type
    same_lr = checkpoint_learning_rate == learning_rate

    if same_optimizer and same_lr:
        try:
            optimizer.load_state_dict( checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state from checkpoint.")
        except Exception as error:
            print("Could not load optimizer state.")
            print("Reason:", error)
            print("Continuing with fresh optimizer state.")
    else:
        print(
            "Optimizer state was not loaded because "
            "optimizer settings changed."
        )
        print("Checkpoint optimizer:", checkpoint_optimizer_type)
        print("Current optimizer:", optimizer_type)
        print(
            "Checkpoint learning rate:",
            checkpoint_learning_rate,
        )
        print("Current learning rate:", learning_rate)
        print("Continuing with fresh optimizer state.")


# ==============================================================
# PRINT INPUT VARIABLES AT START OF RUN
# ==============================================================
print()
print("=" * 60)
print("RUN SETTINGS")
print("=" * 60)
print("Device:", device)
print( "Minibatch size:", "full batch" if batch_size is None else batch_size )
print("Restart from checkpoint:", restart_from_checkpoint)
print("Load optimizer state:", load_optimizer_state)
print("Optimizer:", optimizer_type)
print("Learning rate:", learning_rate)

if not restart_from_checkpoint:
    print("Maximum Stage 1 epochs:", epochs_stage1)

print("Requested Stage 2 epochs:", additional_stage2_epochs)
print("Number of collocation points:", N_f)
print("Save every:", save_every)
print("Stagnation window:", stagnation_window)
print("Stagnation tolerance:", stagnation_tol)
print("Stagnation loss threshold:", stagnation_loss_threshold)
print("Window enrichment mode: manual, only on restart")
print("Fixed beta:", beta_init)
print("Overlap weight:", overlap_weight)

print("=" * 60)
print()


# ============================================================
# INITIALIZE MODELS AND HISTORIES
# ============================================================
global_model = PINN().to(device)
local_models = nn.ModuleList().to(device)
loss_history_stage1 = []
loss_history_stage2 = []
stagnation_history = deque( maxlen=stagnation_window + 1 )


# xCHNGDx: individual loss-component histories
component_history = { "stage1_pde": [], "stage1_bc": [], "stage2_physical": [], "stage2_pde": [], "stage2_bc": [],
                      "stage2_overlap": [], "stage2_weighted_overlap": [] }


# Epoch-axis positions where windows were introduced.
window_creation_epochs = []

stage1_end_epoch = 0
x_snapshot_sol = None
u_snapshot = None


# ============================================================
# RESTART LOGIC
# ============================================================
if restart_from_checkpoint:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

    (
        checkpoint,
        local_models,
        stage1_end_epoch,
        loss_history_stage1,
        loss_history_stage2,
        stagnation_history,
        component_history,
        window_creation_epochs,
        x_snapshot_sol,
        u_snapshot,
    ) = load_checkpoint_models(checkpoint_path, global_model)

    # First reconstruct the optimizer for exactly the models that existed
    # in the checkpoint. This allows an old optimizer state to be restored.
    optimizer = make_optimizer(global_model, local_models)
    maybe_load_optimizer_state(checkpoint, optimizer, load_optimizer_state)

    print()
    print("=" * 60)
    print("PREVIOUS CHECKPOINT STATE")
    print("=" * 60)

    if len(local_models) == 0:
        print("Previous model: GLOBAL PINN ONLY")
        print("Existing local windows: 0")
    else:
        print(
            f"Previous model: GLOBAL PINN + {len(local_models)} local window(s)"
        )

        for index, local_model in enumerate(local_models):
            xL, xR, beta = local_model.window_parameters()
            print(
                f"    Existing window {index + 1}: "
                f"xL={xL.item():.6f}, "
                f"xR={xR.item():.6f}, "
                f"beta={beta.item():.2f}"
            )

    print("=" * 60)
    print()

    add_windows_choice = input(
        "Do you want to add more local windows? Enter yes or no: "
    ).strip().lower()

    if add_windows_choice in ["yes", "y"]:
        number_of_new_windows = int(
            input("How many NEW local windows do you want to add? ")
        )

        if number_of_new_windows < 1:
            raise ValueError("Number of new windows must be at least 1.")

        # Store the solution immediately before this enrichment event.
        x_snapshot_sol, u_snapshot = take_solution_snapshot(
            global_model, local_models
        )

        creation_epoch = (
            len(loss_history_stage1)
            + len(loss_history_stage2)
        )

        print()
        print("Enter the INITIAL location of each new window.")
        print("These locations remain trainable during Stage 2.")
        print()

        for new_index in range(number_of_new_windows):
            window_number = len(local_models) + 1

            xL_init = float(
                input(
                    f"Enter initial xL for new window {window_number}: "
                )
            )
            xR_init = float(
                input(
                    f"Enter initial xR for new window {window_number}: "
                )
            )

            if not (0.0501 <= xL_init < xR_init <= 0.9499):
                raise ValueError(
                    "Each new window must satisfy "
                    "0.0501 <= xL < xR <= 0.9499."
                )

            new_local_model = create_new_local_model(
                xL_init=xL_init,
                xR_init=xR_init
            )

            local_models.append(new_local_model)
            add_local_model_to_optimizer(
                optimizer,
                new_local_model
            )

            window_creation_epochs.append(
                creation_epoch
            )

            xL, xR, beta = new_local_model.window_parameters()

            print(
                f"Created local window {window_number}: "
                f"xL={xL.item():.6f}, "
                f"xR={xR.item():.6f}, "
                f"beta={beta.item():.2f}"
            )

        print()
        print(
            "New local corrections were initialized to zero, "
            "so adding them did not instantaneously change the solution."
        )

    elif add_windows_choice in ["no", "n"]:
        number_of_new_windows = 0
        print()
        print("No new local windows were added.")

    else:
        raise ValueError("Choice must be yes or no.")

    # Every restarted training run gets a fresh stagnation interval.
    stagnation_history = deque(
        maxlen=stagnation_window + 1
    )

    if len(local_models) == 0:
        print()
        print(
            "The checkpoint contains only the global PINN and no new "
            "window was added."
        )
        print(
            "There is no enriched Stage 2 model to train. "
            "Exiting without changing the checkpoint."
        )
        print()

        sys.stdout = sys.stdout.terminal
        log_file.close()
        sys.exit(0)

    # Save immediately after manually adding windows, before long training.
    save_checkpoint(
        checkpoint_path,
        global_model,
        local_models,
        optimizer,
        stage1_end_epoch,
        loss_history_stage1,
        loss_history_stage2,
        stagnation_history,
        window_creation_epochs,
        x_snapshot_sol,
        u_snapshot
    )

    print()
    print("Restarted from checkpoint.")
    print("Continuing enriched training.")
    print("Optimizer:", optimizer_type)
    print("Learning rate:", learning_rate)
    print("Current local windows:", len(local_models))
    print("Existing Stage 2 epochs:", len(loss_history_stage2))
    print(
        "Fresh stagnation samples for this run:",
        len(stagnation_history)
    )
    print(
        "Maximum epochs requested for this run:",
        additional_stage2_epochs
    )
    print()

else:
    optimizer = make_optimizer(global_model, local_models)


# ============================================================
# STAGE 1: GLOBAL TRAINING
# ============================================================
if not restart_from_checkpoint:
    print()
    print("=" * 60)
    print("STAGE 1: GLOBAL TRAINING")
    print("=" * 60)
    print()

    stagnation_detected = False

    for epoch in range(epochs_stage1):
        global_model.train()
        epoch_losses = [] 
        epoch_pde_losses = []  # xCHNGDx -> to store pde loss vs epoch
        epoch_bc_losses = []   # xCHNGDx -> to store bc loss vs epoch.

        for x_batch in iter_collocation_batches( x_global, batch_size):
            loss, loss_pde, loss_bc = global_loss( global_model, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_pde_losses.append(loss_pde.item())  # xCHNGDx -> storing pde loss for this epoch.
            epoch_bc_losses.append(loss_bc.item())    # xCHNGDx -> storing bc loss for this epoch.

        mean_epoch_loss = float(np.mean(epoch_losses))
        loss_history_stage1.append(mean_epoch_loss)

        # xCHNGDx -> Storing data.
        component_history["stage1_pde"].append( float(np.mean(epoch_pde_losses)) )
        component_history["stage1_bc"].append( float(np.mean(epoch_bc_losses)) )

        if epoch % save_every == 0:
            global_model.eval()

            with torch.inference_mode():
                x_test = ( torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device))
                u_pred = global_model(x_test)
                u_exact = exact_solution(x_test)
                rel_error = ( torch.norm(u_exact - u_pred) / torch.norm(u_exact))

            print(
                f"[GLOBAL] "
                f"Epoch {epoch:6d} | "
                f"Mean minibatch loss = {mean_epoch_loss:.4e} | "
                f"RelL2 = {rel_error.item():.4e}"
            )

            save_global_solution(epoch, global_model)
            save_global_eta_plot(epoch, global_model)

        stage1_stagnated, stage1_change = check_stagnation( loss_history_stage1, stagnation_window, stagnation_tol, stagnation_loss_threshold)

        if stage1_stagnated:
            print()
            print("STAGNATION DETECTED DURING GLOBAL TRAINING")
            print( f"Stagnation absolute change = {stage1_change:.6e} "
                   f"| tolerance = {stagnation_tol:.6e}")
            print(f"Stopping Stage 1 at epoch {epoch}")
            print()

            stagnation_detected = True
            break

    stage1_end_epoch = len(loss_history_stage1) - 1

    if not stagnation_detected:
        print()
        print("Stage 1 reached its maximum epoch count before stagnation.")
        print()

    # VERSION 5:
    # A fresh run NEVER introduces a local model automatically.
    # Save the global-only state and stop. The next run can restart
    # from this checkpoint and manually add one or more windows.
    save_checkpoint(
        checkpoint_path,
        global_model,
        local_models,
        optimizer,
        stage1_end_epoch,
        loss_history_stage1,
        loss_history_stage2,
        stagnation_history,
        window_creation_epochs,
        x_snapshot_sol,
        u_snapshot
    )

    print("=" * 60)
    print("STAGE 1 STOPPED")
    print("=" * 60)

    if stagnation_detected:
        print("Reason: global physical loss stagnated.")
    else:
        print("Reason: maximum Stage 1 epoch count was reached.")

    print("Global-only checkpoint saved at:")
    print(checkpoint_path)
    print()
    print(
        "Run Version 5 again with restart=yes. "
        "The code will show the previous model and ask whether "
        "you want to add local windows."
    )
    print("=" * 60)

    sys.stdout = sys.stdout.terminal
    log_file.close()
    sys.exit(0)


# ============================================================
# STAGE 2: CONTINUOUSLY ENRICHED TRAINING
# ============================================================
print()
print("=" * 60)
print("STAGE 2: MANUALLY ENRICHED TRAINING")
print("=" * 60)
print()

if len(local_models) == 0:
    raise RuntimeError( "No local models are available for Stage 2.")

start_epoch_stage2 = len(loss_history_stage2)
final_epoch_stage2 = ( start_epoch_stage2 + additional_stage2_epochs)

for epoch in range(start_epoch_stage2, final_epoch_stage2):
    global_model.train()
    local_models.train()

    epoch_train_losses = []
    epoch_physical_losses = []
    epoch_overlap_losses = []
    epoch_pde_losses = []  # xCHNGDx -> Adding PDE loss per epoch.
    epoch_bc_losses = []   # xCHNGDx -> Adding bc loss per epoch.
    nan_detected = False
    epoch_weighted_overlap_losses = []  # xCHNGDx -> Adding overlap loss per epoch.

    for x_batch in iter_collocation_batches( x_global, batch_size):
        ( loss_train, loss_physical, loss_pde, loss_bc, loss_overlap ) = enriched_loss( global_model, local_models, x_batch)
        optimizer.zero_grad()
        if not torch.isfinite(loss_train):
            print( "NaN or infinity detected during enriched training.")
            nan_detected = True
            break
        loss_train.backward() #xCHNGDx
        optimizer.step()
        project_all_window_parameters(local_models)
        epoch_train_losses.append(loss_train.item()) #xCHNGDx
        epoch_physical_losses.append(loss_physical.item()) #xCHNGDx
        epoch_overlap_losses.append(loss_overlap.item()) #xCHNGDx
        epoch_pde_losses.append(loss_pde.item()) # xCHNGDx
        epoch_bc_losses.append(loss_bc.item()) # xCHNGDx
        epoch_weighted_overlap_losses.append( overlap_weight * loss_overlap.item() ) #xCHNGDx

    if nan_detected:
        break

    mean_train_loss = float(np.mean(epoch_train_losses))
    mean_physical_loss = float(np.mean(epoch_physical_losses))
    mean_pde_loss = float(np.mean(epoch_pde_losses))                  # xCHNGDx
    mean_bc_loss = float(np.mean(epoch_bc_losses))                    # xCHNGDx
    mean_overlap_loss = float(np.mean(epoch_overlap_losses))
    mean_weighted_overlap_loss = float( np.mean(epoch_weighted_overlap_losses) ) # xCHNGDx
    loss_history_stage2.append(mean_train_loss)
    stagnation_history.append(mean_physical_loss)

    stagnated, change = check_stagnation( stagnation_history, stagnation_window, stagnation_tol, stagnation_loss_threshold )

    # xCHNGDx
    component_history["stage2_physical"].append(mean_physical_loss)
    component_history["stage2_pde"].append(mean_pde_loss)
    component_history["stage2_bc"].append(mean_bc_loss)
    component_history["stage2_overlap"].append(mean_overlap_loss)
    component_history["stage2_weighted_overlap"].append( mean_weighted_overlap_loss)


    total_epoch = len(loss_history_stage1) + epoch

    if epoch % save_every == 0:
        global_model.eval()
        local_models.eval()

        with torch.inference_mode():
            x_test = ( torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device) )
            u_pred = enriched_solution( x_test, global_model, local_models)
            u_exact = exact_solution(x_test)
            rel_error = ( torch.norm(u_exact - u_pred) / torch.norm(u_exact) )
            window_values = get_window_parameter_values( local_models )

        print(
                f"[ENRICHED] "
                f"Total epoch = {total_epoch:6d} | "
                f"Stage2 epoch = {epoch:6d} | "
                f"Train loss = {mean_train_loss:.4e} | "
                f"Physical loss = {mean_physical_loss:.4e} | "
                f"Weighted Overlap loss = {mean_weighted_overlap_loss:.4e} | "
                f"RelL2 = {rel_error.item():.4e} | "
                f"Windows = {len(local_models)} | "
                f"Stagnation samples = {len(stagnation_history)}"
                )
        
        if change is not None:
            print(
                    f"Stagnation absolute change = {change:.6e} "
                    f"| tolerance = {stagnation_tol:.6e}"
                )
        else:
            print(
                    f"Stagnation absolute change = N/A "
                    f"| need {1 + stagnation_window} samples"
                 )

        for index, (xL, xR, beta) in enumerate(window_values):
            print(
                f"    Window {index + 1}: "
                f"xL={xL:.4f}, "
                f"xR={xR:.4f}, "
                f"beta={beta:.2f}"
            )

        save_enriched_solution( total_epoch, global_model, local_models, u_snapshot=u_snapshot, x_snapshot=x_snapshot_sol)
        save_enriched_eta_plot( total_epoch, global_model, local_models)
        save_all_windows_plot( local_models, total_epoch)

        save_checkpoint( checkpoint_path, global_model, local_models, optimizer, stage1_end_epoch, loss_history_stage1,
                         loss_history_stage2, stagnation_history, window_creation_epochs, x_snapshot_sol, u_snapshot)

    # ========================================================
    # VERSION 5: STAGNATION STOPS THE RUN.
    # NO WINDOW IS ADDED AUTOMATICALLY.
    # ========================================================
    if stagnated:
        print()
        print("=" * 60)
        print("STAGNATION DETECTED DURING ENRICHED TRAINING")
        print("=" * 60)
        print(
            f"Stagnation absolute change = {change:.6e} "
            f"| tolerance = {stagnation_tol:.6e}"
        )
        print()
        print("Current trained window locations:")

        for index, local_model in enumerate(local_models):
            xL, xR, beta = local_model.window_parameters()
            print(
                f"    Window {index + 1}: "
                f"xL={xL.item():.6f}, "
                f"xR={xR.item():.6f}, "
                f"beta={beta.item():.2f}"
            )

        print()
        print("Version 5 will NOT introduce another window automatically.")

        save_enriched_solution(
            total_epoch,
            global_model,
            local_models,
            u_snapshot=u_snapshot,
            x_snapshot=x_snapshot_sol
        )
        save_enriched_eta_plot(
            total_epoch,
            global_model,
            local_models
        )
        save_all_windows_plot(
            local_models,
            total_epoch
        )

        save_checkpoint(
            checkpoint_path,
            global_model,
            local_models,
            optimizer,
            stage1_end_epoch,
            loss_history_stage1,
            loss_history_stage2,
            stagnation_history,
            window_creation_epochs,
            x_snapshot_sol,
            u_snapshot
        )

        print()
        print("Checkpoint saved at:")
        print(checkpoint_path)
        print()
        print(
            "Run Version 5 again with restart=yes. "
            "The code will show these windows and ask whether "
            "you want to add more."
        )
        print("=" * 60)

        sys.stdout = sys.stdout.terminal
        log_file.close()
        sys.exit(0)


# ============================================================
# FINAL ERROR
# ============================================================
x_test = torch.linspace(0.0, 1.0, 2000).view(-1, 1).to(device)
global_model.eval()
local_models.eval()

with torch.inference_mode():
    u_pred = enriched_solution( x_test, global_model, local_models)
    u_exact = exact_solution(x_test)

relative_L2_error = ( torch.norm(u_exact - u_pred) / torch.norm(u_exact))

print()
print("FINAL RELATIVE L2 ERROR")
print(relative_L2_error.item())
print("FINAL NUMBER OF LOCAL WINDOWS")
print(len(local_models))

with open( os.path.join(output_folder, "final_error.txt"), "w") as file:
    file.write( "Final relative L2 error = "
                f"{relative_L2_error.item():.12e}\n")
    file.write( f"Final number of local windows = {len(local_models)}\n")


# ============================================================
# LOSS HISTORY
# ============================================================
n1 = len(loss_history_stage1)
n2 = len(loss_history_stage2)

plt.figure(figsize=(10, 4))

if n1 > 0:
    plt.plot( np.arange(n1), loss_history_stage1, label="Stage 1: Global")

if n2 > 0:
    plt.plot( np.arange(n1, n1 + n2), loss_history_stage2, label="Stage 2: Enriched")

for index, creation_epoch in enumerate(window_creation_epochs):
    plt.axvline( creation_epoch, color="black", linestyle=":", linewidth=0.8, alpha=0.7,
                 label=( "Local-window creation" if index == 0 else None ))

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss history with manual checkpoint-driven enrichment")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig( os.path.join(output_folder, "loss_history.png"), dpi=150, bbox_inches="tight")
plt.close()

# LOSS COMPONENT HISTORY
plt.figure(figsize=(11, 5))
eps_plot = 1e-30

# Stage 1
n_stage1_components = len(component_history["stage1_pde"])
if n_stage1_components > 0:
    x_stage1 = np.arange(n_stage1_components)
    plt.plot( x_stage1, np.maximum( np.array(loss_history_stage1[:n_stage1_components]), eps_plot ), 
              label="Stage 1 total", linewidth=1.5 )

    plt.plot( x_stage1, np.maximum( np.array(component_history["stage1_pde"]), eps_plot ), label="Stage 1 PDE" )
    plt.plot( x_stage1, np.maximum( np.array(component_history["stage1_bc"]), eps_plot ), label="Stage 1 BC" )

# Stage 2
n_stage2_components = len(component_history["stage2_pde"])
if n_stage2_components > 0:
    stage2_start = n1 + n2 - n_stage2_components
    x_stage2 = np.arange( stage2_start, stage2_start + n_stage2_components )
    plt.plot( x_stage2, np.maximum( np.array(loss_history_stage2[-n_stage2_components:]), eps_plot ),
        label="Stage 2 total", linewidth=1.5 )
    plt.plot( x_stage2, np.maximum( np.array(component_history["stage2_physical"]), eps_plot ), label="Stage 2 physical" )
    plt.plot( x_stage2, np.maximum( np.array(component_history["stage2_pde"]), eps_plot ), label="Stage 2 PDE" )
    plt.plot( x_stage2, np.maximum( np.array(component_history["stage2_bc"]), eps_plot ), label="Stage 2 BC" )
    plt.plot( x_stage2, np.maximum( np.array(component_history["stage2_weighted_overlap"]), eps_plot ), 
              label=r"Stage 2 weighted overlap")

# Mark creation of new windows
for creation_epoch in window_creation_epochs:
    plt.axvline( creation_epoch, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Individual loss components")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig( os.path.join(output_folder, "loss_components.png"), dpi=150, bbox_inches="tight" )
plt.close()


if n1 > 0:
    np.savetxt( os.path.join( output_folder, "loss_history_stage1.txt"), np.array(loss_history_stage1), 
                header="stage1_global_loss")

if n2 > 0:
    np.savetxt( os.path.join( output_folder, "loss_history_stage2.txt"), np.array(loss_history_stage2), 
                header="stage2_enriched_loss")

if len(window_creation_epochs) > 0:
    np.savetxt( os.path.join( output_folder, "window_creation_epochs.txt"), np.array(window_creation_epochs, dtype=int),
                fmt="%d", header="loss_history_epoch_at_which_each_window_was_added")


# ============================================================
# FINAL CHECKPOINT
# ============================================================
save_checkpoint( checkpoint_path, global_model, local_models, optimizer, stage1_end_epoch, loss_history_stage1, loss_history_stage2,
                 stagnation_history, window_creation_epochs, x_snapshot_sol, u_snapshot)

print()
print(f"Saved all results in: {output_folder}")


# ============================================================
# PRINT MODEL STATE DICTIONARIES
# ============================================================
print("=== Global model state_dict ===")
for key, tensor in global_model.state_dict().items():
    print(key, tensor.shape)

for model_index, local_model in enumerate(local_models):
    print( f"=== Local model {model_index + 1} state_dict ===")
    for key, tensor in local_model.state_dict().items():
        print(key, tensor.shape)


# ==============================================================
#    PRINT INPUT VARIABLES
# ==============================================================
print()
print("=" * 60)
print("RUN SETTINGS")
print("=" * 60)

print("Device:", device)

print(
    "Minibatch size:",
    "full batch" if batch_size is None else batch_size
)

print("Restart from checkpoint:", restart_from_checkpoint)
print("Load optimizer state:", load_optimizer_state)

print("Optimizer:", optimizer_type)
print("Learning rate:", learning_rate)

if not restart_from_checkpoint:
    print("Maximum Stage 1 epochs:", epochs_stage1)

print("Requested Stage 2 epochs:", additional_stage2_epochs)

print("Number of collocation points:", N_f)

print("Save every:", save_every)

print("Stagnation window:", stagnation_window)
print("Stagnation tolerance:", stagnation_tol)
print("Stagnation loss threshold:", stagnation_loss_threshold)

print("Window enrichment mode: manual, only on restart")

print("Fixed beta:", beta_init)
print("Overlap weight:", overlap_weight)


print()
print("ACTUAL RUN RESULTS")
print("Stage 1 epochs actually completed:", len(loss_history_stage1))
print("Stage 2 epochs actually completed:", len(loss_history_stage2))
print("Final number of local windows:", len(local_models))
print("Final relative L2 error:", relative_L2_error.item())

print("=" * 60)

print()
print("End of run.")
print("Output log saved to:", log_file_path)
print("=" * 80)

sys.stdout = sys.stdout.terminal
log_file.close()