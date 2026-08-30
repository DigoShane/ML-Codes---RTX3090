#Version 4 uses repeated adaptive enrichment. 
# The solution is represented as a global neural network plus a sum of local windowed corrections. Training begins with only the global model. 
# A stagnation check is applied to the recent loss history. When the loss stagnates, a new trainable windowed local model is introduced. 
# The new local model has its own trainable xL, width_fraction, and beta, so its spatial support is learned by minimizing the total PINN loss, 
# not by an explicit residual hotspot detector. After adding a new local model, the optimizer is rebuilt so that it includes the global model and 
# all previously introduced local models. The stagnation history is then reset so that another window is not introduced immediately because of 
# the old pre-enrichment loss history.

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

output_folder = "Version4_continuous_windows"

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
    additional_stage2_epochs = int( input("Enter number of additional Stage 2 epochs: "))
else:
    epochs_stage1 = int(input("Enter maximum number of Stage 1 epochs: "))
    additional_stage2_epochs = int(input("Enter number of Stage 2 epochs: "))


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
# PROBLEM DEFINITION
# ============================================================
omega = 15
helmholtz_k = 20


def exact_solution(x):
    return torch.sin(omega * np.pi * x)


def forcing(x):
    return ( (omega * np.pi) ** 2 - helmholtz_k**2 ) * torch.sin(omega * np.pi * x)


# ============================================================
# TRAINING PARAMETERS
# ============================================================
N_f = int(input("Enter no. of Collocation points: "))
save_every = 100

# The stagnation test compares two consecutive blocks of this length.
stagnation_window = 1000
stagnation_tol = 1e-3
stagnation_loss_threshold = 1e-4

# VERSION 4 ADDITION:
maximum_number_of_windows = 10

beta_init = 100.0

# Each new window starts from this broad trainable initial guess.
initial_window_left_fraction = 0.10
initial_window_right_fraction = 0.90

#This is used to control how strongly overlap is discouraged.
overlap_weight = 10.0

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

        # VERSION 4 ADDITION:
        # A newly added correction starts at exactly zero. Therefore, adding a
        # new window does not instantaneously change the current solution.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.xL = nn.Parameter( torch.tensor([[xL_init]], dtype=torch.float32) )
        self.width_fraction = nn.Parameter( torch.tensor([[width_fraction_init]], dtype=torch.float32))

        if beta_init > 20.0:
            raw_beta_init = beta_init
        else:
            raw_beta_init = np.log(np.expm1(beta_init))

        self.raw_beta = nn.Parameter( torch.tensor([[raw_beta_init]], dtype=torch.float32))

    def window_parameters(self):
        beta = F.softplus(self.raw_beta)
        xR = self.xL + (1.0 - self.xL) * self.width_fraction
        return self.xL, xR, beta

    def window(self, x):
        xL, xR, beta = self.window_parameters()
        left_switch = torch.sigmoid(beta * (x - xL))
        right_switch = torch.sigmoid(beta * (xR - x))
        return left_switch * right_switch

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
        minimum_width = 1e-3

        local_model.xL.clamp_(0.0, 1.0 - minimum_width) # underscore used to rewrite original value.

        # Enforce xR - xL >= minimum_width.
        remaining_width = torch.clamp( 1.0 - local_model.xL, min=minimum_width)
        minimum_fraction = minimum_width / remaining_width

        local_model.width_fraction.copy_( torch.maximum( local_model.width_fraction, minimum_fraction))
        local_model.width_fraction.clamp_(max=1.0 - 1e-4)


def project_all_window_parameters(local_models):
    for local_model in local_models:
        project_window_parameters(local_model)


def initial_local_window():
    xL_init = initial_window_left_fraction
    xR_init = initial_window_right_fraction

    if xR_init <= xL_init:
        raise ValueError( "Initial local window must satisfy xR_init > xL_init.")

    return xL_init, xR_init


# VERSION 4 ADDITION:
def create_new_local_model():
    xL_init, xR_init = initial_local_window()

    local_model = WindowedLocalPINN( xL_init=xL_init, xR_init=xR_init, beta_init=beta_init).to(device)

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
print("Maximum number of local windows:", maximum_number_of_windows)
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

    for i in range(len(local_models)):
        W_i = local_models[i].window(x_overlap)
        for j in range(i + 1, len(local_models)):
            W_j = local_models[j].window(x_overlap)
            overlap_loss = ( overlap_loss + torch.mean(W_i * W_j))

    return overlap_loss


def enriched_loss(global_model, local_models, x_batch):
    residual, _ = enriched_residual( x_batch, global_model, local_models)
    loss_pde = torch.mean(residual**2)
    u0 = enriched_solution(x0, global_model, local_models)
    u1 = enriched_solution(x1, global_model, local_models)
    loss_bc = torch.mean(u0**2) + torch.mean(u1**2)
    loss_overlap = window_overlap_loss(local_models)
    loss = loss_pde + loss_bc + overlap_weight * loss_overlap
    return loss, loss_pde, loss_bc


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
    if len(loss_history) < 2 * stagnation_window:
        return False

    old_loss = np.mean( loss_history[-2 * stagnation_window : -stagnation_window])
    new_loss = np.mean( loss_history[-stagnation_window:])

    if not np.isfinite(old_loss) or not np.isfinite(new_loss):
        return False

    if old_loss <= 0.0:
        return False

    improvement = (old_loss - new_loss) / old_loss

    return ( improvement < stagnation_tol and new_loss > loss_threshold )


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
                    "stagnation_history": stagnation_history, "window_creation_epochs": window_creation_epochs,
                    "x_snapshot_sol": ( None if x_snapshot_sol is None else x_snapshot_sol.tolist()),
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
    stagnation_history = checkpoint.get( "stagnation_history", [])

    window_creation_epochs = checkpoint.get( "window_creation_epochs", [len(loss_history_stage1)] if len(local_models) > 0 else [])

    x_snapshot_sol = checkpoint.get("x_snapshot_sol", None)
    u_snapshot = checkpoint.get("u_snapshot", None)

    if x_snapshot_sol is not None:
        x_snapshot_sol = np.array(x_snapshot_sol)

    if u_snapshot is not None:
        u_snapshot = np.array(u_snapshot)

    return ( checkpoint, local_models, stage1_end_epoch, loss_history_stage1, loss_history_stage2, stagnation_history,
             window_creation_epochs, x_snapshot_sol, u_snapshot)


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


# ============================================================
# INITIALIZE MODELS AND HISTORIES
# ============================================================
global_model = PINN().to(device)
local_models = nn.ModuleList().to(device)
loss_history_stage1 = []
loss_history_stage2 = []
stagnation_history = []

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
        raise FileNotFoundError( f"No checkpoint found at: {checkpoint_path}")

    ( checkpoint, local_models, stage1_end_epoch, loss_history_stage1, loss_history_stage2, stagnation_history,
        window_creation_epochs, x_snapshot_sol, u_snapshot) = load_checkpoint_models( checkpoint_path, global_model)

    if len(local_models) == 0:
        raise RuntimeError(
            "Checkpoint does not contain any local model. "
            "Restart is supported only after enrichment begins."
        )

    optimizer = make_optimizer(global_model, local_models)
    maybe_load_optimizer_state( checkpoint, optimizer, load_optimizer_state)

    print()
    print("Restarted from checkpoint.")
    print("Continuing enriched training.")
    print("Optimizer:", optimizer_type)
    print("Learning rate:", learning_rate)
    print("Existing local windows:", len(local_models))
    print("Existing Stage 2 epochs:", len(loss_history_stage2))
    print( "Losses collected toward next stagnation check:", len(stagnation_history) )
    print( "Additional Stage 2 epochs requested:", additional_stage2_epochs)
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

        for x_batch in iter_collocation_batches( x_global, batch_size):
            loss, loss_pde, loss_bc = global_loss( global_model, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_epoch_loss = float(np.mean(epoch_losses))
        loss_history_stage1.append(mean_epoch_loss)

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

        if check_stagnation( loss_history_stage1, stagnation_window, stagnation_tol, stagnation_loss_threshold):
            print()
            print("STAGNATION DETECTED DURING GLOBAL TRAINING")
            print(f"Stopping Stage 1 at epoch {epoch}")
            print()

            stagnation_detected = True
            break

    stage1_end_epoch = len(loss_history_stage1) - 1

    if not stagnation_detected:
        print()
        print("Stage 1 reached its maximum epoch count.")
        print("Introducing the first local model anyway.")
        print()


# ============================================================
# CREATE FIRST LOCAL WINDOW
# ============================================================
if not restart_from_checkpoint:
    if maximum_number_of_windows < 1:
        raise ValueError( "maximum_number_of_windows must be at least 1.")

    print()
    print("=" * 60)
    print("CREATING LOCAL WINDOW 1")
    print("=" * 60)
    print()

    # Snapshot of the global solution immediately before enrichment.
    x_snapshot_sol, u_snapshot = take_solution_snapshot( global_model, local_models)
    new_local_model = create_new_local_model()
    local_models.append(new_local_model)
    add_local_model_to_optimizer( optimizer, new_local_model)
    first_creation_epoch = len(loss_history_stage1)
    window_creation_epochs.append(first_creation_epoch)

    # VERSION 4 CHANGE:
    stagnation_history = []

    xL, xR, beta = new_local_model.window_parameters()
    print("Created local window 1")
    print("Initial xL =", xL.item())
    print("Initial xR =", xR.item())
    print("Initial beta =", beta.item())
    print()

    save_all_windows_plot( local_models, first_creation_epoch)
    save_checkpoint( checkpoint_path, global_model, local_models, optimizer, stage1_end_epoch, loss_history_stage1, 
                     loss_history_stage2, stagnation_history, window_creation_epochs, x_snapshot_sol, u_snapshot)


# ============================================================
# STAGE 2: CONTINUOUSLY ENRICHED TRAINING
# ============================================================
print()
print("=" * 60)
print("STAGE 2: CONTINUOUSLY ENRICHED TRAINING")
print("=" * 60)
print()

if len(local_models) == 0:
    raise RuntimeError( "No local models are available for Stage 2.")

start_epoch_stage2 = len(loss_history_stage2)
final_epoch_stage2 = ( start_epoch_stage2 + additional_stage2_epochs)

for epoch in range(start_epoch_stage2, final_epoch_stage2):
    global_model.train()
    local_models.train()

    epoch_losses = []
    nan_detected = False

    for x_batch in iter_collocation_batches( x_global, batch_size):
        loss, loss_pde, loss_bc = enriched_loss( global_model, local_models, x_batch)
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            print( "NaN or infinity detected during enriched training.")
            nan_detected = True
            break
        loss.backward()
        optimizer.step()
        project_all_window_parameters(local_models)
        epoch_losses.append(loss.item())

    if nan_detected:
        break

    mean_epoch_loss = float(np.mean(epoch_losses))
    loss_history_stage2.append(mean_epoch_loss)
    stagnation_history.append(mean_epoch_loss)
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
            f"Mean minibatch loss = {mean_epoch_loss:.4e} | "
            f"RelL2 = {rel_error.item():.4e} | "
            f"Windows = {len(local_models)} | "
            f"Stagnation samples = {len(stagnation_history)}"
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
    # VERSION 4 CHANGE: REPEATED STAGNATION-DRIVEN ENRICHMENT
    # ========================================================
    if ( len(local_models) < maximum_number_of_windows and check_stagnation( stagnation_history, stagnation_window, stagnation_tol,
            stagnation_loss_threshold ) ):
        print()
        print("STAGNATION DETECTED DURING ENRICHED TRAINING")
        print( "Introducing local window", len(local_models) + 1)
        print()

        x_snapshot_sol, u_snapshot = take_solution_snapshot( global_model, local_models)
        new_local_model = create_new_local_model()
        local_models.append(new_local_model)
        add_local_model_to_optimizer( optimizer, new_local_model)

        creation_epoch = ( len(loss_history_stage1) + len(loss_history_stage2))
        window_creation_epochs.append(creation_epoch)

        stagnation_history = []
        xL, xR, beta = new_local_model.window_parameters()
        print(
            f"Created local window {len(local_models)} at "
            f"loss-history epoch {creation_epoch}."
        )
        print(
            f"Initial parameters: "
            f"xL={xL.item():.4f}, "
            f"xR={xR.item():.4f}, "
            f"beta={beta.item():.2f}"
        )
        print(
            "The new local correction was initialized to zero, "
            "so the current solution was preserved."
        )
        print()

        save_enriched_solution( creation_epoch, global_model, local_models, u_snapshot=u_snapshot, x_snapshot=x_snapshot_sol)
        save_enriched_eta_plot( creation_epoch, global_model, local_models)
        save_all_windows_plot( local_models, creation_epoch)

        save_checkpoint( checkpoint_path, global_model, local_models, optimizer, stage1_end_epoch, loss_history_stage1,
                         loss_history_stage2, stagnation_history, window_creation_epochs, x_snapshot_sol, u_snapshot)


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
plt.title("Loss history with repeated local enrichment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig( os.path.join(output_folder, "loss_history.png"), dpi=150, bbox_inches="tight")
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