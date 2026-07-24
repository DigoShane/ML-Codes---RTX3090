################################################################################################
# The objective of Version 2 is the following:
# 1. It sets up a PINN problem for a 1D Helmholtz PDE.
#    The code is trying to solve a boundary-value problem of the form
#    −u′′−k^2 u=f(x) 
#    on the domain [0,1], using PINNs. The exact solution is chosen in advance as 
#    u(x)=sin(ωπx), and the forcing function is constructed so that this exact solution satisfies the PDE.
#
# 2. It creates a global neural network to approximate the solution on the whole domain.
#    The first model, called global_model, is a fully connected neural network that takes x as input and outputs 
#    an approximation to u(x). This model is trained over the entire interval [0,1].
#
# 3. It repeatedly saves plots during global training.
#    Every 100 epochs, the code saves a plot of the current PINN solution against the exact solution. 
#    It also saves a residual-indicator plot showing where the PDE residual is large.
#
# 4. It monitors whether the global training has stagnated.
#    The code tracks the loss history and compares the average loss over two recent windows of training. 
#    If the newer average loss is not sufficiently better than the older average loss, or the newer loss doesnt change much, 
#    the code declares stagnation.
#
#    It evaluates the residual on a finer grid after global training.
#    After Stage 1, the code evaluates the global model’s PDE residual on a much denser set of points than the original 
#    training grid. This finer residual field is used to identify where the global PINN is failing most.
#    It constructs a residual indicator.
#    The code defines a quantity
#    η(x)=log(r(x)^2),
#    where r(x) is the PDE residual. This residual indicator makes it easier to locate regions where the residual is large.
# 5. It detects a residual hotspot using a thresholding logic.
#    The code finds points where the residual indicator is larger than a fixed fraction (f=0.5 here) of its maximum value. 
#
# 6. It identifies the largest connected high-residual region.
#    After thresholding, the code searches through the Boolean mask and finds the longest continuous interval 
#   where the residual indicator remains high. This interval is treated as the main hotspot.
#
# 7. It defines an enrichment region around the hotspot.
#    The detected hotspot becomes the interval [xL, xR]. If this interval is too narrow, the code expands it to 
#    satisfy a minimum width requirement.
#
# 8. It introduces a local correction neural network.
#    After the hotspot is found, the code creates a second neural network called local_model. 
#    This network is intended to correct the global model only in the problematic region.
#
# 9. It defines a window function to localize the correction.
#    The local network is multiplied by a window function. The window function is part of the NN, so it should be part of the
#    training. 
#
# 10. It forms an enriched solution.
#     The enriched approximation is the global solution plus a localized correction:
#     global PINN output plus windowed local PINN output. Outside the enrichment region, the solution is just the global model. 
#     Inside the enrichment region, the local model adds a correction. 
# 
# 11. It trains the enriched model in a second stage.
#     In Stage 2, the code trains the combined global-plus-local model using the same type of PINN loss: 
#     PDE residual loss plus boundary-condition loss.
#
# 12. It updates both the global model and local model during enrichment.
#     The Stage 2 optimizer is given the parameters of both the global model and the local model. 
#     So the local correction is not the only part being trained; the original global model also continues changing.
#     It saves plots during enriched training.
#  
# 13. After Stage 2, the code compares the final enriched solution against the known exact solution and computes the relative 
#     L2 -type error.
#     At the end, the code plots the loss history from Stage 1 and Stage 2 on the same figure, with a vertical 
#     line marking when enrichment started.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import os
import shutil

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# User options
batch_choice = input("Enter minibatch size (or press Enter for full-batch): ").strip()
batch_size = int(batch_choice) if batch_choice else None

output_folder = "Version3-Part2_eval"

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
    epochs_stage2 = int(input("Enter number of additional Stage 2 epochs: "))
else:
    epochs_stage1 = int(input("Enter maximum number of Stage 1 epochs: "))
    epochs_stage2 = int(input("Enter number of Stage 2 epochs: "))

if os.path.exists(output_folder) and not restart_from_checkpoint:
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

checkpoint_path = f"{output_folder}/checkpoint.pt"

# Problem defn.
omega = 15
helmholtz_k = 20

def exact_solution(x):
    return torch.sin(omega * np.pi * x)

def forcing(x):
    return ((omega * np.pi)**2 - helmholtz_k**2) * torch.sin(omega * np.pi * x)

# Training Parameters.
N_f = int(input("Enter no. of Collocation points: "))
save_every = 100

window = 500 # over which stagnation is determined.
stagnation_tol = 1e-3
stagnation_loss_threshold = 1e-4

hotspot_tau = 0.5
hotspot_min_width = 0.10

beta_init = 100.0


# This is only an initial guess; xL and xR will be learned.
initial_window_left_fraction = 0.10
initial_window_right_fraction = 0.90


# COLLOCATION AND BOUNDARY POINTS
x_global = torch.linspace(0.0, 1.0, N_f).view(-1, 1).to(device)

x0 = torch.tensor([[0.0]], device=device)
x1 = torch.tensor([[1.0]], device=device)


def get_collocation_batch(x_pool, batch_size):
    if batch_size is None or batch_size >= x_pool.shape[0]:
        return x_pool
    idx = torch.randperm(x_pool.shape[0], device=x_pool.device)[:batch_size]
    return x_pool[idx]

# NN
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
        return self.net(x) # defined self.net as the NN.


## Anything part of nn.parameters is trainable (also requires_grad = true)

class WindowedLocalPINN(nn.Module): 
    """
    Simple 1D windowed local correction model.
    It represents: u_local_windowed(x) = W(x) * NN_local(xi)
    where:
        W(x) = sigmoid(beta*(x - xL)) * sigmoid(beta*(xR - x))
    and xi = (x - center) / half_width.

    The trainable parameters are:  xL, xR, beta
    The parameterization guarantees:
        0 < xL < xR < 1
    """

    def __init__(self, xL_init, xR_init, beta_init=100.0):
        super().__init__() 

        xL_init = float(xL_init) 
        xR_init = float(xR_init) 

        if xR_init <= xL_init: 
            raise ValueError("xR_init must be greater than xL_init.") 

        xL_init = min(max(xL_init, 1e-4), 1.0 - 2e-4) 
        xR_init = min(max(xR_init, xL_init + 1e-4), 1.0 - 1e-4)  

        width_fraction_init = (xR_init - xL_init) / (1.0 - xL_init) #  inverts the forward relationship xR = xL+f*(1-xL)
        width_fraction_init = min(max(width_fraction_init, 1e-4), 1.0 - 1e-4)

        self.net = nn.Sequential( 
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

        self.xL = nn.Parameter( torch.tensor([[xL_init]], dtype=torch.float32) )
        self.width_fraction = nn.Parameter( torch.tensor([[width_fraction_init]], dtype=torch.float32) )
        if beta_init > 20.0:  
            raw_beta_init = beta_init  
        else:  
            raw_beta_init = np.log(np.expm1(beta_init)) 
        self.raw_beta = nn.Parameter( torch.tensor([[raw_beta_init]], dtype=torch.float32) ) 

    def window_parameters(self):
        beta = F.softplus(self.raw_beta)  # log(1 + exp(z))
        xR = self.xL + (1 - self.xL) * self.width_fraction

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


def project_window_parameters(local_model):
    """
    Hard projection step.
    Keeps:
        0 <= xL <= 1 - min_width
        0 < width_fraction < 1
    """

    with torch.no_grad():
        domain_width = 1 # local_model.x_max - local_model.x_min
        min_width = 1e-3 * domain_width

        lower_xL = 0 # local_model.x_min.item()
        upper_xL = 1 - min_width # (local_model.x_max - min_width).item()

        local_model.xL.clamp_(lower_xL, upper_xL)
        local_model.width_fraction.clamp_(1e-4, 1.0 - 1e-4)


#Define initial local window
def initial_local_window():
    xL_init = initial_window_left_fraction
    xR_init = initial_window_right_fraction

    if xR_init <= xL_init:
        raise ValueError("Initial local window must satisfy xR_init > xL_init.")

    return xL_init, xR_init


# Optimizer factory
def create_optimizer(parameters, optimizer_type, learning_rate):
    if optimizer_type == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)

    elif optimizer_type == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)

    elif optimizer_type == "gd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.0)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

# This is the right way to define optimizer. See the 1D toy example for linear regression: https://github.com/DigoShane/GitHub-ML/blob/main/Training_the_sum_of_2_NN's.ipynb
def make_optimizer(global_model, local_model=None):
    if local_model is None:
        parameters = global_model.parameters()
    else:
        parameters = list(global_model.parameters()) + list(local_model.parameters())
        # model.parameters() is a generator so wrapping in list to enable adding them.

    return create_optimizer(parameters, optimizer_type, learning_rate)

print()
print("Optimizer:", optimizer_type)
print("Learning rate:", learning_rate)
print()


# derivative
def derivatives_from_output(u, x_req):
    u_x = torch.autograd.grad( u, x_req, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad( u_x, x_req, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_x, u_xx


# residual
def global_residual(global_model, x):
    x_req = x.clone().detach().requires_grad_(True)
    u = global_model(x_req)
    _, u_xx = derivatives_from_output(u, x_req)

    return -u_xx - helmholtz_k**2 * u - forcing(x_req)

def enriched_solution(x, global_model, local_model):
    return global_model(x) + local_model(x)


def enriched_residual(x, global_model, local_model):
    x_req = x.clone().detach().requires_grad_(True)
    u = enriched_solution(x_req, global_model, local_model)
    _, u_xx = derivatives_from_output(u, x_req)

    return -u_xx - helmholtz_k**2 * u - forcing(x_req)


# Loss Fn
def global_loss(global_model, x_batch):
    r = global_residual(global_model, x_batch)
    loss_pde = torch.mean(r**2)
    loss_bc = ( torch.mean(global_model(x0)**2) + torch.mean(global_model(x1)**2) )
    loss = loss_pde + loss_bc

    return loss, loss_pde, loss_bc


def enriched_loss(global_model, local_model, x_batch):
    r = enriched_residual(x_batch, global_model, local_model)
    loss_pde = torch.mean(r**2)
    u0 = enriched_solution(x0, global_model, local_model)
    u1 = enriched_solution(x1, global_model, local_model)
    loss_bc = torch.mean(u0**2) + torch.mean(u1**2) # both u0 and u1 should be 0.
    loss = loss_pde + loss_bc 

    return loss, loss_pde, loss_bc, loss_window


#residual indicator
def residual_indicator_from_residual(r):
    r_sq = r.detach()**2
    # commented out. Choose one or the other.
    #grad_r = torch.autograd.grad( r, x, grad_outputs=torch.ones_like(r), create_graph=False, retain_graph=True)[0]
    grad_r_sq = 0 # grad_r.detach()**2
    eta = r_sq + grad_r_sq 
    
    eta = eta.cpu().numpy().flatten()
    eta = np.log(eta + 1e-12)
    
    return eta


# Saving and plotting. Called during Stage 1.
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
    plt.savefig( f"{output_folder}/global_solution_epoch_{epoch:05d}.png", dpi=150, bbox_inches="tight")
    plt.close()


# Called during Stage 1, where global model applies.
def save_global_eta_plot(epoch, global_model):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)

    r = global_residual(global_model, x_test)
    eta = residual_indicator_from_residual(r)

    x_np = x_test.detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 4))
    plt.plot( x_np, eta, linewidth=0.8, label=r"$\eta=\log(r^2)$")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
    plt.xlabel("x")
    plt.ylabel(r"$\eta(x)$")
    plt.title(f"Global residual indicator — Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"{output_folder}/global_eta_epoch_{epoch:05d}.png", dpi=150, bbox_inches="tight")
    plt.close()


# Plots the current window for w(x) and vertical lines for xL ad xR.
#Called during local model creation and stage 2.
def save_trainable_window_plot(local_model, epoch):
    x_plot = torch.linspace(0, 1, 2000).view(-1, 1).to(device)
    local_model.eval()
    with torch.inference_mode():
        w_plot = local_model.window(x_plot)
        xL_current, xR_current, beta_current = local_model.window_parameters()

    plt.figure(figsize=(8, 4))
    plt.plot( x_plot.cpu().numpy(), w_plot.cpu().numpy(), label="Trainable smooth window")
    plt.axvline( xL_current.item(), color="black", linestyle=":", linewidth=0.8, label="xL")
    plt.axvline( xR_current.item(), color="black", linestyle="--", linewidth=0.8, label="xR")
    plt.xlabel("x")
    plt.ylabel("W(x)")
    plt.title(
        f"Window — Epoch {epoch}: "
        f"xL={xL_current.item():.4f}, "
        f"xR={xR_current.item():.4f}, "
        f"beta={beta_current.item():.2f}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"{output_folder}/window_epoch_{epoch:05d}.png", dpi=150, bbox_inches="tight" )
    plt.close()

# u_snapshot is the solution just before enrichment starts. Obtained during LOCAL MODEL CREATION.
# x_snapshot passed in as a discretization of the domain.
def save_enriched_solution( epoch, global_model, local_model, xL, xR, u_snapshot=None, x_snapshot=None):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)

    global_model.eval()
    local_model.eval()

    with torch.inference_mode():
        u_pred = enriched_solution(x_test, global_model, local_model, xL, xR)
        u_exact = exact_solution(x_test)

    plt.figure(figsize=(8, 4))
    plt.plot( x_test.cpu().numpy(), u_exact.cpu().numpy(), label="Exact")
    plt.plot( x_test.cpu().numpy(), u_pred.cpu().numpy(), "--", label="Enriched PINN")

    if u_snapshot is not None and x_snapshot is not None:
        plt.plot( x_snapshot, u_snapshot, linestyle="--", linewidth=0.7, alpha=0.7, label="Global PINN at enrichment start")

    plt.axvline(xL, color="black", linestyle=":", linewidth=0.8)
    plt.axvline(xR, color="black", linestyle=":", linewidth=0.8)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"Enriched solution — Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"{output_folder}/enriched_solution_epoch_{epoch:05d}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_enriched_eta_plot( epoch, global_model, local_model):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)
    r = enriched_residual(x_test, global_model, local_model, xL, xR)
    eta = residual_indicator_from_residual(r)
    x_np = x_test.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 4))
    plt.plot( x_np, eta, linewidth=0.8, label=r"Current $\eta=\log(r^2)$")
    if x_eta_snapshot is not None and eta_snapshot is not None:
        plt.plot( x_eta_snapshot, eta_snapshot, linestyle="--", linewidth=0.7, alpha=0.7, label=r"$\eta$ at enrichment start")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
    plt.axvline(xL, color="black", linestyle=":", linewidth=0.8)
    plt.axvline(xR, color="black", linestyle=":", linewidth=0.8)
    plt.xlabel("x")
    plt.ylabel(r"$\eta(x)$")
    plt.title(f"Enriched residual indicator — Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"{output_folder}/enriched_eta_epoch_{epoch:05d}.png", dpi=150, box_inches="tight")
    plt.close()


# Stagnation check
def check_stagnation(loss_history, window, stagnation_tol, loss_threshold):
    if len(loss_history) <= 2 * window:
        return False

    old_loss = np.mean(loss_history[-2 * window:-window])
    new_loss = np.mean(loss_history[-window:])
    improvement = (old_loss - new_loss) / old_loss
    if improvement < stagnation_tol and new_loss > loss_threshold: # 2nd condition is checking if we have converged or not.
        return True

    return False


# Checkpointing
def save_checkpoint( checkpoint_path, global_model, local_model, optimizer, stage1_end_epoch, loss_history_stage1, 
                     loss_history_stage2, x_snapshot_sol, u_snapshot):
    checkpoint = {  "global_model_state_dict": global_model.state_dict(),
                    "local_model_state_dict": ( None if local_model is None else local_model.state_dict() ),
                    "optimizer_state_dict": optimizer.state_dict(), "checkpoint_optimizer_type": optimizer_type,
                    "checkpoint_learning_rate": learning_rate, "stage1_end_epoch": stage1_end_epoch,
                    "loss_history_stage1": loss_history_stage1, "loss_history_stage2": loss_history_stage2,
                    "x_snapshot_sol": x_snapshot_sol, "u_snapshot": u_snapshot}

    torch.save(checkpoint, checkpoint_path)

# When you restart, this creates the local model.
def create_local_model_from_checkpoint(checkpoint):
    state_dict = checkpoint["local_model_state_dict"]

    if state_dict is None:
        raise RuntimeError( "Checkpoint does not contain a local model. "
            "Restart is only supported for Stage 2.")

    # Since we used "self.xL = nn.Parameter(torch.tensor([[xL_init]], ...))", it is now a parameter tracked by the dict.
    xL_loaded = state_dict["xL"].item() # .item() extracts std Python number out of the 0-dimensional PyTorch tensor.
    width_fraction_loaded = state_dict["width_fraction"].item()
    xR_loaded = xL_loaded + width_fraction_loaded
    #creating a new model for the next re-run.
    local_model = WindowedLocalPINN( xL_init=xL_loaded, xR_init=xR_loaded, beta_init=beta_init).to(device)
    local_model.load_state_dict(state_dict)

    return local_model

# torch.load 's map_location=device -> when a checkpoint is saved, each tensor inside it remembers which device it was on at 
#                                      save time (e.g., "cuda:0"). If you later try to load that checkpoint on a machine/session 
#                                      where that exact device isn't available (e.g., you saved on GPU but are now loading on a 
#                                      CPU-only machine, or a machine with different GPU indexing), torch.load would otherwise 
#                                      try to recreate tensors on the original device and crash if that device doesn't exist. 
#                                      map_location=device tells torch.load to remap every tensor to the specified device as 
#                                      it's loaded, overriding whatever device it was originally saved on — ensuring the loaded 
#                                      checkpoint always lands correctly on whatever device your current script is using 
#                                      (CPU or GPU), regardless of where it was originally saved. 

def load_checkpoint_models(checkpoint_path, global_model):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    global_model.load_state_dict(checkpoint["global_model_state_dict"])
    local_model = create_local_model_from_checkpoint(checkpoint)
    stage1_end_epoch = checkpoint["stage1_end_epoch"]
    loss_history_stage1 = checkpoint["loss_history_stage1"]
    loss_history_stage2 = checkpoint["loss_history_stage2"]
    x_snapshot_sol = checkpoint["x_snapshot_sol"]
    u_snapshot = checkpoint["u_snapshot"]

    return ( checkpoint, local_model, stage1_end_epoch, loss_history_stage1, loss_history_stage2,
             x_snapshot_sol, u_snapshot)


def maybe_load_optimizer_state(checkpoint, optimizer, load_optimizer_state):
    if not load_optimizer_state: # We are not loading the optimizer state when rerun (at checkpoint).
       print("Loaded model weights only. Optimizer state was not loaded.")
       return

    # In the dict, the value corresponds to adam, sgd or gd.
    checkpoint_optimizer_type = checkpoint.get("checkpoint_optimizer_type", None) 
    checkpoint_learning_rate = checkpoint.get("checkpoint_learning_rate", None)
    # dict.get(key, default) looks up key; if the key exists, it returns the corresponding value;
    # if the key does not exist, instead of raising a KeyError (which checkpoint["checkpoint_optimizer_type"] would do), 
    # it returns the default you provided — here, None.

    #The following evaluate to true or flase values.
    same_optimizer = ( checkpoint_optimizer_type == optimizer_type )
    same_lr = ( checkpoint_learning_rate == learning_rate )

    if same_optimizer and same_lr: # both true. 
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state from checkpoint.")
        except Exception as error:
            print("Could not load optimizer state.")
            print("Reason:", error)
            print("Continuing with fresh optimizer state.")
    else:
        print("Optimizer state was not loaded because optimizer settings changed.")
        print("Checkpoint optimizer:", checkpoint_optimizer_type)
        print("Current optimizer:", optimizer_type)
        print("Checkpoint learning rate:", checkpoint_learning_rate)
        print("Current learning rate:", learning_rate)
        print("Continuing with fresh optimizer state.")


# Initialize model history
global_model = PINN().to(device)
local_model = None

loss_history_stage1 = []
loss_history_stage2 = []

# We store where stage1 ends. Then when we count epochs in stage 2. This is then used to define total_epoch, which stores the total no. of epoch throughout.
stage1_end_epoch = 0

x_snapshot_sol = None
u_snapshot = None


# Restart Logic
if restart_from_checkpoint:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

    ( checkpoint, local_model, stage1_end_epoch, loss_history_stage1, loss_history_stage2,
        x_snapshot_sol, u_snapshot ) = load_checkpoint_models(checkpoint_path, global_model)
    optimizer = make_optimizer(global_model, local_model)
    maybe_load_optimizer_state(checkpoint, optimizer, load_optimizer_state)

    print()
    print("Restarted from checkpoint.")
    print("Restart means: continuing Stage 2 enriched training only.")
    print("Optimizer:", optimizer_type)
    print("Learning rate:", learning_rate)
    print("Existing Stage 2 epochs:", len(loss_history_stage2))
    print("Additional Stage 2 epochs requested:", additional_stage2_epochs)
    print()

else:
    optimizer = make_optimizer(global_model, None)


# Stage 1: Global TRaining
if not restart_from_checkpoint: # not restarting from checkpoint means we are training the single NN.

    print()
    print("=" * 60)
    print("STAGE 1: GLOBAL TRAINING")
    print("=" * 60)
    print()

    stagnation_detected = False

    # 1. For epoch in range
    # 2. Call model dot train
    # 3. Do the forward pass
    # 4. Calculate the Lass (loss)
    # 5. Optimizer zero (grad)
    # 6. Loss backwards
    # 7. Optimizer step
    # Testing 
    # 1. Call model dot eval
    # 2. With torch infarance mall (inference mode)
    # 3. Do the forward pass
    # 4. Calculate the lass (loss)
    # Print out what's happening

    for epoch in range(epochs_stage1):
        global_model.train() # forward pass calculated via "u = global_model(x_req)" called via global_loss, which calls pde_residual, which contains this.
        x_batch = get_collocation_batch(x_global, batch_size)
        loss, loss_pde, loss_bc = global_loss(global_model, x_batch) 
        optimizer.zero_grad() #?? This was initially above
        loss.backward()
        optimizer.step()

        loss_history_stage1.append(loss.item()) # This is the testing loop since we are not doing supervised learning.

        if epoch % save_every == 0: # print out whats happening+ Testing ( but only at certain points).
            global_model.eval()
            with torch.inference_mode():
                x_test = torch.linspace(0, 1, 1000).view(-1, 1).to(device)
                u_pred = global_model(x_test)
                u_exact = exact_solution(x_test)
                rel_error = torch.norm(u_exact - u_pred) / torch.norm(u_exact) 
                # this is exactly what the testing was doing in the supervised learning problem.

            print(
                f"[GLOBAL] "
                f"Epoch {epoch:6d} | "
                f"Loss = {loss.item():.4e} | "
                f"PDE = {loss_pde.item():.4e} | "
                f"BC = {loss_bc.item():.4e} | "
                f"RelL2 = {rel_error.item():.4e}")

            save_global_solution(epoch, global_model)
            save_global_eta_plot(epoch, global_model)

        if check_stagnation( loss_history_stage1, window, stagnation_tol, stagnation_loss_threshold):
            print()
            print("STAGNATION DETECTED DURING GLOBAL TRAINING")
            print(f"Stopping Stage 1 at epoch {epoch}")
            print()

            stagnation_detected = True
            break

    stage1_end_epoch = len(loss_history_stage1) - 1

    if not stagnation_detected:
        print()
        print("Stage 1 reached maximum epochs without stagnation.")
        print("Introducing local model anyway.")
        print()


# LOCAL MODEL CREATION
# The code is set up such that stage 1 and stage 2 are run each time the code is run. 
# We want to make sure that if we are starting from a check point, then we are starting from an old code. 
# So we can avoid this local model creation. This is why the following is checking that.
# It does 4 things -- (1) Creates the initial local window (2) Saves the global model prediction before enrichment.
# (3) Instantiates the local model (4) Creates a new optimizer containing both global and local model parameters.
if not restart_from_checkpoint: 
    print()
    print("=" * 60)
    print("CREATING LOCAL MODEL")
    print("=" * 60)
    print()

    xL_init, xR_init = initial_local_window() # initial window creation.
    print("Initial local window guess:")
    print("xL_init =", xL_init)
    print("xR_init =", xR_init)
    print()

    x_snapshot_sol_torch = torch.linspace(0, 1, 1000).view(-1, 1).to(device)
    global_model.eval()
    with torch.inference_mode(): # saving the u(x_snapshot) of the global model.
        u_snapshot = global_model(x_snapshot_sol_torch).cpu().numpy().flatten()

    #Instantiate the local model.
    x_snapshot_sol = x_snapshot_sol_torch.cpu().numpy().flatten() # this will be passed as x_snapshot soln to save_enriched_solution.
    local_model = WindowedLocalPINN( xL_init=xL_init, xR_init=xR_init, beta_init=beta_init).to(device)
    project_window_parameters(local_model) # project xL and xR to within the domain.
    optimizer = make_optimizer(global_model, local_model) # point 4 above.

    save_trainable_window_plot(local_model, stage1_end_epoch)

    save_checkpoint( checkpoint_path, global_model, local_model, optimizer, stage1_end_epoch, loss_history_stage1, 
                     loss_history_stage2, x_snapshot_sol, u_snapshot)


# Stage 2: Enriched TRaining
print()
print("=" * 60)
print("STAGE 2: ENRICHED TRAINING")
print("=" * 60)
print()

if local_model is None: # we just defined the local model above.
    raise RuntimeError("local_model is None. Cannot run Stage 2.")

# This is how resumability is defined for stage 2.
start_epoch_stage2 = len(loss_history_stage2) # When stage 2 runs for the 1st time, this is 0.
final_epoch_stage2 = start_epoch_stage2 + additional_stage2_epochs

# Training over only the additional_stage2_epochs
for epoch in range(start_epoch_stage2, final_epoch_stage2):
    global_model.train()
    local_model.train()
    x_batch = get_collocation_batch(x_global, batch_size)
    loss, loss_pde, loss_bc = enriched_loss(global_model, local_model, x_batch)
    optimizer.zero_grad()
    if torch.isnan(loss):
        print("NaN detected during enriched training.")
        break
    loss.backward()
    optimizer.step()

    project_window_parameters(local_model)
    loss_history_stage2.append(loss.item())

    total_epoch = stage1_end_epoch + epoch

    # Testing time.
    if epoch % save_every == 0:
        global_model.eval()
        local_model.eval()

        with torch.inference_mode():
            x_test = torch.linspace(0, 1, 1000).view(-1, 1).to(device)
            u_pred = enriched_solution(x_test, global_model, local_model)
            u_exact = exact_solution(x_test)
            rel_error = torch.norm(u_exact - u_pred) / torch.norm(u_exact)

            xL_current, xR_current, beta_current = local_model.window_parameters()

        print(
            f"[ENRICHED] "
            f"Total epoch = {total_epoch:6d} | "
            f"Stage2 epoch = {epoch:6d} | "
            f"Loss = {loss.item():.4e} | "
            f"PDE = {loss_pde.item():.4e} | "
            f"BC = {loss_bc.item():.4e} | "
            f"RelL2 = {rel_error.item():.4e} | "
            f"xL = {xL_current.item():.4f} | "
            f"xR = {xR_current.item():.4f} | "
            f"beta = {beta_current.item():.2f}"
        )

        save_enriched_solution( total_epoch, global_model, local_model, u_snapshot=u_snapshot, x_snapshot=x_snapshot_sol)
        save_enriched_eta_plot( total_epoch, global_model, local_model)
        save_trainable_window_plot(local_model, total_epoch)
        save_checkpoint( checkpoint_path, global_model, local_model, optimizer, stage1_end_epoch, loss_history_stage1,
                         loss_history_stage2, x_snapshot_sol, u_snapshot)



# FInal Error
x_test = torch.linspace(0, 1, 2000).view(-1, 1).to(device)
global_model.eval()
local_model.eval()

with torch.inference_mode():
    u_pred = enriched_solution(x_test, global_model, local_model)
    u_exact = exact_solution(x_test)
relative_L2_error = torch.norm(u_exact - u_pred) / torch.norm(u_exact)

print()
print("FINAL RELATIVE L2 ERROR")
print(relative_L2_error.item())

with open(f"{output_folder}/final_error.txt", "w") as f:
    f.write(f"Final relative L2 error = {relative_L2_error.item():.12e}\n")



# Save Loss History
n1 = len(loss_history_stage1)
n2 = len(loss_history_stage2)

plt.figure(figsize=(10, 4))

if n1 > 0:
    plt.plot( np.arange(n1), loss_history_stage1, label="Stage 1: Global")
if n2 > 0:
    plt.plot( np.arange(n1, n1 + n2), loss_history_stage2, label="Stage 2: Enriched")
if n1 > 0 and n2 > 0:
    plt.axvline( n1, color="black", linestyle=":", linewidth=1.0, label="Enrichment start")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss history")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig( f"{output_folder}/loss_history.png", dpi=150, bbox_inches="tight")
plt.close()


if n1 > 0:
    np.savetxt( f"{output_folder}/loss_history_stage1.txt", np.array(loss_history_stage1), header="stage1_global_loss")
if n2 > 0:
    np.savetxt( f"{output_folder}/loss_history_stage2.txt", np.array(loss_history_stage2), header="stage2_enriched_loss")



# Save Check point. Called during Local model creation and stage 2.
save_checkpoint( checkpoint_path, global_model, local_model, optimizer, stage1_end_epoch, loss_history_stage1,
                 loss_history_stage2, x_snapshot_sol, u_snapshot)
print()
print(f"Saved all results in: {output_folder}")


# printing out the global and local model dict just to see how things are stored.
print("=== Global model state_dict ===")
for key, tensor in global_model.state_dict().items():
    print(key, tensor.shape)

print("=== Local model state_dict ===")
for key, tensor in local_model.state_dict().items():
    print(key, tensor.shape)









