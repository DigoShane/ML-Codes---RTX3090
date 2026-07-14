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


# COLLOCATION AND BOUNDARY POINTS
x_global = torch.linspace(0.0, 1.0, N_f).view(-1, 1).to(device)

x0 = torch.tensor([[0.0]], device=device)
x1 = torch.tensor([[1.0]], device=device)


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

class WindowedLocalPINN(nn.Module):  #n#
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
        x_min <= xL <= x_max - min_width
        0 < width_fraction < 1
    """

    with torch.no_grad():
        domain_width = 1 # local_model.x_max - local_model.x_min
        min_width = 1e-3 * domain_width

        lower_xL = 0 # local_model.x_min.item()
        upper_xL = 1 - min_width # (local_model.x_max - min_width).item()

        local_model.xL.clamp_(lower_xL, upper_xL)
        local_model.width_fraction.clamp_(1e-4, 1.0 - 1e-4)


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

def make_optimizer(global_model, local_model=None):
    if local_model is None:
        parameters = global_model.parameters()
    else:
        parameters = list(global_model.parameters()) + list(local_model.parameters())

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
def global_loss(global_model):
    r = global_residual(global_model, x_global)
    loss_pde = torch.mean(r**2)
    loss_bc = ( torch.mean(global_model(x0)**2) + torch.mean(global_model(x1)**2) )
    loss = loss_pde + loss_bc

    return loss, loss_pde, loss_bc


def enriched_loss(global_model, local_model):
    r = enriched_residual(x_global, global_model, local_model)
    loss_pde = torch.mean(r**2)
    u0 = enriched_solution(x0, global_model, local_model)
    u1 = enriched_solution(x1, global_model, local_model)
    loss_bc = torch.mean(u0**2) + torch.mean(u1**2) # both u0 and u1 should be 0.
    loss = loss_pde + loss_bc 

    return loss, loss_pde, loss_bc, loss_window


#residual indicator
def residual_indicator_from_residual(r):
    r_sq = r.detach()**2
    # commented out
    #grad_r = torch.autograd.grad( r, x, grad_outputs=torch.ones_like(r), create_graph=False, retain_graph=True)[0]
    grad_r_sq = 0 # grad_r.detach()**2
    eta = r_sq + grad_r_sq 
    
    eta = eta.cpu().numpy().flatten()
    eta = np.log(eta + 1e-12)
    
    return eta


# Saving and plotting
def save_global_solution(epoch, global_model):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)

    global_model.eval()

    with torch.no_grad():
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

def save_trainable_window_plot(local_model, epoch):
    x_plot = torch.linspace(x_min, x_max, 2000).view(-1, 1).to(device)

    local_model.eval()

    with torch.no_grad():
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


def save_enriched_solution( epoch, global_model, local_model, xL, xR, u_snapshot=None, x_snapshot=None):
    x_test = torch.linspace(0.0, 1.0, 1000).view(-1, 1).to(device)

    global_model.eval()
    local_model.eval()

    with torch.no_grad():
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


def save_enriched_eta_plot( epoch, global_model, local_model, xL, xR, x_eta_snapshot=None, eta_snapshot=None):
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

def save_window_plot(xL, xR):
    x_plot = torch.linspace(0.0, 1.0, 2000).view(-1, 1).to(device)

    w_plot = window_function(x_plot, xL, xR)
    plt.figure(figsize=(8, 4))

    plt.plot( x_plot.cpu().numpy(), w_plot.cpu().numpy(), label="Window")
    plt.axvline(xL, color="black", linestyle=":", linewidth=0.8)
    plt.axvline(xR, color="black", linestyle=":", linewidth=0.8)
    plt.xlabel("x")
    plt.ylabel("w(x)")
    plt.title("Characteristic enrichment window")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"{output_folder}/enrichment_window.png", dpi=150, bbox_inches="tight")
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


def create_local_model_from_checkpoint(checkpoint):
    state_dict = checkpoint["local_model_state_dict"]

    if state_dict is None:
        raise RuntimeError( "Checkpoint does not contain a local model. "
            "Restart is only supported for Stage 2.")

    xL_loaded = state_dict["xL"].item() # .item() extracts std Python number out of the 0-dimensional PyTorch tensor.
    width_fraction_loaded = state_dict["width_fraction"].item()
    xR_loaded = xL_loaded + width_fraction_loaded
    local_model = WindowedLocalPINN( xL_init=xL_loaded, xR_init=xR_loaded, beta_init=beta_init).to(device)
    local_model.load_state_dict(state_dict)

    return local_model


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

    #The following evaluate to true or flase values.
    same_optimizer = checkpoint_optimizer_type == optimizer_type
    same_lr = checkpoint_learning_rate == learning_rate

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

















