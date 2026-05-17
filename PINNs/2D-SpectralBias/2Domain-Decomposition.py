import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DOMAIN DEFINITIONS
# ============================================================

def in_square(x):
    return (x[:,0]>=0) & (x[:,0]<=1) & (x[:,1]>=0) & (x[:,1]<=1)

def in_circle(x):
    return (x[:,0]**2 + x[:,1]**2 <= 1)

def in_ellipse(x):
    return ((x[:,0]/1.5)**2 + (x[:,1]/1.0)**2 <= 1)

def in_cardioid(x):
    r = torch.sqrt(x[:,0]**2 + x[:,1]**2)
    theta = torch.atan2(x[:,1], x[:,0])
    return r <= (1 + torch.cos(theta))

# SELECT DOMAIN HERE
domain_fn = in_circle

# ============================================================
# SAMPLE POINTS IN DOMAIN
# ============================================================

def sample_domain(N):
    pts_list = []

    while True:
        x = torch.rand(N, 2) * 2 - 1  # sample in [-1,1]^2
        mask = domain_fn(x)

        valid = x[mask]
        if len(valid) > 0:
            pts_list.append(valid)

        pts = torch.cat(pts_list, dim=0)

        if pts.shape[0] >= N:
            return pts[:N].to(device)

# ============================================================
# POISSON PROBLEM
# ============================================================

def exact_u(x):
    return torch.sin(np.pi*x[:,0:1]) * torch.sin(np.pi*x[:,1:2])

def f(x):
    return 2*np.pi**2 * exact_u(x)

# ============================================================
# NEURAL NETWORK
# ============================================================

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# PDE RESIDUAL
# ============================================================

def laplacian(model, x):
    x.requires_grad_(True)
    u = model(x)

    grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(grad_u[:,0], x, torch.ones_like(grad_u[:,0]), create_graph=True)[0][:,0]
    u_yy = torch.autograd.grad(grad_u[:,1], x, torch.ones_like(grad_u[:,1]), create_graph=True)[0][:,1]

    return u_xx + u_yy

def residual(model, x):
    return -laplacian(model, x) - f(x)

# ============================================================
# VORONOI PARTITION
# ============================================================

def assign_cells(x, seeds):
    dists = torch.cdist(x, seeds)
    return torch.argmin(dists, dim=1)

# ============================================================
# MAIN SETUP
# ============================================================

N_points = 5000
points = sample_domain(N_points)

# Random seeds
N_seeds = 10
seeds = sample_domain(N_seeds)

cell_ids = assign_cells(points, seeds)

# Pick one region
target_region = 0

mask1 = (cell_ids == target_region)
mask2 = ~mask1

x1 = points[mask1]
x2 = points[mask2]

# ============================================================
# INTERFACE POINTS (approximate)
# ============================================================

def get_interface_points(points, cell_ids, target):
    interface_pts = []
    for i in range(len(points)):
        neighbors = torch.norm(points - points[i], dim=1) < 0.05
        if torch.any(cell_ids[neighbors] != cell_ids[i]):
            if cell_ids[i] == target or torch.any(cell_ids[neighbors] == target):
                interface_pts.append(points[i])
    if len(interface_pts) == 0:
        return torch.empty(0,2).to(device)
    return torch.stack(interface_pts)

x_interface = get_interface_points(points, cell_ids, target_region)

# ============================================================
# MODELS
# ============================================================

model1 = PINN().to(device)
model2 = PINN().to(device)

optimizer = torch.optim.Adam(
    list(model1.parameters()) + list(model2.parameters()), lr=1e-3
)

# ============================================================
# LOSS FUNCTION
# ============================================================

def loss_fn():
    loss_pde = torch.mean(residual(model1, x1)**2) + \
               torch.mean(residual(model2, x2)**2)

    # Boundary (approximate: points near boundary)
    boundary_mask = torch.norm(points, dim=1) > 0.95
    xb = points[boundary_mask]

    loss_bc = torch.mean(model1(xb)**2) + torch.mean(model2(xb)**2)

    # Interface continuity
    if len(x_interface) > 0:
        loss_int = torch.mean((model1(x_interface) - model2(x_interface))**2)
    else:
        loss_int = 0.0

    return loss_pde + 10*loss_bc + 10*loss_int

# ============================================================
# TRAINING
# ============================================================

epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4e}")

# ============================================================
# VISUALIZATION
# ============================================================

x_test = sample_domain(2000)

pred = []
with torch.no_grad():
    for x in x_test:
        d = torch.norm(x - seeds, dim=1)
        region = torch.argmin(d)
        if region == target_region:
            pred.append(model1(x.view(1,2)).item())
        else:
            pred.append(model2(x.view(1,2)).item())

pred = np.array(pred)

plt.scatter(x_test[:,0].cpu(), x_test[:,1].cpu(), c=pred, cmap='viridis')
plt.colorbar()
plt.title("Predicted Solution")
plt.show()