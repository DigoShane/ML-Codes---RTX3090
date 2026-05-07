import numpy as np
import torch
import glob

files = sorted(glob.glob("dataset/sample_*.npz"))

X_list = []
Y_list = []

for f in files:
    data = np.load(f)

    K = data["K"]
    coords = data["coords"]      # (N,2)
    m = data["m"]                # (N,2)

    N = coords.shape[0]

    K_col = np.full((N,1), K)

    X = np.hstack([K_col, coords])   # (N,3)
    Y = m                            # (N,2)

    X_list.append(X)
    Y_list.append(Y)

X_all = np.vstack(X_list)
Y_all = np.vstack(Y_list)

# Convert to torch
X_all = torch.tensor(X_all, dtype=torch.float32)
Y_all = torch.tensor(Y_all, dtype=torch.float32)

print(X_all.shape, Y_all.shape)


import torch.nn as nn

class FieldNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        m = self.net(x)
        # enforce |m| = 1
        return m / (torch.norm(m, dim=1, keepdim=True) + 1e-8)

import torch.optim as optim

model = FieldNet()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

batch_size = 4096
epochs = 2000

for epoch in range(epochs):

    idx = torch.randperm(X_all.shape[0])
    X_shuffled = X_all[idx]
    Y_shuffled = Y_all[idx]

    for i in range(0, X_all.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]

        optimizer.zero_grad()

        pred = model(X_batch)
        loss = loss_fn(pred, Y_batch)

        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6e}")

K_test = 2.3

coords = X_list[0][:,1:]   # reuse mesh

K_col = torch.full((coords.shape[0],1), K_test)
coords_t = torch.tensor(coords, dtype=torch.float32)

X_test = torch.cat([K_col, coords_t], dim=1)

with torch.no_grad():
    m_pred = model(X_test).numpy()

def compute_energy(model, K):

    xy = torch.rand(5000,2)

    K_col = K.expand(xy.shape[0],1)
    X = torch.cat([K_col, xy], dim=1)

    X.requires_grad_(True)

    m = model(X)

    # compute gradients
    grads = torch.autograd.grad(
        m, X,
        grad_outputs=torch.ones_like(m),
        create_graph=True
    )[0]

    dm_dx = grads[:,1:3]

    exchange = (dm_dx**2).sum(dim=1).mean()

    anisotropy = (1 - m[:,0]**2).mean()

    return exchange + K * anisotropy

K_opt = torch.tensor([[1.0]], requires_grad=True)

optimizer_K = optim.Adam([K_opt], lr=0.05)

history = []

for i in range(200):

    optimizer_K.zero_grad()

    J = compute_energy(model, K_opt)

    J.backward()
    optimizer_K.step()

    with torch.no_grad():
        K_opt.clamp_(0.1, 5)

    history.append(K_opt.item())

    if i % 20 == 0:
        print(i, K_opt.item(), J.item())

