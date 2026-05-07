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

import matplotlib.pyplot as plt
import numpy as np
import torch

def compare_with_fem(model, file_path):

    data = np.load(file_path)

    K = data["K"]
    coords = data["coords"]      # (N,2)
    m_true = data["m"]           # (N,2)

    device = next(model.parameters()).device

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    K_col = torch.full((coords.shape[0],1), float(K), device=device)

    X = torch.cat([K_col, coords_t], dim=1)

    with torch.no_grad():
        m_pred = model(X).cpu().numpy()

    error = np.linalg.norm(m_pred - m_true, axis=1)

    # Plot error field
    plt.figure(figsize=(6,5))
    plt.scatter(coords[:,0], coords[:,1],
                c=error, cmap="viridis", s=5)

    plt.colorbar(label="|m_NN - m_FEM|")
    plt.title(f"Error field for K = {K:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

    print("Mean error:", error.mean())
    print("Max error:", error.max())

def plot_line_cut(model, file_path):

    data = np.load(file_path)

    K = data["K"]
    coords = data["coords"]
    m_true = data["m"]

    mask = np.abs(coords[:,1] - 0.5) < 0.01

    coords_cut = coords[mask]
    m_true_cut = m_true[mask]

    coords_t = torch.tensor(coords_cut, dtype=torch.float32)
    K_col = torch.full((coords_cut.shape[0],1), float(K))

    X = torch.cat([K_col, coords_t], dim=1)

    with torch.no_grad():
        m_pred_cut = model(X).numpy()

    idx = np.argsort(coords_cut[:,0])
    x = coords_cut[idx,0]

    plt.plot(x, m_true_cut[idx,0], label="FEM m_x")
    plt.plot(x, m_pred_cut[idx,0], "--", label="NN m_x")

    plt.legend()
    plt.title(f"Line cut at y≈0.5, K={K:.2f}")
    plt.xlabel("x")
    plt.ylabel("m_x")
    plt.show()


compare_with_fem(model, "dataset/sample_0000.npz")
compare_with_fem(model, "dataset/sample_0010.npz")
compare_with_fem(model, "dataset/sample_0020.npz")

plot_line_cut(model, "dataset/sample_0010.npz")