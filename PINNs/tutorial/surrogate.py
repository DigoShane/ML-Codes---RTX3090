import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 1. True solution
# =========================================================
def true_solution(x, D):
    return np.sin(np.pi * x) / D

# =========================================================
# 2. Generate training data
# =========================================================
N = 5000

D_data = np.random.uniform(0.5, 2.0, (N, 1))
x_data = np.random.uniform(0.0, 1.0, (N, 1))

u_data = true_solution(x_data, D_data)

X = np.hstack([D_data, x_data])

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(u_data, dtype=torch.float32).to(device)

# =========================================================
# 3. Define surrogate model (MLP)
# =========================================================
class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, D, x):
        inp = torch.cat([D, x], dim=1)
        return self.net(inp)

model = Surrogate().to(device)

# =========================================================
# 4. Train surrogate
# =========================================================
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 2000

for epoch in range(epochs):
    optimizer.zero_grad()

    D_batch = X[:, 0:1]
    x_batch = X[:, 1:2]

    pred = model(D_batch, x_batch)

    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

# =========================================================
# 5. Validate surrogate
# =========================================================
x_test = np.linspace(0, 1, 200)
D_test = np.ones_like(x_test) * 1.0

x_test_t = torch.tensor(x_test[:, None], dtype=torch.float32).to(device)
D_test_t = torch.tensor(D_test[:, None], dtype=torch.float32).to(device)

with torch.no_grad():
    u_pred = model(D_test_t, x_test_t).cpu().numpy()

u_true = true_solution(x_test, D_test)

plt.figure()
plt.plot(x_test, u_true, label="True")
plt.plot(x_test, u_pred, "--", label="Surrogate")
plt.legend()
plt.title("Surrogate vs True")
plt.show()

# =========================================================
# 6. Optimization over D
# =========================================================

# Discretize domain for inner max
x_grid = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)

# Initialize D as trainable parameter
D_opt = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True, device=device)

optimizer_D = optim.Adam([D_opt], lr=0.05)

history = []

for step in range(200):
    optimizer_D.zero_grad()

    D_expand = D_opt.expand_as(x_grid)

    u_vals = model(D_expand, x_grid)

    # max over x
    max_u = torch.max(u_vals)

    loss = max_u
    loss.backward()

    optimizer_D.step()

    # enforce bounds
    with torch.no_grad():
        D_opt.clamp_(0.5, 2.0)

    history.append(D_opt.item())

    if step % 20 == 0:
        print(f"Step {step}: D = {D_opt.item():.4f}, max u = {max_u.item():.4f}")

# =========================================================
# 7. Plot convergence
# =========================================================
plt.figure()
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("D")
plt.title("Optimization Convergence")
plt.show()

print("\nFinal optimized D:", D_opt.item())
print("True optimal D: 2.0")