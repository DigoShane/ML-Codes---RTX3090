import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 7  # try k = 1 and k = 10
N_f = 100000  # collocation points

def f(x):
    return (k*np.pi)**2 * torch.sin(k*np.pi*x)

def exact_solution(x):
    return torch.sin(k*np.pi*x)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PINN().to(device)

def pde_residual(x):
    x.requires_grad_(True)

    u = model(x)

    # First derivative
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]

    # Second derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                              create_graph=True)[0]

    return -u_xx - f(x)

    x_f = torch.linspace(0, 1, N_f).view(-1,1).to(device)

# Boundary points
x_b = torch.tensor([[0.0],[1.0]], device=device)
u_b = torch.tensor([[0.0],[0.0]], device=device)

def loss_fn():
    # PDE loss
    res = pde_residual(x_f)
    loss_pde = torch.mean(res**2)

    # Boundary loss
    u_pred = model(x_b)
    loss_bc = torch.mean((u_pred - u_b)**2)

    return loss_pde + loss_bc

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4e}")

x_test = torch.linspace(0,1,200).view(-1,1).to(device)

u_pred = model(x_test).detach().cpu().numpy()
u_exact = exact_solution(x_test).cpu().numpy()

plt.figure(figsize=(6,4))
plt.plot(x_test.cpu(), u_exact, label="Exact")
plt.plot(x_test.cpu(), u_pred, '--', label="PINN")
plt.title(f"k = {k}")
plt.legend()
plt.show()

error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
print(f"Relative L2 Error: {error:.2e}")

plt.figure()
plt.plot(loss_history)
plt.yscale('log')
plt.title("Training Loss")
plt.show()