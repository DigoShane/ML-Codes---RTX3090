from fenics import *
import numpy as np
import os

# ----------------------------
# Output folder
# ----------------------------
os.makedirs("dataset", exist_ok=True)

# ----------------------------
# Mesh and space
# ----------------------------
mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, "CG", 1)

# ----------------------------
# Boundary conditions
# ----------------------------
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1)

bc_left = DirichletBC(V, Constant(0.0), Left())
bc_right = DirichletBC(V, Constant(np.pi), Right())
bcs = [bc_left, bc_right]

# ----------------------------
# Functions
# ----------------------------
theta = Function(V)
v = TestFunction(V)

A = Constant(1.0)

# ----------------------------
# Sampling K
# ----------------------------
K_values = np.concatenate([
    np.linspace(0.1, 1.0, 15),
    np.linspace(1.0, 5.0, 25)
])

# ----------------------------
# Coordinates (fixed mesh)
# ----------------------------
coords = mesh.coordinates()

# ----------------------------
# Loop over K
# ----------------------------
for i, K_val in enumerate(K_values):
    print(f"Running K = {K_val:.3f}")

    K = Constant(K_val)

    # Weak form
    F = A*dot(grad(theta), grad(v))*dx + (K/2)*sin(2*theta)*v*dx

    solve(F == 0, theta, bcs)

    # Extract values
    theta_vals = theta.compute_vertex_values(mesh)

    # Compute m
    m_x = np.cos(theta_vals)
    m_y = np.sin(theta_vals)
    m_vals = np.vstack([m_x, m_y]).T

    # Save
    np.savez(
        f"dataset/sample_{i:04d}.npz",
        K=K_val,
        coords=coords,
        theta=theta_vals,
        m=m_vals
    )