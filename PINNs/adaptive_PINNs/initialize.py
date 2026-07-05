"""
initialize.py

Function-based initialization utilities for the PINN code.

Important limitation:
A finite-width Tanh network generally cannot be assigned exact closed-form
weights for an arbitrary function such as sin(omega*pi*x). Therefore this file
implements a practical function-matching initialization: it first trains the
network in a supervised way to approximate the target function on sample points,
then optionally adds Gaussian noise to the trained parameters.
"""

import os
import torch
import matplotlib.pyplot as plt


def add_gaussian_noise(model, std):
    """
    Add independent Gaussian noise N(0, std^2) to every trainable parameter.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network whose parameters are modified in-place.
    std : float
        Standard deviation of Gaussian noise. If std <= 0, no noise is added.
    """
    std = float(std)
    if std <= 0.0:
        return

    with torch.no_grad():
        for param in model.parameters():
            param.add_(std * torch.randn_like(param))

def save_initialization_fit_plot(
    model,
    target_function,
    x_min,
    x_max,
    device,
    plot_path,
    plot_title="Initialization fit before noise",
    num_plot_points=1000,
):
    """
    Plot the neural network approximation after function-based initialization
    against the target function.

    This should be called before Gaussian noise is added.
    """

    if plot_path is None:
        return

    model.eval()

    x_plot = torch.linspace(float(x_min), float(x_max), int(num_plot_points), device=device)
    x_plot = x_plot.view(-1, 1)

    with torch.no_grad():
        y_target = target_function(x_plot)

        if not torch.is_tensor(y_target):
            y_target = torch.tensor(y_target, dtype=x_plot.dtype, device=device)

        y_target = y_target.to(device=device, dtype=x_plot.dtype)

        if y_target.ndim == 1:
            y_target = y_target.view(-1, 1)

        y_nn = model(x_plot)

    x_np = x_plot.detach().cpu().numpy()
    y_target_np = y_target.detach().cpu().numpy()
    y_nn_np = y_nn.detach().cpu().numpy()

    plot_dir = os.path.dirname(plot_path)
    if plot_dir != "":
        os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(x_np, y_target_np, label="Target / exact solution")
    plt.plot(x_np, y_nn_np, "--", label="NN after initialization fit")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    model.train()

def initialize_from_function( model, target_function, x_min=0.0, x_max=1.0, num_points=2000, fit_epochs=5000, lr=1.0e-3, noise_std=0.0, device=None, print_every=1000, plot_path=None, plot_title="Initialization fit before noise"):
    """
    Initialize a neural network by fitting it to a target function, then adding
    Gaussian noise.

    This modifies `model` in-place.

    Parameters
    ----------
    model : torch.nn.Module
        The PINN model to initialize.
    target_function : callable
        Function of x returning target values. It should accept a tensor x with
        shape (N, 1) and return a tensor compatible with shape (N, 1).
    x_min, x_max : float
        Interval on which to fit the target function.
    num_points : int
        Number of sample points used for the supervised fit.
    fit_epochs : int
        Number of supervised fitting epochs before PINN training starts.
    lr : float
        Learning rate used for the initialization fit.
    noise_std : float
        Standard deviation of Gaussian noise added after fitting.
    device : torch.device or str or None
        Device on which to run initialization. If None, uses the model's device.
    print_every : int or None
        Print initialization progress every this many epochs. If None or <= 0,
        suppress progress printing.

    Returns
    -------
    info : dict
        Dictionary containing the final pre-noise fit loss and settings.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.to(device)
    model.train()

    x_fit = torch.linspace(float(x_min), float(x_max), int(num_points), device=device)
    x_fit = x_fit.view(-1, 1)

    with torch.no_grad():
        y_fit = target_function(x_fit)
        if not torch.is_tensor(y_fit):
            y_fit = torch.tensor(y_fit, dtype=x_fit.dtype, device=device)
        y_fit = y_fit.to(device=device, dtype=x_fit.dtype)
        if y_fit.ndim == 1:
            y_fit = y_fit.view(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    final_loss = None
    for epoch in range(int(fit_epochs)):
        optimizer.zero_grad()
        y_pred = model(x_fit)
        loss = torch.mean((y_pred - y_fit) ** 2)
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())

        if print_every is not None and int(print_every) > 0:
            if epoch % int(print_every) == 0 or epoch == int(fit_epochs) - 1:
                print(f"Initialization fit | epoch {epoch:6d} | MSE = {final_loss:.4e}")

    # Save plot after the function-fitting initialization, but before noise.
    save_initialization_fit_plot(
    	model=model,
	target_function=target_function,
	x_min=x_min,
	x_max=x_max,	
	device=device,
	plot_path=plot_path,
	plot_title=plot_title)

    # Now add Gaussian noise to the fitted weights.
    add_gaussian_noise(model, noise_std)

    return {
    "final_fit_mse_before_noise": final_loss,
    "noise_std": float(noise_std),
    "num_points": int(num_points),
    "fit_epochs": int(fit_epochs),
    "lr": float(lr),
    "x_min": float(x_min),
    "x_max": float(x_max),
    "plot_path": plot_path,
    }