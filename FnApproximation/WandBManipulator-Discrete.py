import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# USER SETTINGS
# ============================================================

L = 1.0
a = 0.5

N_HIDDEN = 6

# locality_strength = 10 means:
# set 1 has 1.0 effect in region 1 and 0.1 effect in region 2
# set 2 has 0.1 effect in region 1 and 1.0 effect in region 2
locality_strength = 10.0
epsilon = 1.0 / locality_strength

X_MIN = 0.0
X_MAX = L
N_POINTS = 500

USE_SMOOTH_GATES = False
BETA = 50.0


def target_function(x):
    """
    Define the target function here.
    """
    return np.sin(2.0 * np.pi * x)


# ============================================================
# GATES
# ============================================================

def sharp_gates(x, a, epsilon):
    """
    Sharp localization gates.

    g1 = 1       in region 1, epsilon in region 2
    g2 = epsilon in region 1, 1       in region 2
    """

    g1 = np.ones_like(x)
    g2 = np.ones_like(x)

    region1 = x <= a
    region2 = x > a

    g1[region1] = 1.0
    g1[region2] = epsilon

    g2[region1] = epsilon
    g2[region2] = 1.0

    return g1, g2


def smooth_gates(x, a, epsilon, beta):
    """
    Smooth localization gates.

    beta controls sharpness.
    Larger beta gives a sharper transition near x = a.
    """

    s = 0.5 * (1.0 + np.tanh(beta * (x - a)))

    g1 = (1.0 - s) + epsilon * s
    g2 = epsilon * (1.0 - s) + s

    return g1, g2


# ============================================================
# ONE GATED NEURAL NETWORK
# ============================================================

def gated_neural_network(x, W, b, A, c):
    """
    One-hidden-layer gated neural network.

    x : shape (num_points,)
    W : shape (6,)
    b : shape (6,)
    A : shape (6,)
    c : scalar

    Neurons 1,2,3 belong mostly to region 1.
    Neurons 4,5,6 belong mostly to region 2.
    """

    z = W[None, :] * x[:, None] + b[None, :]
    h = np.tanh(z)

    y_set1 = h[:, 0:3] @ A[0:3]
    y_set2 = h[:, 3:6] @ A[3:6]

    if USE_SMOOTH_GATES:
        g1, g2 = smooth_gates(x, a, epsilon, BETA)
    else:
        g1, g2 = sharp_gates(x, a, epsilon)

    y = g1 * y_set1 + g2 * y_set2 + c

    return y, g1, g2, y_set1, y_set2


# ============================================================
# GUI
# ============================================================

class GatedNeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Gated Neural Network")

        self.x = np.linspace(X_MIN, X_MAX, N_POINTS)
        self.y_target = target_function(self.x)

        rng = np.random.default_rng(0)

        self.W = 0.5 * rng.normal(size=N_HIDDEN)
        self.b = np.zeros(N_HIDDEN)
        self.A = 0.5 * rng.normal(size=N_HIDDEN)
        self.c = 0.0

        self.slider_vars = {}

        self.create_layout()
        self.create_plot()
        self.create_sliders()

        self.update_plot()

    def create_layout(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas_controls = tk.Canvas(self.control_frame, width=430)
        self.scrollbar = ttk.Scrollbar(
            self.control_frame,
            orient="vertical",
            command=self.canvas_controls.yview
        )

        self.scrollable_frame = ttk.Frame(self.canvas_controls)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_controls.configure(
                scrollregion=self.canvas_controls.bbox("all")
            )
        )

        self.canvas_controls.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw"
        )

        self.canvas_controls.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 5))

        self.target_line, = self.ax.plot(
            self.x,
            self.y_target,
            linewidth=2,
            label="Target function"
        )

        self.nn_line, = self.ax.plot(
            self.x,
            np.zeros_like(self.x),
            "--",
            linewidth=2,
            label="Gated neural network"
        )

        self.interface_line = self.ax.axvline(
            a,
            linestyle=":",
            linewidth=2,
            label="interface x = a"
        )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("u(x)")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_slider(self, parent, name, initial_value, row, min_value=-5.0, max_value=5.0):
        label = ttk.Label(parent, text=name)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=2)

        var = tk.DoubleVar(value=initial_value)

        slider = ttk.Scale(
            parent,
            from_=min_value,
            to=max_value,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda value: self.update_plot()
        )
        slider.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

        value_label = ttk.Label(parent, text=f"{initial_value:.3f}", width=8)
        value_label.grid(row=row, column=2, padx=5, pady=2)

        def update_label(*args):
            value_label.config(text=f"{var.get():.3f}")

        var.trace_add("write", update_label)

        self.slider_vars[name] = var

    def create_sliders(self):
        row = 0

        title = ttk.Label(
            self.scrollable_frame,
            text="Interactive Gated Neural Network Parameters",
            font=("Arial", 12, "bold")
        )
        title.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        info = ttk.Label(
            self.scrollable_frame,
            text=(
                f"Region 1: [0, {a}]\n"
                f"Region 2: ({a}, {L}]\n"
                f"Locality strength = {locality_strength}\n"
                f"Weak influence = {epsilon}"
            )
        )
        info.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=10)
        row += 1

        # ----------------------------------------------------
        # Set 1 parameters
        # ----------------------------------------------------
        section = ttk.Label(
            self.scrollable_frame,
            text="Set 1: neurons 1, 2, 3 mostly affect region 1",
            font=("Arial", 10, "bold")
        )
        section.grid(row=row, column=0, columnspan=3, sticky="w", pady=8)
        row += 1

        for i in range(3):
            self.create_slider(
                self.scrollable_frame,
                f"W[{i+1}]",
                self.W[i],
                row
            )
            row += 1

            self.create_slider(
                self.scrollable_frame,
                f"b[{i+1}]",
                self.b[i],
                row
            )
            row += 1

            self.create_slider(
                self.scrollable_frame,
                f"A[{i+1}]",
                self.A[i],
                row
            )
            row += 1

        # ----------------------------------------------------
        # Set 2 parameters
        # ----------------------------------------------------
        section = ttk.Label(
            self.scrollable_frame,
            text="Set 2: neurons 4, 5, 6 mostly affect region 2",
            font=("Arial", 10, "bold")
        )
        section.grid(row=row, column=0, columnspan=3, sticky="w", pady=8)
        row += 1

        for i in range(3, 6):
            self.create_slider(
                self.scrollable_frame,
                f"W[{i+1}]",
                self.W[i],
                row
            )
            row += 1

            self.create_slider(
                self.scrollable_frame,
                f"b[{i+1}]",
                self.b[i],
                row
            )
            row += 1

            self.create_slider(
                self.scrollable_frame,
                f"A[{i+1}]",
                self.A[i],
                row
            )
            row += 1

        # ----------------------------------------------------
        # Output bias
        # ----------------------------------------------------
        section = ttk.Label(
            self.scrollable_frame,
            text="Global output bias",
            font=("Arial", 10, "bold")
        )
        section.grid(row=row, column=0, columnspan=3, sticky="w", pady=8)
        row += 1

        self.create_slider(
            self.scrollable_frame,
            "c",
            self.c,
            row
        )
        row += 1

        # ----------------------------------------------------
        # Reset button
        # ----------------------------------------------------
        reset_button = ttk.Button(
            self.scrollable_frame,
            text="Reset all parameters to zero",
            command=self.reset_parameters
        )
        reset_button.grid(row=row, column=0, columnspan=3, pady=15)

        self.scrollable_frame.columnconfigure(1, weight=1)

    def read_parameters_from_sliders(self):
        for i in range(N_HIDDEN):
            self.W[i] = self.slider_vars[f"W[{i+1}]"].get()
            self.b[i] = self.slider_vars[f"b[{i+1}]"].get()
            self.A[i] = self.slider_vars[f"A[{i+1}]"].get()

        self.c = self.slider_vars["c"].get()

    def update_plot(self):
        self.read_parameters_from_sliders()

        y_nn, g1, g2, y_set1, y_set2 = gated_neural_network(
            self.x,
            self.W,
            self.b,
            self.A,
            self.c
        )

        self.nn_line.set_ydata(y_nn)

        y_min = min(np.min(self.y_target), np.min(y_nn))
        y_max = max(np.max(self.y_target), np.max(y_nn))

        padding = 0.1 * max(1.0, y_max - y_min)
        self.ax.set_ylim(y_min - padding, y_max + padding)

        mse = np.mean((y_nn - self.y_target) ** 2)

        self.ax.set_title(
            f"MSE = {mse:.6f}, locality strength = {locality_strength}"
        )

        self.canvas.draw_idle()

    def reset_parameters(self):
        for name, var in self.slider_vars.items():
            var.set(0.0)

        self.update_plot()


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GatedNeuralNetworkGUI(root)

    def on_closing():
        try:
            plt.close(app.fig)
        except Exception:
            pass

        try:
            root.quit()
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()