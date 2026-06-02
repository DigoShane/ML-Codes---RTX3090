import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# USER SETTINGS
# ============================================================

N_INPUTS = 1          # Change this to any N >= 1
N_HIDDEN = 6          # Fixed as requested
X_MIN = -5.0
X_MAX = 5.0
N_POINTS = 400


def target_function(x):
    """
    Specify the function you want to approximate here.

    x is a NumPy array.
    """
    return np.sin(x)


# ============================================================
# ACTIVATION FUNCTION
# ============================================================

def activation(z):
    return np.tanh(z)


# ============================================================
# NEURAL NETWORK
# ============================================================

def neural_network(X, W1, b1, W2, b2):
    """
    One-hidden-layer neural network.

    X  : shape (num_points, N_INPUTS)
    W1 : shape (N_HIDDEN, N_INPUTS)
    b1 : shape (N_HIDDEN,)
    W2 : shape (N_HIDDEN,)
    b2 : scalar

    Output:
        y : shape (num_points,)
    """

    Z1 = X @ W1.T + b1
    A1 = activation(Z1)

    y = A1 @ W2 + b2

    return y


# ============================================================
# GUI APPLICATION
# ============================================================

class OneLayerNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive One-Hidden-Layer Neural Network")

        # ----------------------------
        # Input grid for plotting
        # ----------------------------
        self.x = np.linspace(X_MIN, X_MAX, N_POINTS)
        self.y_target = target_function(self.x)

        # Values of remaining input coordinates if N_INPUTS > 1
        self.fixed_inputs = np.zeros(N_INPUTS)

        # ----------------------------
        # Initial network parameters
        # ----------------------------
        rng = np.random.default_rng(0)

        self.W1 = 0.5 * rng.normal(size=(N_HIDDEN, N_INPUTS))
        self.b1 = np.zeros(N_HIDDEN)

        self.W2 = 0.5 * rng.normal(size=N_HIDDEN)
        self.b2 = 0.0

        # ----------------------------
        # Main layout
        # ----------------------------
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # ----------------------------
        # Matplotlib figure
        # ----------------------------
        self.fig, self.ax = plt.subplots(figsize=(7, 5))

        self.target_line, = self.ax.plot(
            self.x,
            self.y_target,
            linewidth=2,
            label="Target function"
        )

        self.nn_line, = self.ax.plot(
            self.x,
            np.zeros_like(self.x),
            linewidth=2,
            linestyle="--",
            label="Network output"
        )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("output")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ----------------------------
        # Scrollable controls
        # ----------------------------
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

        # ----------------------------
        # Slider variables
        # ----------------------------
        self.slider_vars = {}

        self.create_sliders()

        # Initial plot
        self.update_plot()

    def create_slider(self, parent, name, initial_value, row):
        """
        Create a labeled slider.
        """

        label = ttk.Label(parent, text=name)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=2)

        var = tk.DoubleVar(value=initial_value)

        slider = ttk.Scale(
            parent,
            from_=-5.0,
            to=5.0,
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
        """
        Create sliders for:
        - fixed input values if N_INPUTS > 1
        - input-to-hidden weights W1
        - hidden biases b1
        - hidden-to-output weights W2
        - output bias b2
        """

        row = 0

        title = ttk.Label(
            self.scrollable_frame,
            text="Interactive Network Parameters",
            font=("Arial", 12, "bold")
        )
        title.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # ----------------------------------------------------
        # Fixed inputs for dimensions 2,...,N
        # ----------------------------------------------------
        if N_INPUTS > 1:
            section = ttk.Label(
                self.scrollable_frame,
                text="Fixed input coordinates",
                font=("Arial", 10, "bold")
            )
            section.grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
            row += 1

            for j in range(1, N_INPUTS):
                name = f"x{j+1}_fixed"
                self.create_slider(
                    self.scrollable_frame,
                    name,
                    self.fixed_inputs[j],
                    row
                )
                row += 1

        # ----------------------------------------------------
        # W1 sliders
        # ----------------------------------------------------
        section = ttk.Label(
            self.scrollable_frame,
            text="Input-to-hidden weights W1",
            font=("Arial", 10, "bold")
        )
        section.grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        for i in range(N_HIDDEN):
            for j in range(N_INPUTS):
                name = f"W1[{i+1},{j+1}]"
                self.create_slider(
                    self.scrollable_frame,
                    name,
                    self.W1[i, j],
                    row
                )
                row += 1

        # ----------------------------------------------------
        # b1 sliders
        # ----------------------------------------------------
        section = ttk.Label(
            self.scrollable_frame,
            text="Hidden-layer biases b1",
            font=("Arial", 10, "bold")
        )
        section.grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        for i in range(N_HIDDEN):
            name = f"b1[{i+1}]"
            self.create_slider(
                self.scrollable_frame,
                name,
                self.b1[i],
                row
            )
            row += 1

        # ----------------------------------------------------
        # W2 sliders
        # ----------------------------------------------------
        section = ttk.Label(
            self.scrollable_frame,
            text="Hidden-to-output weights W2",
            font=("Arial", 10, "bold")
        )
        section.grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        for i in range(N_HIDDEN):
            name = f"W2[{i+1}]"
            self.create_slider(
                self.scrollable_frame,
                name,
                self.W2[i],
                row
            )
            row += 1

        # ----------------------------------------------------
        # b2 slider
        # ----------------------------------------------------
        section = ttk.Label(
            self.scrollable_frame,
            text="Output bias b2",
            font=("Arial", 10, "bold")
        )
        section.grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        self.create_slider(
            self.scrollable_frame,
            "b2",
            self.b2,
            row
        )
        row += 1

        # ----------------------------------------------------
        # Reset button
        # ----------------------------------------------------
        reset_button = ttk.Button(
            self.scrollable_frame,
            text="Reset all parameters",
            command=self.reset_parameters
        )
        reset_button.grid(row=row, column=0, columnspan=3, pady=15)

        self.scrollable_frame.columnconfigure(1, weight=1)

    def read_parameters_from_sliders(self):
        """
        Read all current slider values and update W1, b1, W2, b2.
        """

        # Fixed coordinates for x2, ..., xN
        if N_INPUTS > 1:
            for j in range(1, N_INPUTS):
                name = f"x{j+1}_fixed"
                self.fixed_inputs[j] = self.slider_vars[name].get()

        # W1
        for i in range(N_HIDDEN):
            for j in range(N_INPUTS):
                name = f"W1[{i+1},{j+1}]"
                self.W1[i, j] = self.slider_vars[name].get()

        # b1
        for i in range(N_HIDDEN):
            name = f"b1[{i+1}]"
            self.b1[i] = self.slider_vars[name].get()

        # W2
        for i in range(N_HIDDEN):
            name = f"W2[{i+1}]"
            self.W2[i] = self.slider_vars[name].get()

        # b2
        self.b2 = self.slider_vars["b2"].get()

    def build_input_array(self):
        """
        Build input array X of shape (N_POINTS, N_INPUTS).

        The first coordinate varies with x.
        The remaining coordinates are fixed constants.
        """

        X = np.zeros((N_POINTS, N_INPUTS))

        X[:, 0] = self.x

        if N_INPUTS > 1:
            for j in range(1, N_INPUTS):
                X[:, j] = self.fixed_inputs[j]

        return X

    def update_plot(self):
        """
        Recompute the neural network output and update the plot.
        """

        self.read_parameters_from_sliders()

        X = self.build_input_array()

        y_nn = neural_network(
            X,
            self.W1,
            self.b1,
            self.W2,
            self.b2
        )

        self.nn_line.set_ydata(y_nn)

        y_min = min(np.min(self.y_target), np.min(y_nn))
        y_max = max(np.max(self.y_target), np.max(y_nn))

        padding = 0.1 * max(1.0, y_max - y_min)

        self.ax.set_ylim(y_min - padding, y_max + padding)

        mse = np.mean((y_nn - self.y_target) ** 2)
        self.ax.set_title(f"Mean squared error = {mse:.5f}")

        self.canvas.draw_idle()

    def reset_parameters(self):
        """
        Reset all parameters to zero.
        """

        for name, var in self.slider_vars.items():
            var.set(0.0)

        self.update_plot()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = OneLayerNetworkGUI(root)
    root.mainloop()