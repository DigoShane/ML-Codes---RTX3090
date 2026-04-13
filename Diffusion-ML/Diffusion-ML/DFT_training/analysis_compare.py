import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
from torch_geometric.loader import DataLoader
from model import BarrierGNN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test set
dataset = torch.load("../data/processed/dft_test.pt")
loader = DataLoader(dataset, batch_size=8, shuffle=False)

def evaluate_model(model_path):
    model = BarrierGNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds, true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.append(pred.cpu().numpy())
            true.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds)
    true = np.concatenate(true)

    mae = np.mean(np.abs(preds - true))
    rmse = np.sqrt(np.mean((preds - true)**2))

    return true, preds, mae, rmse


true_s, pred_s, mae_s, rmse_s = evaluate_model("../models/dft_scratch.pt")
true_f, pred_f, mae_f, rmse_f = evaluate_model("../models/dft_finetuned.pt")

# Save metrics
df = pd.DataFrame({
    "Model": ["Scratch", "Fine-Tuned"],
    "MAE": [mae_s, mae_f],
    "RMSE": [rmse_s, rmse_f]
})
df.to_csv("results/metrics_comparison.csv", index=False)

# ---- Parity Plots ----
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(true_s, pred_s, alpha=0.6)
plt.plot([true_s.min(), true_s.max()],
         [true_s.min(), true_s.max()], color="red")
plt.title("Scratch Model")
plt.xlabel("True DFT Barrier (eV)")
plt.ylabel("Predicted (eV)")

plt.subplot(1,2,2)
plt.scatter(true_f, pred_f, alpha=0.6)
plt.plot([true_f.min(), true_f.max()],
         [true_f.min(), true_f.max()], color="red")
plt.title("Fine-Tuned Model")
plt.xlabel("True DFT Barrier (eV)")

plt.tight_layout()
plt.savefig("results/parity_comparison.png")
plt.show()

# ---- Error vs Barrier Bin ----
bins = np.linspace(true_s.min(), true_s.max(), 6)

def bin_error(true, pred):
    errors = []
    centers = []
    for i in range(len(bins)-1):
        mask = (true >= bins[i]) & (true < bins[i+1])
        if np.sum(mask) > 0:
            errors.append(np.mean(np.abs(pred[mask] - true[mask])))
            centers.append((bins[i] + bins[i+1]) / 2)
    return centers, errors

cent_s, err_s = bin_error(true_s, pred_s)
cent_f, err_f = bin_error(true_f, pred_f)

plt.figure()
plt.plot(cent_s, err_s, marker='o', label="Scratch")
plt.plot(cent_f, err_f, marker='o', label="Fine-Tuned")
plt.xlabel("Barrier (eV)")
plt.ylabel("MAE per Bin")
plt.legend()
plt.savefig("results/error_vs_barrier.png")
plt.show()
