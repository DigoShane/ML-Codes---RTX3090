# evaluate_dft_finetune.py

import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
from torch_geometric.loader import DataLoader
from model import BarrierGNN
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_path, dataset_path):

    # Load precomputed DFT graphs
    dataset = torch.load("../data/processed/dft_test.pt")
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

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

    print("DFT Fine-Tuned Model Evaluation")
    print("--------------------------------")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("True mean:", true.mean())
    print("Pred mean:", preds.mean())

    # Parity plot
    plt.figure()
    plt.scatter(true, preds, alpha=0.6)
    plt.plot([true.min(), true.max()],
             [true.min(), true.max()],
             color="red")
    plt.xlabel("True DFT Barrier (eV)")
    plt.ylabel("Predicted DFT Barrier (eV)")
    plt.title("DFT Fine-Tuned Model Parity Plot")
    plt.show()


if __name__ == "__main__":
    evaluate("../models/dft_finetuned.pt",
             "../data/raw/nebDFT2k/nebDFT2k_centroids.xyz")
