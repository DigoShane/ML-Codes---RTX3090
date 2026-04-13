# evaluate.py

import torch
from torch_geometric.loader import DataLoader
from dataset import load_dataset
from model import BarrierGNN
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_path, dataset_path):

    dataset = load_dataset(dataset_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

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

    print("MAE:", mae)
    print("RMSE:", rmse)

    print("True mean:", true.mean())
    print("Pred mean:", preds.mean())

    # Parity plot
    plt.scatter(true, preds, alpha=0.5)
    plt.plot([true.min(), true.max()],
             [true.min(), true.max()],
             color="red")
    plt.xlabel("True Barrier (eV)")
    plt.ylabel("Predicted Barrier (eV)")
    plt.title("Parity Plot")
    plt.show()

if __name__ == "__main__":
    evaluate("models/bvse_pretrained.pt",
             "data/raw/nebBVSE122k/nebBVSE122k_test.xyz")
