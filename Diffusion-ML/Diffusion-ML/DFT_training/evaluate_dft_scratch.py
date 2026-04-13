# evaluate_dft_scratch.py

import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
from torch_geometric.loader import DataLoader
from model import BarrierGNN
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("../data/processed/dft_test.pt")
loader = DataLoader(dataset, batch_size=8, shuffle=False)

model = BarrierGNN().to(device)
model.load_state_dict(torch.load("../models/dft_scratch.pt"))
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

print("DFT Scratch Model")
print("------------------")
print("MAE:", mae)
print("RMSE:", rmse)
