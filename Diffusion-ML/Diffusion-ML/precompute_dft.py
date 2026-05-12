# precompute_dft.py

import torch
from dataset import load_dataset
import os

os.makedirs("data/processed", exist_ok=True)

#converts to a graph. each node and edge is assigned a feature. 
dataset = load_dataset("data/raw/nebDFT2k/nebDFT2k_centroids.xyz")

torch.save(dataset, "data/processed/dft_centroids.pt")

print("DFT graphs saved.")
