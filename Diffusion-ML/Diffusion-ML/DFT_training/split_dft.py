# split_dft.py

import torch
import numpy as np
import os

os.makedirs("../data/processed", exist_ok=True)

dataset = torch.load("../data/processed/dft_centroids.pt", weights_only=False)

np.random.seed(42)

indices = np.random.permutation(len(dataset))

split = int(0.8 * len(dataset))

train_idx = indices[:split]
test_idx = indices[split:]

train_set = [dataset[i] for i in train_idx]
test_set = [dataset[i] for i in test_idx]

torch.save(train_set, "../data/processed/dft_train.pt")
torch.save(test_set, "../data/processed/dft_test.pt")

print("Train size:", len(train_set))
print("Test size:", len(test_set))
