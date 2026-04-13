# dft_scratch_train.py

import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
from torch_geometric.loader import DataLoader
from model import BarrierGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torch.load("../data/processed/dft_train.pt")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = BarrierGNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(50):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss/len(loader))

torch.save(model.state_dict(), "../models/dft_scratch.pt")
