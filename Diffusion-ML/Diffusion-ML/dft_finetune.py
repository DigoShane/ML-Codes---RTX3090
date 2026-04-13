# dft_finetune.py

import torch
from torch_geometric.loader import DataLoader
from model import BarrierGNN
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DFT dataset
dataset = torch.load("data/processed/dft_centroids.pt")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load pretrained model
model = BarrierGNN().to(device)
model.load_state_dict(torch.load("models/bvse_pretrained.pt"))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(40):
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

torch.save(model.state_dict(), "models/dft_finetuned.pt")
