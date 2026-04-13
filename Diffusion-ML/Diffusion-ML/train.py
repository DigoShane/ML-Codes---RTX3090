import torch
from torch_geometric.loader import DataLoader
from dataset import load_dataset
from model import BarrierGNN
from sklearn.metrics import mean_absolute_error
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("data/raw/nebBVSE122k/nebBVSE122k_train.xyz")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = BarrierGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(20):
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
        if epoch == 0:
           print("Sample predictions:", pred[:5])
           print("Sample targets:", batch.y[:5])

    print("Epoch:", epoch, "Loss:", total_loss/len(loader))

torch.save(model.state_dict(), "models/bvse_pretrained.pt")
