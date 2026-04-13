# model.py

import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool


class BarrierGNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Edge network: maps distance -> weight matrix
        edge_nn = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * 64)  # input_dim * output_dim
        )

        self.conv1 = NNConv(2, 64, edge_nn, aggr='mean')

        edge_nn2 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64 * 128)
        )

        self.conv2 = NNConv(64, 128, edge_nn2, aggr='mean')

        self.regressor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)
        return self.regressor(x).view(-1)
