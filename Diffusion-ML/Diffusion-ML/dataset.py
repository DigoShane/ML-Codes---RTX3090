# dataset.py

import torch
from torch_geometric.data import Data
from ase.io import read
import numpy as np
from scipy.spatial import cKDTree


def build_graph(atoms, cutoff=5.0):
    pos = atoms.get_positions()
    Z = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()

    # Find centroid atom
    centroid_idx = symbols.index("X")
    centroid_pos = pos[centroid_idx]

    # Node features:
    # [atomic number, distance to centroid]
    dist_to_centroid = np.linalg.norm(pos - centroid_pos, axis=1)
    node_features = np.stack([Z, dist_to_centroid], axis=1)

    # Build neighbor graph
    tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff)

    edge_index = []
    edge_attr = []

    for i, j in pairs:
        dist = np.linalg.norm(pos[i] - pos[j])
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append([dist])
        edge_attr.append([dist])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([atoms.info["em"]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def load_dataset(path):
    atoms_list = read(path, index=":")
    dataset = [build_graph(atoms) for atoms in atoms_list]
    return dataset
