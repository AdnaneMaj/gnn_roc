import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self,edge_index, embedding_dim=64, num_layers=3):
        super().__init__()
        self.users_index = edge_index[0].unique()
        self.items_index = edge_index[1].unique()

        # Initial embeddings
        self.E0 = nn.Embedding(self.users_index.size(0)+self.items_index.size(0), embedding_dim)

        nn.init.xavier_uniform_(self.E0.weight)

        # LG Convolution layers
        self.convs = nn.ModuleList([
            LGConv() for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index):

        E0 = self.E0.weight

        # Store embeddings from each layer
        layer_embeddings = [E0]

        # Perform message passing
        x = E0
        for conv in self.convs:
            x = conv(x, edge_index) 
            layer_embeddings.append(x)

        # Aggregate embeddings
        E = torch.mean(torch.stack(layer_embeddings), dim=0)
        
        return E0,E