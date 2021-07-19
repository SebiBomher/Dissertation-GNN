
import torch
from torch_geometric_temporal.nn.attention.stgcn import STConv

class ST_GCN(torch.nn.Module):
    def __init__(self, node_features,num_nodes,hidden_channels,kernel_size,K):
        super(ST_GCN, self).__init__()
        self.STCONV1 = STConv(num_nodes = num_nodes,in_channels=node_features,hidden_channels=hidden_channels,out_channels=hidden_channels,kernel_size=kernel_size,K=K)
        self.STCONV2 = STConv(num_nodes = num_nodes,in_channels=node_features,hidden_channels=hidden_channels,out_channels=hidden_channels,kernel_size=kernel_size,K=K)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.STCONV1(x, edge_index, edge_weight)
        h = self.STCONV2(x, edge_index, edge_weight)
        h = self.linear(h)
        return h

