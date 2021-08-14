
import torch
from torch_geometric_temporal.nn.attention.stgcn import STConv

class STConvModel(torch.nn.Module):
    def __init__(self, node_features,num_nodes,hidden_channels,kernel_size,K):
        super(STConvModel, self).__init__()
        self.STCONV1 = STConv(num_nodes = num_nodes,in_channels=node_features,hidden_channels=hidden_channels,out_channels=node_features,kernel_size=kernel_size,K=K)
        self.STCONV2 = STConv(num_nodes = num_nodes,in_channels=node_features,hidden_channels=hidden_channels,out_channels=node_features,kernel_size=kernel_size,K=K)
        self.STCONV3 = STConv(num_nodes = num_nodes,in_channels=node_features,hidden_channels=hidden_channels,out_channels=node_features,kernel_size=kernel_size,K=K)
        self.STCONV4 = STConv(num_nodes = num_nodes,in_channels=node_features,hidden_channels=hidden_channels,out_channels=node_features,kernel_size=kernel_size,K=K)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.STCONV1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.STCONV2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.STCONV3(x, edge_index, edge_weight)
        x = x.relu()
        x = self.STCONV4(x, edge_index, edge_weight)
        x = x.relu()
        x = self.linear(x)
        return x


