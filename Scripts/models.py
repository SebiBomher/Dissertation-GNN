
import torch
from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.attention.astgcn import ASTGCN
from torch_geometric_temporal.nn.attention.mstgcn import MSTGCN
from torch_geometric_temporal.nn.attention.gman import GMAN

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


class ASTGCNModel(torch.nn.Module):
    def __init__(self):
        super(ASTGCN, self).__init__()

    def forward(self, x, edge_index, edge_weight):
        return x

class MSTGCNModel(torch.nn.Module):
    def __init__(self):
        super(MSTGCN, self).__init__()
        self.STCONV1 = ASTGCN()


    def forward(self, x, edge_index, edge_weight):
        return x

class GMANModel(torch.nn.Module):
    def __init__(self):
        super(GMAN, self).__init__()

    def forward(self, x, edge_index, edge_weight):
        return x
