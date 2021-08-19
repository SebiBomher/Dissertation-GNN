
import torch
from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.attention.astgcn import ASTGCN
from torch_geometric_temporal.nn.attention.mstgcn import MSTGCN

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
    def __init__(self,nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input,num_of_vertices):
        super(ASTGCN, self).__init__()
        self.ASTGCN =ASTGCN(nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input,num_of_vertices)

    def forward(self, x, edge_index):
        x = self.ASTGCN(x, edge_index)
        x = x.relu()
        return x

class MSTGCNModel(torch.nn.Module):
    def __init__(self,nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input):
        super(MSTGCN, self).__init__()
        self.MSTGCN = MSTGCN(nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input)

    def forward(self, x, edge_index, edge_weight):
        x = self.MSTGCN(x, edge_index)
        return x

class LinearRegression():
    def __init__(self):
        self.t = 1