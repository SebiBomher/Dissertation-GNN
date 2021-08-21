
import torch
from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.attention.astgcn import ASTGCN
from torch_geometric_temporal.nn.attention.mstgcn import MSTGCN
from torch_geometric_temporal.nn.recurrent import GCLSTM
from torch.nn import ReLU,Linear,Module,Conv2d

class STConvModel(Module):
    def __init__(self, node_features,num_nodes,hidden_channels,kernel_size,K):
        super(STConvModel, self).__init__()
        self.STCONV = STConv(num_nodes = num_nodes,in_channels=node_features,hidden_channels=hidden_channels,out_channels=node_features,kernel_size=kernel_size,K=K)
        self.linear = Linear(node_features, 1)
        self.ReLU = ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.STCONV(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.STCONV(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.STCONV(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.STCONV(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.linear(x)
        return x

class CustomModel(Module):
    def __init__(self,node_features,K):
        super(CustomModel, self).__init__()
        self.GCLSTM_1 = GCLSTM(in_channels= node_features,out_channels= 64,K=K)
        self.GCLSTM_2 = GCLSTM(in_channels= 32,out_channels= 16,K=K)
        self.Conv2d_1 = Conv2d(in_channels=64,out_channels=32)
        self.Conv2d_2 = Conv2d(in_channels=16,out_channels=node_features)
        self.linear = Linear(node_features, 1)
        self.ReLU = ReLU()

    def block(self, x, edge_index, edge_weight):
        x = self.GCLSTM_1(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.Conv2d_1(x)
        x = self.ReLU(x)
        x = self.GCLSTM_2(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.Conv2d_2(x)
        return x

    def forward(self, x, edge_index, edge_weight):
        Left = self.block(x, edge_index, edge_weight)
        Right = self.block(x, edge_index, edge_weight)
        x = torch.cat((Left, Right), 0)
        x = self.ReLU(x)
        Left = self.block(x, edge_index, edge_weight)
        Right = self.block(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.linear(x)
        return x

class ASTGCNModel(torch.nn.Module):
    def __init__(self,nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input,num_of_vertices):
        super(ASTGCNModel, self).__init__()
        self.ASTGCN1 =ASTGCN(nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input,num_of_vertices)
        self.ASTGCN2 =ASTGCN(nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input,num_of_vertices)
        self.linear = torch.nn.Linear(in_channels, 1)

    def forward(self, x, edge_index):
        x = self.ASTGCN1(x, edge_index)
        x = x.relu()
        x = self.ASTGCN2(x, edge_index)
        x = x.relu()
        x = self.linear(x)
        return x

class MSTGCNModel(torch.nn.Module):
    def __init__(self,nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input):
        super(MSTGCNModel, self).__init__()
        self.MSTGCN1 = MSTGCN(nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input)
        self.MSTGCN2 = MSTGCN(nb_block,in_channels,K,nb_chev_filter,nb_time_filter,time_strides,num_for_predict,len_input)
        self.linear = torch.nn.Linear(in_channels, 1)

    def forward(self, x, edge_index):
        x = self.MSTGCN1(x, edge_index)
        x = x.relu()
        x = self.MSTGCN1(x, edge_index)
        x = x.relu()
        x = self.linear(x)
        return x

class LinearRegression():
    def __init__(self):
        self.t = 1