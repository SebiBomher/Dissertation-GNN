
from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.recurrent import GCLSTM
from torch_geometric.nn import GCNConv
from torch.nn import ReLU,Linear,Module,MaxPool1d,BatchNorm1d,Dropout

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
        self.GCLSTM1 = GCLSTM(in_channels= node_features,out_channels= 128,K=K)
        self.Conv1 = GCNConv(in_channels=128,out_channels=128)
        self.GCLSTM2 = GCLSTM(in_channels= 64,out_channels= 64,K=K)
        self.Conv2 = GCNConv(in_channels=64,out_channels=64)
        self.GCLSTM3 = GCLSTM(in_channels= 32,out_channels= 32,K=K)
        self.Conv3 = GCNConv(in_channels=32,out_channels=32)
        self.GCLSTM4 = GCLSTM(in_channels= 16,out_channels= 16,K=K)
        self.Conv4 = GCNConv(in_channels=16,out_channels=16)
        self.linear = Linear(8, 1)
        self.MaxPool = MaxPool1d(kernel_size=2)
        self.BatchNorm = BatchNorm1d(num_features=8)
        self.Dropout = Dropout()
        self.ReLU = ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.GCLSTM1(x, edge_index, edge_weight)
        x = self.Conv1(x[0], edge_index)
        x = self.ReLU(x)
        x = self.BatchNorm(x)
        x = self.Dropout(x)
        x = self.MaxPool(x)
        x = self.GCLSTM2(x, edge_index, edge_weight)
        x = self.Conv2(x[0], edge_index)
        x = self.ReLU(x)
        x = self.BatchNorm(x)
        x = self.Dropout(x)
        x = self.MaxPool(x)
        x = self.GCLSTM3(x, edge_index, edge_weight)
        x = self.Conv3(x[0], edge_index)
        x = self.ReLU(x)
        x = self.BatchNorm(x)
        x = self.Dropout(x)
        x = self.MaxPool(x)
        x = self.GCLSTM4(x, edge_index, edge_weight)
        x = self.Conv4(x[0], edge_index)
        x = self.ReLU(x)
        x = self.BatchNorm(x)
        x = self.Dropout(x)
        x = self.MaxPool(x)
        x = self.linear(x)
        return x

class LinearRegression():
    def __init__(self):
        self.t = 1