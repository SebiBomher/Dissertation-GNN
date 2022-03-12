import torch
from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.recurrent import GCLSTM, DCRNN
from torch_geometric.nn import GCNConv
from torch.nn import ReLU, Linear, Module, BatchNorm1d, Dropout
from sklearn.linear_model import LinearRegression


class STConvModel(Module):

    def __init__(self, node_features, num_nodes, hidden_channels, kernel_size,
                 K):
        super(STConvModel, self).__init__()

        self.STCONV = STConv(num_nodes=num_nodes,
                             in_channels=node_features,
                             hidden_channels=hidden_channels,
                             out_channels=node_features,
                             kernel_size=kernel_size,
                             K=K)
        self.linear = Linear(node_features, 1)
        self.ReLU = ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.STCONV(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.STCONV(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.STCONV(x, edge_index, edge_weight)
        x = self.ReLU(x)
        x = self.linear(x)
        return x


class DCRNNModel(Module):

    def __init__(self, node_features, hidden_channels, K):
        super(DCRNNModel, self).__init__()

        self.DCRNN = DCRNN(in_channels=node_features,
                           out_channels=hidden_channels,
                           K=K)
        self.Conv1 = GCNConv(in_channels=hidden_channels,
                             out_channels=hidden_channels)
        self.BatchNorm1 = BatchNorm1d(num_features=hidden_channels)
        self.ReLU = ReLU()
        self.Dropout = Dropout()
        self.linear = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.DCRNN(x, edge_index, edge_weight)
        x = self.Conv1(x, edge_index)
        x = self.ReLU(x)
        x = self.BatchNorm1(x)
        x = self.Dropout(x)
        x = self.linear(x)
        return x


class LSTMModel(Module):

    def __init__(self, node_features, hidden_channels, K):
        super(LSTMModel, self).__init__()

        self.GCLSTM1 = GCLSTM(in_channels=node_features,
                              out_channels=hidden_channels,
                              K=K)
        self.Conv1 = GCNConv(in_channels=hidden_channels,
                             out_channels=hidden_channels)
        self.BatchNorm1 = BatchNorm1d(num_features=hidden_channels)
        self.ReLU = ReLU()
        self.Dropout = Dropout()
        self.linear = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.GCLSTM1(x, edge_index, edge_weight)
        x = self.Conv1(x[0], edge_index)
        x = self.ReLU(x)
        x = self.BatchNorm1(x)
        x = self.Dropout(x)
        x = self.linear(x)
        return x


class LinearRegression():

    def __init__(self):
        self.model = LinearRegression()