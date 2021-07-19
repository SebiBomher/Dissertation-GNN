
import torch
from torch_geometric_temporal.nn.attention.stgcn import STConv
from tqdm import tqdm
from matplotlib import pyplot as plt

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

def train(model,dataset,optimizer,nb_epoch,criterion,plot):
    model.train()
    if plot:
        loss_values = []
    for epoch in tqdm(range(nb_epoch)):
        loss = 0
        for time, snapshot in enumerate(dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss += criterion(y_hat,snapshot.y)
        loss = loss / (time+1)
        if plot:
            loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if plot:
        plt.plot(loss_values)
        plt.show()
        
def test(model,dataset,criterion,plot):
    model.eval()
    loss = 0
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        last_y = snapshot.y
        loss += criterion(y_hat,snapshot.y)
    loss = loss / (time+1)
    loss = loss.item()
    print(y_hat)
    print(last_y)
    print("{0}: {1}".format(criterion,loss))

def validate(model,dataset,optimizer,nb_epoch,criterion):
    # TODO
    return True

def train_and_test(model,train_dataset,test_dataset,optimizer,nb_epoch,criterion,plot):
    train(model,train_dataset,optimizer,nb_epoch,criterion,plot)
    test(model,test_dataset,criterion,plot)