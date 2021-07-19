from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import copy
import torch

def MSE(y_true,y_pred):
    return torch.mean((y_pred-y_true)**2)
    
def train(model,train_dataset,validation_dataset,optimizer,nb_epoch,criterion):
    model.train()
    best_val_loss = np.inf
    val_model = model
    for epoch in tqdm(range(nb_epoch)):
        loss = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss += criterion(y_hat,snapshot.y)
        
        # Validation Step at epoch end
        val_loss = test(model,validation_dataset,criterion)
        if val_loss < best_val_loss:
            val_model = copy.deepcopy(model)
            best_val_loss = val_loss

        loss = loss / (time+1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return val_model
        
def test(model,dataset,criterion):
    model.eval()
    loss = 0
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss += criterion(y_hat,snapshot.y)
    loss = loss / (time+1)
    loss = loss.item()
    return loss

def train_val_and_test(model,train_dataset,validation_dataset,test_dataset,optimizer,nb_epoch,criterion):
    best_model = train(model,train_dataset,validation_dataset,optimizer,nb_epoch,criterion)
    test_loss = test(best_model,test_dataset,criterion)
    print("{0}: {1}".format(criterion.__name__,test_loss))