from ray import tune
from torch import nn
from Scripts.models import ST_GCN
from Scripts.data_proccess import get_dataset, get_dataset_experimental
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import copy
import torch
import os
from torch.optim import Adam,SGD,RMSprop,Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau

def RMSE(y_pred,y_true):
    return torch.sqrt(torch.mean((y_pred-y_true)**2))
    
def MAPE(y_pred, y_true):
  return torch.mean(torch.abs((y_true - y_pred) / y_true))

def MAE(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred)))

def MSE(y_true,y_pred):
    return torch.mean((y_pred-y_true)**2)

def train(model,train_dataset,validation_dataset,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience,net):
    model.train()
    best_val_loss = np.inf
    val_model = model
    epoch_no_improvement = EarlyStoppingPatience
    for epoch in tqdm(range(nb_epoch)):
        loss = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss += criterion(y_hat,snapshot.y)

        # Validation Step at epoch end
        val_loss = val(model,validation_dataset,criterion,epoch,optimizer,net)
        if val_loss < best_val_loss:
            val_model = copy.deepcopy(model)
            best_val_loss = val_loss
            epoch_no_improvement = EarlyStoppingPatience
        else:
            epoch_no_improvement -=1

        if epoch_no_improvement == 0:
            print("Early stopping at epoch: {0}".format(epoch))
            break
        loss = loss / (time+1)
        scheduler.step(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return val_model
        
def val(model,dataset,criterion,epoch,optimizer,net):
    model.eval()
    loss = 0
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss += criterion(y_hat,snapshot.y)
    loss = loss / (time+1)
    loss = loss.item()

    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(
            (net.state_dict(), optimizer.state_dict()), path)

    tune.report(loss=loss)

    return loss

def test(model,dataset,criterion):
    model.eval()
    loss = 0
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss += criterion(y_hat,snapshot.y)
    loss = loss / (time+1)
    loss = loss.item()
    return loss

def train_val_and_test(model,train_dataset,validation_dataset,test_dataset,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience,net):
    best_model = train(model,train_dataset,validation_dataset,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience,net)
    test_loss = test(best_model,test_dataset,criterion)
    return best_model,test_loss

def learn(config, checkpoint_dir=None, time_step = 1, criterion = MSE):
    
    learning_rate = 0.01
    num_nodes = 8
    num_features = 3
    kernel_size = 1
    nb_epoch = 200
    EarlyStoppingPatience = 20
    path_model_save = "E:\\FacultateMasterAI\\Dissertation-GNN\\models"
    path_data = "E:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    # train_dataset, validation_dataset, test_dataset, num_nodes = get_dataset(path=path_data, train_test_ratio = 0.8, train_validation_ratio = 0.8,batch_size=config["batch_size"],time_steps=config["time_steps"],epsilon=config["epsilon"],lamda=config["lamda"])
    # print(num_nodes)
    
    model = ST_GCN(node_features = num_features,
                        num_nodes = num_nodes,
                        hidden_channels = config["hidden_channels"],
                        kernel_size = kernel_size,
                        K = config["K"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    if config["optimizer_type"] == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif config["optimizer_type"] == "SGD":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    elif config["optimizer_type"] == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=learning_rate)
    elif config["optimizer_type"] == "Adamax":
        optimizer = Adamax(model.parameters(), lr=learning_rate)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.00000001, threshold_mode='abs')

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_dataset, validation_dataset, test_dataset = get_dataset_experimental(path=path_data, train_test_ratio = 0.8, train_validation_ratio = 0.8,batch_size=config["batch_size"],time_steps=time_step)

    trained_model,test_loss = train_val_and_test(model,train_dataset,validation_dataset,test_dataset,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience,model)
    print("{0}: {1}".format(config["optimizer_type"].__name__,test_loss))
    torch.save(trained_model, os.path.join(path_model_save,"{0}_{1}_{2}_{3}_{4}_{5}__{6}.pth".format(str(config["batch_size"]),str(config["hidden_channels"]),str(config["K"]),str(config["epsilon"]),str(config["lamda"]),str(config["optimizer_type"]),str(test_loss))))
