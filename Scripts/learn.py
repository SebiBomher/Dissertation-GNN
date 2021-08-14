from ray import tune
from torch import nn
from Scripts.models import STConvModel
from Scripts.data_proccess import get_dataset_experimental_STCONV
from tqdm import tqdm
import numpy as np
import copy
import torch
import os
from torch.optim import Adam,SGD,RMSprop,Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Scripts.STConvDataProcess import get_dataset_STCONV

def RMSE(y_pred,y_true):
    return torch.sqrt(torch.mean((y_pred-y_true)**2))
    
def MAPE(y_pred, y_true):
  return torch.mean(torch.abs((y_true - y_pred) / y_true))

def MAE(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred)))

def MSE(y_true,y_pred):
    return torch.mean((y_pred-y_true)**2)

def train(model,train_dataset,validation_dataset,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience):
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
        val_loss = val(model,validation_dataset,criterion,epoch,optimizer)
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
        
def val(model,dataset,criterion,epoch,optimizer):
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
            (model.state_dict(), optimizer.state_dict()), path)

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
    print("Best trial test set loss: {}".format(loss))

def train_val_and_test(model,train_dataset,validation_dataset,test_dataset,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience):
    best_model = train(model,train_dataset,validation_dataset,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience)
    test(best_model,test_dataset,criterion)
    

def learn(config, checkpoint_dir=None, time_step = 1, criterion = MSE, nb_epoch = 200, model_type = "STCONV",nodes_size="Full"):
    learning_rate = 0.01
    num_nodes = 8
    num_features = 3
    kernel_size = 1
    EarlyStoppingPatience = 10
    path_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    path_processed_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed"
    graph_info_txt = "d07_text_meta_2021_03_27.txt"
    if model_type == "STCONV":
        train_dataset, validation_dataset, test_dataset, num_nodes = get_dataset_STCONV(path=path_data,
                                                                                        path_proccessed_data=path_processed_data,
                                                                                        graph_info_txt=graph_info_txt, 
                                                                                        train_ratio = 0.6, 
                                                                                        test_ratio = 0.2, 
                                                                                        val_ratio = 0.2, 
                                                                                        batch_size=config["batch_size"],
                                                                                        time_steps=time_step,
                                                                                        epsilon=config["epsilon"],
                                                                                        lamda=config["lamda"],
                                                                                        nodes_size=nodes_size)
        
        model = STConvModel(node_features = num_features,
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
    elif config["optimizer_type"] == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=learning_rate)
    elif config["optimizer_type"] == "Adamax":
        optimizer = Adamax(model.parameters(), lr=learning_rate)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.00000001, threshold_mode='abs')

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_dataset, validation_dataset, test_dataset = get_dataset_experimental_STCONV(path=path_data,
                                                                                    train_test_ratio = 0.8, 
                                                                                    train_validation_ratio = 0.8,
                                                                                    batch_size=config["batch_size"],
                                                                                    time_steps=time_step)

    train_val_and_test(model,train_dataset,validation_dataset,test_dataset,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience)
