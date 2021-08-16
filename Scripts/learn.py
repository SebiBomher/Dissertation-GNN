from ray import tune
from torch import nn
from torch.functional import Tensor
from Scripts.models import STConvModel
from Scripts.data_proccess import DatasetSize, Graph, get_dataset_experimental_STCONV
from tqdm import tqdm
import numpy as np
import copy
import torch
import os
from torch.optim import Adam,RMSprop,Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Scripts.datasetsClasses import DatasetClass, STConvDataset
from enum import Enum

class LossFunctionType(Enum):
    r"""
    
    """
    RMSE = 0
    MAPE = 1
    MAE = 2
    MSE = 3

class ModelType(Enum):
    r"""
    
    """
    STCONV = 0
    ASTGCN = 1
    MSTGCN = 2
    GMAN = 3

class OptimiserType(Enum):
    r"""
    
    """
    Adam = 0
    RMSprop = 1
    Adamax = 2

class LossFunction():
    def __init__(self,function_type : LossFunctionType, y_true,y_pred):
        self.function_type = function_type
        self.y_true = y_true
        self.y_pred = y_pred

    def RMSE(y_pred,y_true) -> Tensor:
        return torch.sqrt(torch.mean((y_pred-y_true)**2))
        
    def MAPE(y_pred, y_true) -> Tensor:
        return torch.mean(torch.abs((y_true - y_pred) / y_true))

    def MAE(y_pred, y_true) -> Tensor:
        return torch.mean(torch.abs((y_true - y_pred)))

    def MSE(y_true,y_pred) -> Tensor:
        return torch.mean((y_pred-y_true)**2)

class Learn():

    def __train(self):
        self.model.train()
        best_val_loss = np.inf
        val_model = self.model
        epoch_no_improvement = self.EarlyStoppingPatience
        for epoch in tqdm(range(self.nb_epoch)):
            loss = 0
            for time, snapshot in enumerate(self.train_dataset):
                y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                loss += self.criterion(y_hat,snapshot.y)

            # Validation Step at epoch end
            val_loss = self.__val(epoch)
            if val_loss < best_val_loss:
                val_model = copy.deepcopy(self.model)
                best_val_loss = val_loss
                epoch_no_improvement = self.EarlyStoppingPatience
            else:
                epoch_no_improvement -=1

            if epoch_no_improvement == 0:
                print("Early stopping at epoch: {0}".format(epoch))
                break
            loss = loss / (time+1)
            self.scheduler.step(loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return val_model
            
    def __val(self,epoch):
        self.model.eval()
        loss = 0
        for time, snapshot in enumerate(self.validation_dataset):
            y_hat = self.model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss += self.criterion(y_hat,snapshot.y)
        loss = loss / (time+1)
        loss = loss.item()

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (self.model.state_dict(), self.optimizer.state_dict()), path)

        tune.report(loss=loss)

        return loss

    def __test(self,best_model):
        best_model.eval()
        loss = 0
        for time, snapshot in enumerate(self.test_dataset):
            y_hat = best_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss += self.criterion(y_hat,snapshot.y)
        loss = loss / (time+1)
        loss = loss.item()
        print("Best trial test set loss: {}".format(loss))

    def train_val_and_test(self):
        best_model = self.__train()
        self.__test(best_model)
        
    def set_configuration(self,optimizer,nb_epoch,criterion,scheduler,EarlyStoppingPatience,model_type : ModelType):
        self.optimizer = optimizer
        self.nb_epoch = nb_epoch
        self.criterion = criterion
        self.scheduler = scheduler
        self.EarlyStoppingPatience = EarlyStoppingPatience
        self.model_type = model_type

    def set_dataset_and_model(self,config,param):
        if self.model_type == ModelType.STCONV:
            self.train_dataset, self.validation_dataset, self.test_dataset = STConvDataset.get_dataset_STCONV(path_proccessed_data=param["proccessed_data_path"],
                                                                                            graph_info_txt=param["graph_info_txt"], 
                                                                                            train_ratio = param["train_ratio"], 
                                                                                            test_ratio = param["test_ratio"], 
                                                                                            val_ratio = param["val_ratio"], 
                                                                                            batch_size=config["batch_size"],
                                                                                            time_steps=1,
                                                                                            epsilon=config["epsilon"],
                                                                                            lamda=config["lamda"],
                                                                                            nodes_size=param["nodes_size"])
            
            self.model = STConvModel(node_features = param["num_features"],
                                num_nodes = Graph.get_number_nodes_by_size(param["nodes_size"]),
                                hidden_channels = config["hidden_channels"],
                                kernel_size = 1,
                                K = config["K"])


    def learn(config,info, checkpoint_dir=None, time_step = 1, criterion = LossFunctionType.MAE, nb_epoch = 200, model_type = ModelType.STCONV,nodes_size=DatasetSize.Full):
       
        param = {
            "learning_rate" : 0.01,
            "num_nodes": 8,
            "num_features" : 3,
            "EarlyStoppingPatience" : 10,
            "path_data" : "D:\\FacultateMasterAI\\Dissertation-GNN\\Data",
            "path_processed_data" : "D:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed",
            "graph_info_txt" : "d07_text_meta_2021_03_27.txt"
        }

        learn = Learn()
        learn.set_dataset_and_model(config,param)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                learn.model = nn.DataParallel(learn.model)
        learn.model.to(device)

        if config["optimizer_type"] == "Adam":
            optimizer = Adam(learn.model.parameters(), lr=param["learning_rate"])
        elif config["optimizer_type"] == "RMSprop":
            optimizer = RMSprop(learn.model.parameters(), lr=param["learning_rate"])
        elif config["optimizer_type"] == "Adamax":
            optimizer = Adamax(learn.model.parameters(), lr=param["learning_rate"])
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.00000001, threshold_mode='abs')

        if info["checkpoint_dir"]:
            checkpoint = os.path.join(info["checkpoint_dir"], "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            learn.model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        
        learn.set_configuration(optimizer,nb_epoch,criterion,scheduler,param["EarlyStoppingPatience"],model_type)

        learn.train_val_and_test()
