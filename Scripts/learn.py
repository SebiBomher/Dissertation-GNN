from ray import tune
from torch import nn
from torch.functional import Tensor
from Scripts.models import STConvModel
from Scripts.data_proccess import Graph
from tqdm import tqdm
import numpy as np
import copy
import torch
import os
from torch.optim import Adam,RMSprop,Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Scripts.datasetsClasses import STConvDataset
from enum import Enum

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
    
    def RMSE(y_pred,y_true) -> Tensor:
        return torch.sqrt(torch.mean((y_pred-y_true)**2))
        
    def MAPE(y_pred, y_true) -> Tensor:
        return torch.mean(torch.abs((y_true - y_pred) / y_true))

    def MAE(y_pred, y_true) -> Tensor:
        return torch.mean(torch.abs((y_true - y_pred)))

    def MSE(y_true,y_pred) -> Tensor:
        return torch.mean((y_pred-y_true)**2)

class Learn():

    def __init__(self,config,info,param):
        self.batch_size = config["batch_size"]
        self.epsilon = config["epsilon"]
        self.lamda = config["lamda"]
        self.hidden_channels = config["hidden_channels"]
        self.optimizer_type = config["optimizer_type"]
        self.proccessed_data_path = param["proccessed_data_path"]
        self.graph_info_txt = param["graph_info_txt"]
        self.train_ratio = param["train_ratio"]
        self.test_ratio = param["test_ratio"]
        self.val_ratio = param["val_ratio"]
        self.checkpoint_dir = param["checkpoint_dir"]
        self.learning_rate = param["learning_rate"]
        self.EarlyStoppingPatience = param["EarlyStoppingPatience"]
        self.nb_epoch = param["nb_epoch"]
        self.nodes_size = param["nodes_size"]
        self.datareader = param["datareader"]
        self.num_features = param["num_features"]
        self.criterion = info["criterion"]
        self.model_type = info["model_type"]

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

    def __train_val_and_test(self):
        best_model = self.__train()
        self.__test(best_model)

    def __set_for_train(self):
        if self.model_type == ModelType.STCONV:
            self.train_dataset, self.validation_dataset, self.test_dataset = STConvDataset.get_dataset_STCONV(path_proccessed_data=self.proccessed_data_path,
                                                                                            graph_info_txt=self.graph_info_txt, 
                                                                                            train_ratio = self.train_ratio, 
                                                                                            test_ratio = self.test_ratio, 
                                                                                            val_ratio = self.val_ratio, 
                                                                                            batch_size=self.batch_size,
                                                                                            time_steps=1,
                                                                                            epsilon=self.epsilon,
                                                                                            lamda=self.lamda,
                                                                                            nodes_size=self.nodes_size,
                                                                                            datareader= self.datareader)
            
            self.model = STConvModel(node_features = self.num_features,
                                num_nodes = Graph.get_number_nodes_by_size(self.nodes_size),
                                hidden_channels = self.hidden_channels,
                                kernel_size = 1,
                                K = 1)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.model.to(device)

        if self.optimizer_type == OptimiserType.Adam:
            optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimiserType.RMSprop:
            optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimiserType.Adamax:
            optimizer = Adamax(self.model.parameters(), lr=self.learning_rate)

        self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.00000001, threshold_mode='abs')

        if self.checkpoint_dir:
            checkpoint = os.path.join(self.checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            self.model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    def start(config, info, param):
        learn = Learn(param,info,config)
        learn.__set_for_train()
        learn.__train_val_and_test()
