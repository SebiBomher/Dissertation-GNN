import glob
from ray import tune
from torch import nn
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from Scripts.models import CustomModel, STConvModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from Scripts.data_proccess import DatasetSizeNumber, Graph
from tqdm import tqdm
import numpy as np
import copy
import torch
import os
from torch.optim import Adam,RMSprop,Adamax,AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Scripts.datasetsClasses import CustomDataset, LinearRegressionDataset, STConvDataset
from enum import Enum
import pickle
class ModelType(Enum):
    r"""
    
    """
    Custom = 0
    STCONV = 1
    LinearRegression = 2

class OptimiserType(Enum):
    r"""
    
    """
    Adam = 0
    RMSprop = 1
    Adamax = 2
    AdamW = 3
class LossFunction():
    
    def Criterions():
        return [LossFunction.RMSE,LossFunction.MAPE,LossFunction.MAE,LossFunction.MSE]

    def RMSE(y_pred,y_true) -> Tensor:
        return torch.sqrt(torch.mean((y_pred-y_true)**2))
        
    def MAPE(y_pred, y_true) -> Tensor:
        return torch.mean(torch.abs((y_true - y_pred) / y_true))

    def MAE(y_pred, y_true) -> Tensor:
        return torch.mean(torch.abs((y_true - y_pred)))

    def MSE(y_true,y_pred) -> Tensor:
        return torch.mean((y_pred-y_true)**2)

class Learn():

    def __init__(self,param,info,config):
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
        self.checkpoint_LR = param["checkpoint_LR"]
        self.results_folder = param["results_folder"]
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
        dataloader = DataLoader(self.train_dataset,batch_size = 1,shuffle=False,num_workers=0)
        edge_index = self.train_dataset.get_edge_index()
        edge_weight = self.train_dataset.get_edge_weight()
        for epoch in tqdm(range(self.nb_epoch)):
            loss = 0
            iter = 0
            for time, (x,y) in enumerate(dataloader):
                X = x[0]
                Y = y[0]
                y_hat = self.model(X, edge_index, edge_weight)
                loss += self.criterion(y_hat,Y)
                iter +=1

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
            loss = loss / (iter+1)
            self.scheduler.step(loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print("Epoch {0} : Validation loss {1} ; Train loss {2};".format(epoch,val_loss,loss))
        return val_model
            
    def __val(self,epoch):
        self.model.eval()
        loss = 0
        dataloader = DataLoader(self.validation_dataset,batch_size = 1,shuffle=False,num_workers=0)
        edge_index = self.train_dataset.get_edge_index()
        edge_weight = self.train_dataset.get_edge_weight()
        iter = 0
        for time, (x,y) in enumerate(dataloader):
            X = x[0]
            Y = y[0]
            y_hat = self.model(X, edge_index, edge_weight)
            loss += self.criterion(y_hat,Y)
            iter +=1
        loss = loss / (iter+1)
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
        dataloader = DataLoader(self.test_dataset,batch_size = 1,shuffle=False,num_workers=0)
        edge_index = self.train_dataset.get_edge_index()
        edge_weight = self.train_dataset.get_edge_weight()
        iter = 0
        for time, (x,y) in enumerate(dataloader):
            X = x[0]
            Y = y[0]
            y_hat = self.model(X, edge_index, edge_weight)
            loss += self.criterion(y_hat,Y)
            iter +=1

        loss = loss / (iter+1)
        loss = loss.item()
        print("Best trial test set loss: {}".format(loss))

    def __LRTrainAndTest(self):
        
        parameters = {
            'normalize':[True],
        }

        lr_model = LinearRegression()
        clf = GridSearchCV(lr_model, parameters, refit=True, cv=5)
        all_loss = 0
        for index, (X_train, X_test, Y_train, Y_test, node_id) in enumerate(self.train_dataset):
            best_model = clf.fit(X_train,Y_train)
            y_pred = best_model.predict(X_test)
            loss = self.criterion(torch.FloatTensor(Y_test).to(self.train_dataset.device),torch.FloatTensor(y_pred).to(self.train_dataset.device))
            all_loss += loss
            loss = loss.item()
            print("Best trial test set loss: {0} for node id : {1}".format(loss,node_id))
            if not os.path.exists(self.checkpoint_LR):
                os.makedirs(self.checkpoint_LR)
            name_model = os.path.join(self.checkpoint_LR,"model_node_{0}_{1}_{2}.pickle".format(node_id,self.criterion.__name__,loss))
            with open(name_model, 'wb') as handle:
                pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        all_loss = all_loss.item()
        print(all_loss)
        all_loss = all_loss/(DatasetSizeNumber.Medium.value)
        print(all_loss)
        name_file = os.path.join(self.checkpoint_LR,"final_score.txt")
        print("Linear Regression {0} : {1}".format(self.criterion.__name__,all_loss))
        with open(name_file, 'w') as f:
            f.write(str(all_loss))

    

    def __train_val_and_test(self):
        if self.model_type == ModelType.LinearRegression:
            self.__LRTrainAndTest()
        else:
            best_model = self.__train()
            self.__test(best_model)

    def __set_for_data(self):
        device = "cpu"
        self.train_dataset, self.validation_dataset, self.test_dataset = STConvDataset.get_dataset_STCONV(
                                                                                            path_proccessed_data=self.proccessed_data_path,
                                                                                            train_ratio = self.train_ratio, 
                                                                                            test_ratio = self.test_ratio, 
                                                                                            val_ratio = self.val_ratio, 
                                                                                            batch_size=self.batch_size,
                                                                                            time_steps=1,
                                                                                            epsilon=self.epsilon,
                                                                                            lamda=self.lamda,
                                                                                            nodes_size=self.nodes_size,
                                                                                            datareader= self.datareader,
                                                                                            device= device)

        self.train_dataset, self.validation_dataset, self.test_dataset = CustomDataset.get_dataset_Custom(
                                                                                            path_proccessed_data=self.proccessed_data_path,
                                                                                            train_ratio = self.train_ratio, 
                                                                                            test_ratio = self.test_ratio, 
                                                                                            val_ratio = self.val_ratio, 
                                                                                            epsilon=self.epsilon,
                                                                                            lamda=self.lamda,
                                                                                            nodes_size=self.nodes_size,
                                                                                            datareader= self.datareader,
                                                                                            device= device)
        self.train_dataset = LinearRegressionDataset(self.proccessed_data_path,self.datareader,device)

    def __set_for_train(self):
        device = "cpu"
        if self.model_type == ModelType.STCONV:
            self.train_dataset, self.validation_dataset, self.test_dataset = STConvDataset.get_dataset_STCONV(
                                                                                            path_proccessed_data=self.proccessed_data_path,
                                                                                            train_ratio = self.train_ratio, 
                                                                                            test_ratio = self.test_ratio, 
                                                                                            val_ratio = self.val_ratio, 
                                                                                            batch_size=self.batch_size,
                                                                                            time_steps=1,
                                                                                            epsilon=self.epsilon,
                                                                                            lamda=self.lamda,
                                                                                            nodes_size=self.nodes_size,
                                                                                            datareader= self.datareader,
                                                                                            device= device)
            
            self.model = STConvModel(node_features = self.num_features,
                                num_nodes = Graph.get_number_nodes_by_size(self.nodes_size),
                                hidden_channels = self.hidden_channels,
                                kernel_size = 1,
                                K = 1)
        elif self.model_type == ModelType.Custom:
            self.train_dataset, self.validation_dataset, self.test_dataset = CustomDataset.get_dataset_Custom(
                                                                                            path_proccessed_data=self.proccessed_data_path,
                                                                                            train_ratio = self.train_ratio, 
                                                                                            test_ratio = self.test_ratio, 
                                                                                            val_ratio = self.val_ratio, 
                                                                                            epsilon=self.epsilon,
                                                                                            lamda=self.lamda,
                                                                                            nodes_size=self.nodes_size,
                                                                                            datareader= self.datareader,
                                                                                            device= device)
            self.model = CustomModel(node_features = self.num_features, K = 3)

        elif self.model_type == ModelType.LinearRegression:
            self.train_dataset = LinearRegressionDataset(self.proccessed_data_path,self.datareader,device)

        if self.model_type == ModelType.LinearRegression : return

        self.model.to(device)

        if self.optimizer_type == OptimiserType.Adam:
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimiserType.RMSprop:
            self.optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimiserType.Adamax:
            self.optimizer = Adamax(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimiserType.AdamW:
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, threshold=0.00000001, threshold_mode='abs')

        if self.checkpoint_dir:
            checkpoint = os.path.join(self.checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

    def set_data(config,info,param):
        learn = Learn(param,info,config)
        learn.__set_for_data()
        
    def startCUSTOM(config, info, param):
        learn = Learn(param,info,config)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        learn.__set_for_train()
        learn.__train_val_and_test()

    def startSTCONV(config, info, param):
        learn = Learn(param,info,config)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        learn.__set_for_train()
        learn.__train_val_and_test()
