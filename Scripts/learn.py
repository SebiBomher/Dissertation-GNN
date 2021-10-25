import glob
from tokenize import Floatnumber
import numpy as np
import copy
from ray.tune.schedulers.async_hyperband import ASHAScheduler
import torch
import os
from ray import tune
from torch import nn
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from Scripts.Utility import Constants, DatasetSize, ModelType, OptimizerType, Folders
from Scripts.Models import CustomModel, STConvModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from Scripts.DataProccess import DataReader, DatasetSizeNumber, Graph
from tqdm import tqdm
from torch.optim import Adam,RMSprop,Adamax,AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Scripts.DatasetClasses import CustomDataset, LinearRegressionDataset, STConvDataset
from Scripts.Utility import Constant
from enum import Enum
import pickle

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
        self.batch_size = 8
        self.epsilon = config["epsilon"]
        self.lamda = config["lamda"]
        self.hidden_channels = 8
        self.optimizer_type = config["optimizer_type"]
        self.proccessed_data_path = param["proccessed_data_path"]
        self.graph_info_txt = param["graph_info_txt"]
        self.train_ratio = param["train_ratio"]
        self.test_ratio = param["test_ratio"]
        self.val_ratio = param["val_ratio"]
        self.checkpoint_dir = param["checkpoint_dir"]
        self.checkpoint_LR = param["checkpoint_LR"]
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

        if self.optimizer_type == OptimizerType.Adam:
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimizerType.RMSprop:
            self.optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimizerType.Adamax:
            self.optimizer = Adamax(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimizerType.AdamW:
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, threshold=0.00000001, threshold_mode='abs')

        if self.checkpoint_dir:
            checkpoint = os.path.join(self.checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

    def set_data(
            proccessed_data_path : str,
            datareader : DataReader,
            device : str):
        r"""
            Function to call to make all prerequireing data processing such that everything is done before training starts.
        """
        STConvDataset.__save_dataset(proccessed_data_path,datareader,device)
        CustomDataset.__save_dataset(proccessed_data_path,datareader,device)
        LinearRegressionDataset.__save_dataset(proccessed_data_path,datareader,device)
        
    def __startCUSTOM(config, param):
        learn = Learn(param,config)
        learn.__set_for_train()
        learn.__train_val_and_test()

    def __startSTCONV(config, param):
        learn = Learn(param,config)
        learn.__set_for_train()
        learn.__train_val_and_test()

    def startLR(param : dict):
        learn = Learn(param)
        learn.__set_for_train()
        learn.__train_val_and_test()


    def HyperParameterTuning(datasetsize : DatasetSize, model : ModelType, datareader : DataReader, criterion : staticmethod) -> None:
        r"""
            Function for hyper parameter tuning, it receives a datasetsize, model type, data reader and a loss function as a criterion, returns nothing
            Creates parameters for the function for the hyper parameter tuning
        """
        if ModelType == ModelType.Custom:
            learnMethod = Learn.__startCUSTOM
        else:
            learnMethod = Learn.__startSTCONV

        scheduler = ASHAScheduler(
            max_t=Constants.nb_epoch,
            grace_period=Constants.grace_period,
            reduction_factor=Constants.reduction_factor)

        param = {
            "learning_rate" : Constants.learning_rate,
            "num_features" : Constants.num_features,
            "EarlyStoppingPatience" : Constants.EarlyStoppingPatience,
            "path_data" : Folders.path_data,
            "proccessed_data_path" : Folders.proccessed_data_path,
            "graph_info_txt" : Folders.graph_info_path,
            "nb_epoch" : Constants.nb_epoch,
            "datareader" : datareader,
            "nodes_size" : datasetsize,
            "train_ratio" : Constants.train_ratio,
            "val_ratio" : Constants.val_ratio,
            "test_ratio" : Constants.test_ratio,
            "checkpoint_LR" : Folders.checkpoint_LR_path,
            "criterion" : criterion,
            "model_type" : model
        }


        config = {
            "K" : tune.choice([1,3,5,7]),
            "epsilon" : tune.choice([0.1, 0.3, 0.5, 0.7]),
            "optimizer_type" : tune.choice([OptimizerType.Adam,OptimizerType.AdamW,OptimizerType.Adamax,OptimizerType.RMSprop]),
            "lamda" : tune.choice([1, 3, 5, 10])
        }

        result = tune.run(
            tune.with_parameters(learnMethod, param = param),
            resources_per_trial={"cpu": 8, "gpu": 1},
            config=config,
            metric="loss",
            mode="min",
            num_samples=Constants.num_samples,
            scheduler=scheduler
        )

        best_trial = result.get_best_trial("loss", "min", "last")

        print("Best trial config: {}".format(best_trial.config))
        print("Best trial for {} model final validation loss: {}".format(model.name,best_trial.last_result["loss"]))