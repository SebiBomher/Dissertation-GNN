#region Imports

import pickle
import numpy as np
import copy
import torch
import os
from ray import tune
from torch import nn
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from Scripts.Utility import Constants, DatasetSize, DatasetSizeNumber, ModelType, OptimizerType, Folders
from Scripts.Models import CustomModel, STConvModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from Scripts.DataProccess import DataReader,  Graph
from tqdm import tqdm
from torch.optim import Adam,RMSprop,Adamax,AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Scripts.DatasetClasses import CustomDataset, LinearRegressionDataset, STConvDataset
from ray.tune.schedulers.async_hyperband import ASHAScheduler

#endregion

class LossFunction():
    r"""
        Loss Function class, contains 5 methods, 4 of which are the actual loss function and one which contains a list of all of them.
    """
    
    #region Class Functions

    def Criterions() -> list:
        r"""
            Function which returns all loss functions criterions, contains RMSE, MAPE, MAE, MSE
            Class Function.
            No Arguments.
            Returns List.
        """
        return [LossFunction.RMSE,LossFunction.MAPE,LossFunction.MAE,LossFunction.MSE]

    def RMSE(y_pred : Tensor,y_true : Tensor) -> Tensor:
        r"""
            Root Mean Squared Error Function
            Args:
                y_pred : Tensor, Predicted Values
                y_true : Tensor, True Values
            Class Function
            Returns Tensor
        """
        return torch.sqrt(torch.mean((y_pred-y_true)**2))
        
    def MAPE(y_pred : Tensor,y_true : Tensor) -> Tensor:
        r"""
            Mean Average Percentage Error
            Args:
                y_pred : Tensor, Predicted Values
                y_true : Tensor, True Values
            Class Function
            Returns Tensor
        """
        return torch.mean(torch.abs((y_true - y_pred) / y_true))

    def MAE(y_pred : Tensor,y_true : Tensor) -> Tensor:
        r"""
            Mean Average Error
            Args:
                y_pred : Tensor, Predicted Values
                y_true : Tensor, True Values
            Class Function
            Returns Tensor
        """
        return torch.mean(torch.abs((y_true - y_pred)))

    def MSE(y_pred : Tensor,y_true : Tensor) -> Tensor:
        r"""
            Mean Square Error
            Args:
                y_pred : Tensor, Predicted Values
                y_true : Tensor, True Values
            Class Function
            Returns Tensor
        """
        return torch.mean((y_true-y_pred)**2)
    
    #endregion

class Learn():
    
    #region Constructors & Properties

    def __init__(self,datareader : DataReader,model_type : ModelType):
        r"""
            Linear Regression Constructor
            Args:
                datareader : DataReader, datareader for Datareading
                model_type : ModelType, modeltype, which is LinearRegression
        """
        self.datareader = datareader
        self.model_type = model_type
        self.device = Constants.device

    def __init__(self, param: dict, config: dict):
        r"""
            GNN Constructor
            
        """
        self.epsilon = config["epsilon"]
        self.sigma = config["sigma"]
        self.optimizer_type = config["optimizer_type"]
        self.train_ratio = param["train_ratio"]
        self.test_ratio = param["test_ratio"]
        self.val_ratio = param["val_ratio"]
        self.learning_rate = param["learning_rate"]
        self.EarlyStoppingPatience = param["EarlyStoppingPatience"]
        self.nb_epoch = param["nb_epoch"]
        self.nodes_size = param["nodes_size"]
        self.datareader = param["datareader"]
        self.num_features = param["num_features"]
        self.criterion = param["criterion"]
        self.model_type = param["model_type"]
        self.device = Constants.device

    #endregion

    #region Instance Functions.
    
    def __train(self):
        r"""
            Training step in the training process
            Args:
                epoch
            Returns a model, the best at validation loss
        """
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
            
    def __val(self,epoch : int):
        r"""
            Validation step in the training process
            Args:
                epoch
            Returns a float, the loss on validation
        """
        self.model.eval()
        loss = 0
        dataloader = DataLoader(self.validation_dataset,batch_size = 1,shuffle=False,num_workers=0)
        edge_index = self.train_dataset.get_edge_index()
        edge_weight = self.train_dataset.get_edge_weight()
        iter = 0
        for _, (x,y) in enumerate(dataloader):
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

    def __test(self,best_model) -> None:
        r"""
            Testing the GNN model
            Args:
                best_model, the best model from training (evaluated at validation)
            Instance Function
            Returns None.
        """
        best_model.eval()
        loss = 0
        dataloader = DataLoader(self.test_dataset,batch_size = 1,shuffle=False,num_workers=0)
        edge_index = self.train_dataset.get_edge_index()
        edge_weight = self.train_dataset.get_edge_weight()
        iter = 0
        for _, (x,y) in enumerate(dataloader):
            X = x[0]
            Y = y[0]
            y_hat = self.model(X, edge_index, edge_weight)
            loss += self.criterion(y_hat,Y)
            iter +=1

        loss = loss / (iter+1)
        loss = loss.item()
        print("Best trial test set loss: {}".format(loss))

    def __LRTrainAndTest(self) -> None:
        r"""
            Trains and tests Linear Regression
            No Arguments.
            Instance Function.
            Returns None.
        """
        parameters = {
            'normalize':[True],
        }

        lr_model = LinearRegression()
        clf = GridSearchCV(lr_model, parameters, refit=True, cv=5)
        all_loss = 0
        for _, (X_train, X_test, Y_train, Y_test, node_id) in enumerate(self.train_dataset):
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

    def __train_val_and_test(self) -> None:
        r"""
            Trains and tests a model.
            No Arguments.
            Instance Function.
            Returns None.
        """
        best_model = self.__train()
        self.__test(best_model)

    def __set_for_train(self) -> None:
        r"""
            Function to set data for training.
            Instance Function.
            No Arguments.
            Returns None.
        """
        
        if self.model_type == ModelType.STCONV:
            self.train_dataset, self.validation_dataset, self.test_dataset = STConvDataset.get_dataset_STCONV(
                                                                                            train_ratio = self.train_ratio, 
                                                                                            test_ratio = self.test_ratio, 
                                                                                            val_ratio = self.val_ratio, 
                                                                                            time_steps=1,
                                                                                            epsilon=self.epsilon,
                                                                                            sigma=self.sigma,
                                                                                            nodes_size=self.nodes_size,
                                                                                            datareader= self.datareader,
                                                                                            device= self.device)
            
            self.model = STConvModel(node_features = self.num_features,
                                num_nodes = Graph.get_number_nodes_by_size(self.nodes_size),
                                hidden_channels = self.hidden_channels,
                                kernel_size = 1,
                                K = 1)
        elif self.model_type == ModelType.Custom:
            self.train_dataset, self.validation_dataset, self.test_dataset = CustomDataset.get_dataset_Custom(
                                                                                            train_ratio = self.train_ratio, 
                                                                                            test_ratio = self.test_ratio, 
                                                                                            val_ratio = self.val_ratio, 
                                                                                            epsilon=self.epsilon,
                                                                                            sigma=self.sigma,
                                                                                            nodes_size=self.nodes_size,
                                                                                            datareader= self.datareader,
                                                                                            device= self.device)
            self.model = CustomModel(node_features = self.num_features, K = 3)

        elif self.model_type == ModelType.LinearRegression:
            self.train_dataset = LinearRegressionDataset(self.datareader)
            return

        self.model.to(self.device)

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

    #endregion
    
    #region Class Functions

    def set_data(datareader : DataReader) -> None:
        r"""
            Function to call to make all prerequireing data processing such that everything is done before training starts.
            Args:
                datareader : DataReader, datareader for data reading
            Class Function.
            Returns None.
        """
        STConvDataset.__save_dataset(datareader)
        CustomDataset.__save_dataset(datareader)
        LinearRegressionDataset.__save_dataset(datareader)

    def __startCUSTOM(config, param) -> None:
        r"""
            Function to start training Custom
            Class Function
            Args:
                param : dict, contains learning_rate, num_features, EarlyStoppingPatience, nb_epoch, datareader, nodes_size, train_ratio, val_ratio, test_ratio, criterion, model_type
                config : dict, contains K, epsilon, optimizer_type, sigma
            Returns None
        """
        learn = Learn(param,config)
        learn.__set_for_train()
        learn.__train_val_and_test()

    def __startSTCONV(config, param) -> None:
        r"""
            Function to start training STCONV
            Class Function
            Args:
                param : dict, contains learning_rate, num_features, EarlyStoppingPatience, nb_epoch, datareader, nodes_size, train_ratio, val_ratio, test_ratio, criterion, model_type
                config : dict, contains K, epsilon, optimizer_type, sigma
            Returns None
        """
        learn = Learn(param,config)
        learn.__set_for_train()
        learn.__train_val_and_test()

    def startLR(datareader : DataReader) -> None:
        r"""
            Function to start training Linear Regression
            Class Function.
            Args:
                datareader : DataReader, datareader for data reading
            Returns None.
        """
        learn = Learn(datareader = datareader, model_type = ModelType.LinearRegression)
        learn.__set_for_train()
        learn.__LRTrainAndTest()

    def HyperParameterTuning(datasetsize : DatasetSize, model : ModelType, datareader : DataReader, criterion : staticmethod) -> None:
        r"""
            Function for hyper parameter tuning, it receives a datasetsize, model type, data reader and a loss function as a criterion, returns nothing.
            Creates parameters for the function for the hyper parameter tuning.
            Args:
                datasetsize : DatasetSize, datasetsize for which to tune
                model : ModelType, model for which to tune
                datareader : DataReader, datareader class for data reading
                criterion : staticmethod, loss function criterion
            Class Function.
            Returns None.
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
            "nb_epoch" : Constants.nb_epoch,
            "datareader" : datareader,
            "nodes_size" : datasetsize,
            "train_ratio" : Constants.train_ratio,
            "val_ratio" : Constants.val_ratio,
            "test_ratio" : Constants.test_ratio,
            "criterion" : criterion,
            "model_type" : model
        }


        config = {
            "K" : tune.choice([1,3,5,7]),
            "epsilon" : tune.choice([0.1, 0.3, 0.5, 0.7]),
            "optimizer_type" : tune.choice([OptimizerType.Adam,OptimizerType.AdamW,OptimizerType.Adamax,OptimizerType.RMSprop]),
            "sigma" : tune.choice([1, 3, 5, 10])
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

    #endregion

    