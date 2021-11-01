#region Imports

import pickle
import numpy as np
import copy
import pandas as pd
import torch
import os
import shutil
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
from torch.optim import Adam, RMSprop, Adamax, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Scripts.DatasetClasses import CustomDataset, LinearRegressionDataset, STConvDataset
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from datetime import datetime
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
        return [LossFunction.RMSE, LossFunction.MAPE, LossFunction.MAE, LossFunction.MSE]

    def RMSE(y_pred: Tensor, y_true: Tensor) -> Tensor:
        r"""
            Root Mean Squared Error Function
            Args:
                y_pred : Tensor, Predicted Values
                y_true : Tensor, True Values
            Class Function
            Returns Tensor
        """
        return torch.sqrt(torch.mean((y_pred-y_true)**2))

    def MAPE(y_pred: Tensor, y_true: Tensor) -> Tensor:
        r"""
            Mean Average Percentage Error
            Args:
                y_pred : Tensor, Predicted Values
                y_true : Tensor, True Values
            Class Function
            Returns Tensor
        """
        return torch.mean(torch.abs((y_true - y_pred) / y_true))

    def MAE(y_pred: Tensor, y_true: Tensor) -> Tensor:
        r"""
            Mean Average Error
            Args:
                y_pred : Tensor, Predicted Values
                y_true : Tensor, True Values
            Class Function
            Returns Tensor
        """
        return torch.mean(torch.abs((y_true - y_pred)))

    def MSE(y_pred: Tensor, y_true: Tensor) -> Tensor:
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
    def InitGNN(self, param: dict, config: dict):
        r"""
            GNN Constructor
            
        """
        self.epsilon = config["epsilon"]
        self.sigma = config["sigma"]
        self.optimizer_type = config["optimizer_type"]
        self.nodes_size = config["nodes_size"]
        self.model_type = config["model_type"]
        self.train_ratio = param["train_ratio"]
        self.test_ratio = param["test_ratio"]
        self.val_ratio = param["val_ratio"]
        self.learning_rate = param["learning_rate"]
        self.EarlyStoppingPatience = param["EarlyStoppingPatience"]
        self.nb_epoch = param["nb_epoch"]
        self.datareader = param["datareader"]
        self.num_features = param["num_features"]
        self.device = Constants.device

    def InitLR(self, datareader: DataReader, model_type: ModelType):
        r"""
            Linear Regression Constructor
            Args:
                datareader : DataReader, datareader for Datareading
                model_type : ModelType, modeltype, which is LinearRegression
        """
        self.datareader = datareader
        self.model_type = model_type
        self.device = Constants.device

    def __init__(self):
        return
    #endregion

    #region Instance Functions.

    def __train_val_and_test(self, experiment_name: str):
        r"""
            Training step in the training process
            Returns nothing
        """
        dfResults = pd.DataFrame(columns=["Model", "Epsilon", "Sigma", "Size","Criterion", "Loss", "Epoch", "OptimizerType", "Trial", "TestOrVal"])
        self.model.train()
        best_val_loss = np.inf
        val_model = self.model
        epoch_no_improvement = self.EarlyStoppingPatience
        dataloader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=False, num_workers=0)
        edge_index = self.train_dataset.get_edge_index()
        edge_weight = self.train_dataset.get_edge_weight()
        for epoch in tqdm(range(self.nb_epoch)):
            loss = 0
            for index, (x, y) in enumerate(dataloader):
                X = x[0]
                Y = y[0]
                y_hat = self.model(X, edge_index, edge_weight)
                loss += LossFunction.MAE(y_hat, Y)

            # Validation Step at epoch end
            val_loss, dfResults = self.__val(dfResults, epoch)
            if val_loss < best_val_loss:
                val_model = copy.deepcopy(self.model)
                best_val_loss = val_loss
                epoch_no_improvement = self.EarlyStoppingPatience
            else:
                epoch_no_improvement -= 1

            if epoch_no_improvement == 0:
                print("Early stopping at epoch: {0}".format(epoch))
                break
            loss = loss / (index+1)
            self.scheduler.step(loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print("Epoch {0} : Validation loss {1} ; Train loss {2};".format(
                epoch, val_loss, loss))

            # test step if this is the last epoch
            # Save dataframe results
            if epoch == self.nb_epoch - 1:
                dfResults = self.__test(val_model, dfResults, epoch)
                file_save = os.path.join(Folders.results_path, experiment_name, "{0}_{1}_{2}.csv".format(
                    self.model_type.name, str(self.nodes_size.name), str(tune.get_trial_id())))
                dfResults.to_csv(path_or_buf=file_save, index=False)

            # Log to tune
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (self.model.state_dict(), self.optimizer.state_dict()), path)

            tune.report(loss=val_loss)

    def __val(self, dfResults: pd.DataFrame, epoch: int):
        r"""
            Validation step in the training process
            Args:
                epoch : int, what epoch currently is
                dfResults : pd.DataFrame, For results writing
            Returns a float, the loss on validation
        """
        self.model.eval()
        loss = 0
        dataloader = DataLoader(self.validation_dataset,
                                batch_size=1, shuffle=False, num_workers=0)
        edge_index = self.train_dataset.get_edge_index()
        edge_weight = self.train_dataset.get_edge_weight()
        MAE_loss = 0
        for criterion in LossFunction.Criterions():
            loss = 0
            for index, (x, y) in enumerate(dataloader):
                X = x[0]
                Y = y[0]
                y_hat = self.model(X, edge_index, edge_weight)
                loss += criterion(y_hat, Y)
                if criterion == LossFunction.MAE:
                    MAE_loss += criterion(y_hat, Y)

            loss = loss / (index+1)
            loss = loss.item()

            results = {"Model": str(self.model_type.name),
                       "Epsilon": str(self.epsilon),
                       "Sigma": str(self.sigma),
                       "Size": str(self.nodes_size.name),
                       "Criterion": str(criterion.__name__),
                       "Loss": str(loss),
                       "Epoch": str(epoch),
                       "OptimizerType": self.optimizer_type.name,
                       "TestOrVal": "Validation",
                       "Trial": tune.get_trial_id()
                       }
            dfResults = dfResults.append(
                results, ignore_index=True)

            if criterion == LossFunction.MAE:
                MAE_loss = MAE_loss / (index+1)
                MAE_loss = MAE_loss.item()

        return MAE_loss, dfResults

    def __test(self, best_model, dfResults: pd.DataFrame, epoch: int) -> None:
        r"""
            Testing the GNN model
            Args:
                best_model, the best model from training (evaluated at validation)
                dfResults : pd.DataFrame, For results writing
            Instance Function
            Returns None.
        """
        best_model.eval()
        loss = 0
        dataloader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
        edge_index = self.test_dataset.get_edge_index()
        edge_weight = self.test_dataset.get_edge_weight()
        MAE_loss = 0
        for criterion in LossFunction.Criterions():
            loss = 0
            for index, (x, y) in enumerate(dataloader):
                X = x[0]
                Y = y[0]
                y_hat = best_model(X, edge_index, edge_weight)
                loss += criterion(y_hat, Y)
                if criterion == LossFunction.MAE:
                    MAE_loss += criterion(y_hat, Y)

            loss = loss / (index+1)
            loss = loss.item()

            results = {"Model": str(self.model_type.name),
                       "Epsilon": str(self.epsilon),
                       "Sigma": str(self.sigma),
                       "Size": str(self.nodes_size.name),
                       "Criterion": str(criterion.__name__),
                       "Loss": str(loss),
                       "Epoch": str(epoch),
                       "OptimizerType": self.optimizer_type.name,
                       "TestOrVal": "Test",
                       "Trial": tune.get_trial_id()
                       }
            dfResults = dfResults.append(
                results, ignore_index=True)

            if criterion == LossFunction.MAE:
                MAE_loss = MAE_loss / (index+1)
                MAE_loss = MAE_loss.item()

        print("Best trial test set loss: {}".format(MAE_loss))
        return dfResults

    def __LRTrainAndTest(self, experiment_name: str) -> None:
        r"""
            Trains and tests Linear Regression
            No Arguments.
            Instance Function.
            Returns None.
        """
        parameters = {
            'fit_intercept': [True],
        }

        # nr_files = len([name for name in os.listdir(Folders.checkpoint_LR_path) if os.path.isfile(os.path.join(Folders.checkpoint_LR_path, name))])

        # if nr_files == DatasetSizeNumber.Medium.value * 4:
        #     return
        results_folder = os.path.join(
            Folders.results_ray_path, experiment_name, Constants.checkpoint_LR_folder)

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        dfResults = pd.DataFrame(columns=["Node_Id", "Criterion", "Loss"])
        lr_model = LinearRegression()
        clf = GridSearchCV(lr_model, parameters, refit=True, cv=5)
        for _, (X_train, X_test, Y_train, Y_test, node_id) in enumerate(self.train_dataset):
            best_model = clf.fit(X_train, Y_train)
            y_pred = best_model.predict(X_test)
            for criterion in LossFunction.Criterions():
                loss = criterion(torch.FloatTensor(Y_test).to(
                    self.train_dataset.device), torch.FloatTensor(y_pred).to(self.train_dataset.device))
                loss = loss.item()
                result = {"Node_Id": node_id,
                          "Criterion": criterion.__name__, "Loss": str(loss)}
                dfResults = dfResults.append(result, ignore_index=True)
                if criterion == LossFunction.MAE:
                    print("Best trial test set loss {0} : {1} for node id : {2}".format(
                        criterion.__name__, loss, node_id))
                    name_model = os.path.join(results_folder, "model_node_{0}_{1}_{2}.pickle".format(
                        node_id, criterion.__name__, loss))
                    with open(name_model, 'wb') as handle:
                        pickle.dump(best_model, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
        folder_save = os.path.join(Folders.results_path, experiment_name)

        if not os.path.exists(folder_save):
            os.makedirs(folder_save)

        file_save = os.path.join(folder_save, "LinearRegression.csv")
        dfResults.to_csv(path_or_buf=file_save, index=False)

    def __set_for_train(self) -> None:
        r"""
            Function to set data for training.
            Instance Function.
            No Arguments.
            Returns None.
        """

        if self.model_type == ModelType.STCONV:
            self.train_dataset, self.validation_dataset, self.test_dataset = STConvDataset.get_dataset_STCONV(
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                val_ratio=self.val_ratio,
                epsilon=self.epsilon,
                sigma=self.sigma,
                nodes_size=self.nodes_size,
                datareader=self.datareader,
                device=self.device)

            self.model = STConvModel(node_features=self.num_features,
                                     num_nodes=Graph.get_number_nodes_by_size(
                                         self.nodes_size),
                                     hidden_channels=Constants.hidden_channels,
                                     kernel_size=1,
                                     K=1)
        elif self.model_type == ModelType.Custom:
            self.train_dataset, self.validation_dataset, self.test_dataset = CustomDataset.get_dataset_Custom(
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                val_ratio=self.val_ratio,
                epsilon=self.epsilon,
                sigma=self.sigma,
                nodes_size=self.nodes_size,
                datareader=self.datareader,
                device=self.device)
            self.model = CustomModel(node_features=self.num_features, K=3)

        elif self.model_type == ModelType.LinearRegression:
            self.train_dataset = LinearRegressionDataset(self.datareader)
            return

        self.model.to(self.device)

        if self.optimizer_type == OptimizerType.Adam:
            self.optimizer = Adam(self.model.parameters(),
                                  lr=self.learning_rate)
        elif self.optimizer_type == OptimizerType.RMSprop:
            self.optimizer = RMSprop(
                self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimizerType.Adamax:
            self.optimizer = Adamax(
                self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OptimizerType.AdamW:
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.learning_rate)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, threshold=0.00000001, threshold_mode='abs')

    #endregion

    #region Class Functions

    def set_data(datareader: DataReader) -> None:
        r"""
            Function to call to make all prerequireing data processing such that everything is done before training starts.
            Args:
                datareader : DataReader, datareader for data reading
            Class Function.
            Returns None.
        """
        STConvDataset.save_dataset(datareader)
        CustomDataset.save_dataset(datareader)
        LinearRegressionDataset.save_dataset(datareader)

    def __start(config, param, checkpoint_dir=None) -> None:
        r"""
            Function to start training
            Class Function
            Args:
                param : dict, contains learning_rate, num_features, EarlyStoppingPatience, nb_epoch, datareader, nodes_size, train_ratio, val_ratio, test_ratio, criterion, model_type
                config : dict, contains K, epsilon, optimizer_type, sigma
            Returns None
        """
        learn = Learn()
        learn.InitGNN(param, config)
        learn.__set_for_train()
        learn.__train_val_and_test(param["experiment_name"])

    def startLR(datareader: DataReader, experiment_name: str) -> None:
        r"""
            Function to start training Linear Regression
            Class Function.
            Args:
                datareader : DataReader, datareader for data reading
            Returns None.
        """
        learn = Learn()
        learn.InitLR(datareader=datareader,
                     model_type=ModelType.LinearRegression)
        learn.__set_for_train()
        learn.__LRTrainAndTest(experiment_name=experiment_name)

    def trail_dirname_creator(trial):
        return f"{trial.config['model_type'].name}_{trial.config['nodes_size'].name}_{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}"

    def HyperParameterTuning(datasetsize: DatasetSize, model: ModelType, datareader: DataReader, experiment_name: str) -> None:
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

        scheduler = ASHAScheduler(
            max_t=Constants.nb_epoch,
            grace_period=Constants.grace_period,
            reduction_factor=Constants.reduction_factor)

        param = {
            "learning_rate": Constants.learning_rate,
            "num_features": Constants.num_features,
            "EarlyStoppingPatience": Constants.EarlyStoppingPatience,
            "nb_epoch": Constants.nb_epoch,
            "datareader": datareader,
            "train_ratio": Constants.train_ratio,
            "val_ratio": Constants.val_ratio,
            "test_ratio": Constants.test_ratio,
            "experiment_name": experiment_name
        }

        config = {
            "K": tune.choice([1, 3, 5, 7]),
            "epsilon": tune.choice([0.1, 0.3, 0.5, 0.7]),
            "optimizer_type": tune.choice([OptimizerType.Adam, OptimizerType.AdamW, OptimizerType.Adamax, OptimizerType.RMSprop]),
            "sigma": tune.choice([1, 3, 5, 10]),
            "model_type": tune.choice([model]),
            "nodes_size": tune.choice([datasetsize])
        }

        directory_experiment_ray = os.path.join(
            Folders.results_ray_path, experiment_name)

        Learn.__start.__name__ = f"{model.name}_{datasetsize.name}"

        result = tune.run(
            tune.with_parameters(Learn.__start, param=param),
            local_dir=directory_experiment_ray,
            trial_dirname_creator=Learn.trail_dirname_creator,
            resources_per_trial={"cpu": 8, "gpu": 1},
            config=config,
            metric="loss",
            mode="min",
            num_samples=Constants.num_samples,
            scheduler=scheduler,
            verbose=0
        )

        best_trial = result.get_best_trial("loss", "min", "last")

        print("Best trial config: {}".format(best_trial.config))
        print("Best trial for {} model final validation loss: {}".format(
            model.name, best_trial.last_result["loss"]))

    def Run():
        r"""
            Run Method. Starts the learning procedure for all models and datasets
        """
        Folders().CreateFolders()
        datareader = DataReader()

        experiment_name = "Experiment_{0}".format(
            datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
        directory_experiment_ray = os.path.join(
            Folders.results_ray_path, experiment_name)

        if not os.path.exists(directory_experiment_ray):
            os.makedirs(directory_experiment_ray)

        Learn.set_data(datareader=datareader)

        Learn.startLR(datareader=datareader, experiment_name=experiment_name)

        for datasize in DatasetSize:
            for model in ModelType:
                if model != ModelType.LinearRegression:
                    Learn.HyperParameterTuning(
                        datasetsize=datasize, model=model, datareader=datareader, experiment_name=experiment_name)
    #endregion
