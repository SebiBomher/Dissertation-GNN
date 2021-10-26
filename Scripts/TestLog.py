import pandas as pd
import torch
import glob
import os
import pickle
import json
from torch.utils.data.dataloader import DataLoader
from Scripts.Models import CustomModel, STConvModel
from torch_geometric.data.data import Data
from Scripts.Learn import LossFunction
from Scripts.DataProccess import DataReader, DatasetSize, Graph
from Scripts.DatasetClasses import CustomDataset, LinearRegressionDataset, STConvDataset
from pathlib import Path
from torch.optim import Adam, RMSprop, Adamax, AdamW
from Scripts.Utility import Folders


class TestLog():
    """
        Function to test 3 models (Linear Regression, STCONV, Custom) and log results
        Functions:
            __test_and_log_LR
            __test_and_log_STCONV
            __test_and_log_CUSTOM
    """

    def __init__(self,
                 datareader: DataReader,
                 proccessed_data_path: str,
                 checkpoint_LR: str,
                 results_ray: str,
                 results_folder: str
                 ):
        self.datareader = datareader
        self.proccessed_data_path = proccessed_data_path
        self.checkpoint_LR = checkpoint_LR
        self.results_ray = results_ray
        self.results_folder = results_folder
        self.device = "cpu"
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.test_ratio = 0.2
        self.batch_size = 8
        self.learning_rate = 0.01
        self.num_features = 2
        self.__test_and_log()
    """
        Function to test and log the Linear Regression model
    """

    def __test_and_log_LR(self) -> None:
        dfResults = pd.DataFrame(columns=["Node_Id", "Criterion", "Loss"])
        self.LinearRegression = LinearRegressionDataset(
            self.proccessed_data_path, self.datareader, self.device)
        LRFiles = os.path.join(self.checkpoint_LR, "*.pickle")
        for file in glob.glob(LRFiles):
            filename = Path(file).stem
            info = filename.split('_')
            node_id = (int)(info[2])
            with open(file, 'rb') as f:
                LRModel = pickle.load(f)
                X_test, Y_test = self.LinearRegression.get_for_node(node_id)
                y_pred = LRModel.predict(X_test)
                for criterion in LossFunction.Criterions():
                    loss = criterion(torch.FloatTensor(Y_test).to(
                        self.device), torch.FloatTensor(y_pred).to(self.device))
                    loss = loss.item()
                    result = {"Node_Id": node_id,
                              "Criterion": criterion.__name__, "Loss": str(loss)}
                    print(result)
                    dfResults = dfResults.append(result, ignore_index=True)
        file_save = os.path.join(self.results_folder, "LinearRegression.csv")
        print(file_save)
        dfResults.to_csv(path_or_buf=file_save, index=False)
        return True
    """
        Function to test and log the STConv model
    """

    def __test_and_log_STCONV(self) -> None:
        #TODO
        dfResults = pd.DataFrame(
            columns=["Epsilon", "sigma", "Size", "Criterion", "Loss", "OptimizerType", "Checkpoint","Trial"])
        for node_size in DatasetSize:
            if node_size == DatasetSize.Experimental:
                continue
            folders_results = os.path.join(
                self.results_ray, "STCONV_{0}".format(str(node_size.name)), "*")
            index = 0
            for directory in glob.glob(folders_results):
                if os.path.isdir(directory):
                    index +=1
                    params_json = os.path.join(directory, "params.json")
                    with open(params_json, "r") as json_file:
                        data = json.load(json_file)
                        epsilon = (float)(data["epsilon"])
                        sigma = (int)(data["sigma"])
                        optimizer_type = (data["optimizer_type"])
                    _, _, test_dataset_STCONV = STConvDataset.get_dataset_STCONV(
                        path_proccessed_data=self.proccessed_data_path,
                        train_ratio=self.train_ratio,
                        test_ratio=self.test_ratio,
                        val_ratio=self.val_ratio,
                        batch_size=self.batch_size,
                        time_steps=1,
                        epsilon=epsilon,
                        sigma=sigma,
                        nodes_size=node_size,
                        datareader=self.datareader,
                        device=self.device)
                    checkpoints = os.path.join(directory, "checkpoint_*")
                    for checkpoint in glob.glob(checkpoints):
                        checkpoint_index = Path(checkpoint).stem.split("_")[1]
                        model = STConvModel(node_features=self.num_features,
                                            num_nodes=Graph.get_number_nodes_by_size(
                                                node_size),
                                            hidden_channels=8,
                                            kernel_size=1,
                                            K=1)

                        if optimizer_type == "OptimiserType.Adam":
                            optimizer = Adam(
                                model.parameters(), lr=self.learning_rate)
                        elif optimizer_type == "OptimiserType.RMSprop":
                            optimizer = RMSprop(
                                model.parameters(), lr=self.learning_rate)
                        elif optimizer_type == "OptimiserType.Adamax":
                            optimizer = Adamax(
                                model.parameters(), lr=self.learning_rate)
                        elif optimizer_type == "OptimiserType.AdamW":
                            optimizer = AdamW(
                                model.parameters(), lr=self.learning_rate)

                        model_state, optimizer_state = torch.load(
                            os.path.join(checkpoint, "checkpoint"))
                        model.load_state_dict(model_state)
                        optimizer.load_state_dict(optimizer_state)

                        for criterion in LossFunction.Criterions():
                            loss = self.__test(model, criterion, test_dataset_STCONV)
                            results = {"Epsilon": str(epsilon),
                                       "sigma": str(sigma),
                                       "Size": str(node_size.name),
                                       "Criterion": str(criterion.__name__),
                                       "Loss": str(loss),
                                       "OptimizerType": optimizer_type,
                                       "Checkpoint": checkpoint_index,
                                       "Trial": index
                                       }
                            print(results)
                            dfResults = dfResults.append(
                                results, ignore_index=True)

                    file_save = os.path.join(self.results_folder, "STCONV_{0}_{1}.csv".format(str(node_size.name),index))
                    dfResults.to_csv(path_or_buf=file_save, index=False)
                    dfResults = dfResults[0:0]

    """
        Function to test and log the Custom model
    """

    def __test_and_log_CUSTOM(self) -> None:
        # TODO
        dfResults = pd.DataFrame(
            columns=["Epsilon", "sigma", "Size", "Criterion", "Loss", "OptimizerType", "Checkpoint","Trial"])
        for node_size in DatasetSize:
            folders_results = os.path.join(
                self.results_ray, "CUSTOM_{0}".format(str(node_size.name)), "*")
            index = 0
            for directory in glob.glob(folders_results):
                if os.path.isdir(directory):
                    index +=1
                    params_json = os.path.join(directory, "params.json")
                    with open(params_json, "r") as json_file:
                        data = json.load(json_file)
                        epsilon = (float)(data["epsilon"])
                        sigma = (int)(data["sigma"])
                        optimizer_type = (data["optimizer_type"])

                    _, _, test_dataset_CUSTOM = CustomDataset.get_dataset_Custom(
                        path_proccessed_data=self.proccessed_data_path,
                        train_ratio=self.train_ratio,
                        test_ratio=self.test_ratio,
                        val_ratio=self.val_ratio,
                        epsilon=epsilon,
                        sigma=sigma,
                        nodes_size=node_size,
                        datareader=self.datareader,
                        device=self.device)

                    checkpoints = os.path.join(directory, "checkpoint_*")
                    for checkpoint in glob.glob(checkpoints):
                        checkpoint_index = Path(checkpoint).stem.split("_")[1]

                        model = CustomModel(
                            node_features=self.num_features, K=3)

                        if optimizer_type == "OptimiserType.Adam":
                            optimizer = Adam(
                                model.parameters(), lr=self.learning_rate)
                        elif optimizer_type == "OptimiserType.RMSprop":
                            optimizer = RMSprop(
                                model.parameters(), lr=self.learning_rate)
                        elif optimizer_type == "OptimiserType.Adamax":
                            optimizer = Adamax(
                                model.parameters(), lr=self.learning_rate)
                        elif optimizer_type == "OptimiserType.AdamW":
                            optimizer = AdamW(
                                model.parameters(), lr=self.learning_rate)

                        model_state, optimizer_state = torch.load(
                            os.path.join(checkpoint, "checkpoint"))
                        model.load_state_dict(model_state)
                        optimizer.load_state_dict(optimizer_state)

                        for criterion in LossFunction.Criterions():
                            loss = self.__test(model, criterion,test_dataset_CUSTOM)
                            results = {"Epsilon": str(epsilon),
                                       "sigma": str(sigma),
                                       "Size": str(node_size.name),
                                       "Criterion": str(criterion.__name__),
                                       "Loss": str(loss),
                                       "OptimizerType": optimizer_type,
                                       "Checkpoint": checkpoint_index,
                                       "Trial": index
                                       }
                            print(results)
                            dfResults = dfResults.append(
                                results, ignore_index=True)

                    file_save = os.path.join(self.results_folder, "CUSTOM_{0}_{1}.csv".format(str(node_size.name),index))
                    dfResults.to_csv(path_or_buf=file_save, index=False)
                    dfResults = dfResults[0:0]

    def __test(self, model, criterion, dataset):
        model.eval()
        loss = 0
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0)
        edge_index = dataset.get_edge_index()
        edge_weight = dataset.get_edge_weight()
        iter = 0
        for time, (x, y) in enumerate(dataloader):
            X = x[0]
            Y = y[0]
            y_hat = model(X, edge_index, edge_weight)
            loss += criterion(y_hat, Y)
            iter += 1

        loss = loss / (iter+1)
        loss = loss.item()
        return loss

    def __test_and_log(self) -> None:
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        self.__test_and_log_LR()
        self.__test_and_log_STCONV()
        self.__test_and_log_CUSTOM()

    def Run():
        datareader = DataReader(Folders.path_data, Folders.graph_info_path)
        TestLog(datareader, Folders.proccessed_data_path,Folders.checkpoint_LR_path, Folders.results_ray_path, Folders.results_path)