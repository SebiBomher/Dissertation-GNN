from torch.utils.data.dataloader import DataLoader
from Scripts.models import CustomModel, STConvModel
from torch_geometric.data.data import Data
from Scripts.learn import LossFunction
import pandas as pd
import torch
from Scripts.data_proccess import DataReader, DatasetSize,Graph,DatasetSizeNumber
from Scripts.datasetsClasses import CustomDataset, LinearRegressionDataset, STConvDataset
import glob
import os
import pickle
from pathlib import Path
from torch.optim import Adam,RMSprop,Adamax,AdamW

class TestLog():
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
        self.__test_and_log()

    def __test_and_log_LR(self) -> None:
        dfResults = pd.DataFrame(columns=["Node_Id", "Criterion", "Loss"])
        self.LinearRegression = LinearRegressionDataset(self.proccessed_data_path, self.datareader, self.device)
        LRFiles = os.path.join(self.checkpoint_LR, "*.pickle")
        for file in glob.glob(LRFiles):
            filename = Path(file).stem
            info = filename.split('_')
            node_id = (int)(info[2])
            with open(file, 'r') as f:
                LRModel = pickle.load(f)
                X_test,Y_test = self.LinearRegression.get_for_node(node_id)
                y_pred = LRModel.predict(X_test)
                for criterion in LossFunction.Criterions:
                    loss = criterion(torch.FloatTensor(Y_test).to(self.device),torch.FloatTensor(y_pred).to(self.device))
                    loss = loss.item()
                    dfResults = dfResults.append(
                        {"Node_Id" : node_id, 
                        "Criterion" : criterion.__name__, 
                        "Loss" : str(loss)
                        },ignore_index=True)
        file_save = os.path.join(self.results_folder, "LinearRegression.csv")
        dfResults.to_csv(path_or_buf=file_save, index=False)
        return True

    def __test_and_log_STCONV(self) -> None:
        #TODO
        dfResults = pd.DataFrame(columns=["Epsilon", "Lamda", "Size", "Criterion","Loss"])
        for node_size in DatasetSize:
            folders_results = os.path.join(self.results_ray,node_size.value,"*")
            for directory in glob.glob(folders_results):
                if os.path.isdir(directory):
                    foldername = Path(directory).stem.split(',')
                    optimizer = foldername[5]
                    epsilon =(float)(foldername[2])
                    lamda = (int)(foldername[4])
                    _, _, self.test_dataset_STCONV = STConvDataset.get_dataset_STCONV(
                                                                                    path_proccessed_data=self.proccessed_data_path,
                                                                                    train_ratio = self.train_ratio,
                                                                                    test_ratio = self.test_ratio,
                                                                                    val_ratio = self.val_ratio,
                                                                                    batch_size=self.batch_size,
                                                                                    time_steps=1,
                                                                                    epsilon=epsilon,
                                                                                    lamda=lamda,
                                                                                    nodes_size=node_size,
                                                                                    datareader= self.datareader,
                                                                                    device= self.device)
                    checkpoints = os.path.join(folders_results,"ckecpoint_*")
                    for checkpoint in checkpoints:
                        model = STConvModel(node_features = 2,
                                num_nodes = Graph.get_number_nodes_by_size(node_size),
                                hidden_channels = 8,
                                kernel_size = 1,
                                K = 1)

                        if optimizer == "Adam":
                            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
                        elif optimizer ==  "RMSprop":
                            self.optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate)
                        elif optimizer == "Adamax":
                            self.optimizer = Adamax(self.model.parameters(), lr=self.learning_rate)
                        elif optimizer == "AdamW":
                            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

                        model_state, optimizer_state = torch.load(os.path.join(checkpoint, "checkpoint"))
                        model.load_state_dict(model_state)
                        optimizer.load_state_dict(optimizer_state)
                        loss = self.__test(model)
                        for criterion in LossFunction.Criterions:
                            loss = self.__test(model,criterion)
                            dfResults = dfResults.append(
                                {"Epsilon" : str(epsilon), 
                                "Lamda" : str(lamda), 
                                "Size" : str(node_size.name),
                                "Criterion" : str(criterion.__name__),
                                "Loss" : str(loss)
                                },ignore_index=True)

        file_save = os.path.join(self.results_folder, "STCONV.csv")
        dfResults.to_csv(path_or_buf=file_save, index=False)

    def __test_and_log_CUSTOM(self) -> None:
        # TODO
        dfResults = pd.DataFrame(columns=["Epsilon", "Lamda", "Size", "Criterion", "Loss"])
        for node_size in DatasetSize:
            folders_results = os.path.join(self.results_ray,node_size.value,"*")
            for directory in glob.glob(folders_results):
                if os.path.isdir(directory):
                    foldername = Path(directory).stem.split(',')
                    optimizer = foldername[5]
                    epsilon =(float)(foldername[2])
                    lamda = (int)(foldername[4])
                    _, _, test_dataset_CUSTOM = CustomDataset.get_dataset_Custom(
                                                                                    path_proccessed_data=self.proccessed_data_path,
                                                                                    train_ratio = self.train_ratio,
                                                                                    test_ratio = self.test_ratio,
                                                                                    val_ratio = self.val_ratio,
                                                                                    epsilon=epsilon,
                                                                                    lamda=lamda,
                                                                                    nodes_size=node_size,
                                                                                    datareader= self.datareader,
                                                                                    device= self.device)

                    checkpoints = os.path.join(folders_results,"ckecpoint_*")
                    for checkpoint in checkpoints:
                        model = CustomModel(node_features = self.num_features, K = 3)

                        if optimizer == "Adam":
                            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
                        elif optimizer ==  "RMSprop":
                            self.optimizer = RMSprop(self.model.parameters(), lr=self.learning_rate)
                        elif optimizer == "Adamax":
                            self.optimizer = Adamax(self.model.parameters(), lr=self.learning_rate)
                        elif optimizer == "AdamW":
                            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

                        model_state, optimizer_state = torch.load(os.path.join(checkpoint, "checkpoint"))
                        model.load_state_dict(model_state)
                        optimizer.load_state_dict(optimizer_state)
                        for criterion in LossFunction.Criterions:
                            loss = self.__test(model,criterion)
                            dfResults = dfResults.append(
                                {"Epsilon" : str(epsilon), 
                                "Lamda" : str(lamda), 
                                "Size" : str(node_size.name),
                                "Criterion" : str(criterion.__name__),
                                "Loss" : str(loss)
                                },ignore_index=True)
                        
        file_save = os.path.join(self.results_folder, "CUSTOM.csv")
        dfResults.to_csv(path_or_buf=file_save, index=False)

        def __test(self,best_model,criterion):
            best_model.eval()
            loss = 0
            dataloader = DataLoader(test_dataset_CUSTOM,batch_size = 1,shuffle=False,num_workers=0)
            edge_index = test_dataset_CUSTOM.get_edge_index()
            edge_weight = test_dataset_CUSTOM.get_edge_weight()
            iter = 0
            for time, (x,y) in enumerate(dataloader):
                X = x[0]
                Y = y[0]
                y_hat = best_model(X, edge_index, edge_weight)
                loss += criterion(y_hat,Y)
                iter +=1

            loss = loss / (iter+1)
            loss = loss.item()
            return loss

    def __test_and_log(self) -> None:
        self.__test_and_log_LR()
        self.__test_and_log_STCONV()
        self.__test_and_log_CUSTOM()
