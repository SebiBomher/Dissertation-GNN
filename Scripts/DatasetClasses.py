#region Imports

import math
import os
import torch
import numpy as np
import glob
from Scripts.DataProccess import DataReader, Graph
from Scripts.Utility import Constants, DatasetSize, DatasetSizeNumber, Folders
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#endregion


class DatasetClass(object):
    r"""
        Helping class so there is no need to implement the same variables in the constructors and the graph set and get
    """
    #region Constructor & Properties

    def __init__(self,
                 sigma: int,
                 epsilon: float,
                 size: DatasetSize,
                 datareader: DataReader,
                 device: str = 'cpu',
                 time_start: int = 0,
                 time_stop: float = -1):
        r"""
    
        """
        self.proccessed_data_path = Folders.proccessed_data_path
        self.sigma = sigma
        self.epsilon = epsilon
        self.time_start = time_start
        self.time_stop = time_stop
        self.data_reader = datareader
        self.size = size
        self.device = device
        self.__set_graph()

    #endregion

    #region Instance Functions

    def __set_graph(self):
        r"""
            Function to set the graph which contains the edge indices and edge weights
            Instance Function.
            No Arguments.
            Returns None.
        """
        self.graph = Graph(self.epsilon, self.sigma,
                           self.size, self.data_reader)

    def get_edge_index(self):
        r"""
            Function to get the edge index of the setted graph
            Instance Function.
            No Arguments.
            Returns None.
        """
        if self.graph.edge_index is None:
            return self.graph.edge_index
        else:
            return torch.LongTensor(self.graph.edge_index).to(self.device)

    def get_edge_weight(self):
        r"""
            Function to get the edge weight of the setted graph
            Instance Function.
            No Arguments.
            Returns None.
        """
        if self.graph.edge_weight is None:
            return self.graph.edge_weight
        else:
            return torch.FloatTensor(self.graph.edge_weight).to(self.device)

    #endregion


class LinearRegressionDataset():
    r"""
        Class for Linear Regression data manipulation
        Instance functions:
        Class Functions:
    """
    #region Constructor & Properties

    def __init__(self,
                 datareader: DataReader):
        r"""
            Constructor, Receives a datareader
        """
        self.proccessed_data_path = Folders.proccessed_data_path
        self.proccessed_data_path_model = os.path.join(
            self.proccessed_data_path, "LinearRegression")
        self.datareader = datareader
        self.device = Constants.device

    #endregion

    #region Instance Functions

    def __getitem__(self, time_index: int):
        r"""
            Function for iterator to get the current item in a collection.
            Args:
                time_index : int, index in the collection
            Instance Function
            Returns a tuple of data for train and test, labels for train and test and the node id
        """
        name_x = os.path.join(self.proccessed_data_path_model,
                              "Data", 'X_{0}*.npy'.format(str(time_index)))
        name_y = os.path.join(self.proccessed_data_path_model,
                              "Data", 'Y_{0}*.npy'.format(str(time_index)))

        for filename in glob.glob(name_x):
            node_id = filename[-10:-4]
            X = np.load(filename)
        for filename in glob.glob(name_y):
            Y = np.load(filename)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=False)
        return X_train, X_test, Y_train, Y_test, node_id

    def __next__(self):
        r"""
            Function to return the next item in the collecion
            Instance Function
            Returns the next __getitem__
        """
        if self.t < DatasetSizeNumber.All.value:
            snapshot = self.__getitem__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        r"""
            Iterator initializer
            Instance Function
            Returns self
        """
        self.t = 0
        return self

    def get_for_node(self, node: int):
        r"""
            Function which returns test data and test labels for a specific node
        """
        assert node % math.pow(10, 6) >= 1
        name_x = os.path.join(self.proccessed_data_path_model,
                              "Data", 'X_*_{0}.npy'.format(str(node)))
        name_y = os.path.join(self.proccessed_data_path_model,
                              "Data", 'Y_*_{0}.npy'.format(str(node)))
        for filename in glob.glob(name_x):
            X = np.load(filename)
        for filename in glob.glob(name_y):
            Y = np.load(filename)
        _, X_test, _, Y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=False)
        return X_test, Y_test

    #endregion

    #region Class Functions

    def __arrange_data(data, num_nodes):
        r"""
            Class Function to arrange from raw data to ordered data such that in each item there is information for a row
        """
        New_Data = []
        for i in range(num_nodes):
            Data = []
            for k in range((int)(len(data)/num_nodes)):
                Data.append(data[i + (k * num_nodes)])
            New_Data.append(Data)
        return New_Data

    def save_dataset(datareader: DataReader):
        r"""
            Function which saves data for easier go through at train time
        """
        if not LinearRegressionDataset.need_load():
            return
        proccessed_data_path_model = os.path.join(
            Folders.proccessed_data_path, "LinearRegression")

        X, Y = datareader.get_clean_data_by_nodes(DatasetSize.All)

        X = LinearRegressionDataset.__arrange_data(
            X, DatasetSizeNumber.All.value)
        Y = LinearRegressionDataset.__arrange_data(
            Y, DatasetSizeNumber.All.value)

        nodes_ids = Graph.get_nodes_ids_by_size(DatasetSize.All)

        if not os.path.exists(proccessed_data_path_model):
            os.makedirs(proccessed_data_path_model)

        name_folder = os.path.join(proccessed_data_path_model, 'Data')

        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

        for index, data in enumerate(zip(X, nodes_ids)):
            name_x = os.path.join(
                name_folder, 'X_{0}_{1}.npy'.format(str(index), str(data[1])))
            np.save(name_x, data[0])

        for index, data in enumerate(zip(Y, nodes_ids)):
            name_y = os.path.join(
                name_folder, 'Y_{0}_{1}.npy'.format(str(index), str(data[1])))
            np.save(name_y, data[0])

        return

    def need_load():
        r"""
            Function to determine wheter to start saving data
        """
        return not os.path.exists(os.path.join(os.path.join(Folders.proccessed_data_path, "LinearRegression"), 'Data'))

    def get_previous_node_for_node_with_LR(datareader : DataReader, node : int, datasetsize: DatasetSize):
        
        dataset = LinearRegressionDataset(datareader)
        nodes_ids = Graph.get_nodes_ids_by_size(datasetsize)
        data_list = []
        nodes_used = []
        for _, (X_train, X_test, Y_train, Y_test, node_id) in enumerate(dataset):
            node_id = (int)(node_id)
            Y_train = [item for sublist in Y_train for item in sublist]
            if node_id in nodes_ids and node != node_id:
                data_list.append(Y_train)
                nodes_used.append(node_id)
            if node == node_id:
                labels_node = Y_train
        data_list = np.transpose(data_list)
        regression = LinearRegression(positive = True).fit(data_list, labels_node)
        coeffiecients = regression.coef_.tolist()
        results = zip(nodes_used, coeffiecients)
        sorted_results = sorted(results, key=lambda tup: tup[1],reverse = True)
        best = sorted_results[:3]
        return [result[0] for result in best]

    def set_graph_with_LR(datareader : DataReader,size : DatasetSize):
        Folders().CreateFolders()
        nodes_ids = Graph.get_nodes_ids_by_size(size).tolist()
        edge_index = []
        edge_weight = []
        for node in nodes_ids:
            nodes_relevant = LinearRegressionDataset.get_previous_node_for_node_with_LR(datareader,node,size)
            for node_relevant in nodes_relevant:
                edge_index.append([nodes_ids.index(node_relevant),nodes_ids.index(node)])
        edge_index = [list(x) for x in set(tuple(x) for x in edge_index)]
        edge_weight = np.ones(len(edge_index))
        edge_index = np.transpose(edge_index)

        name_folder_weight = os.path.join(Folders.proccessed_data_path,'Data_EdgeWeight')
        name_folder_index = os.path.join(Folders.proccessed_data_path,'Data_EdgeIndex')

        if not os.path.exists(name_folder_weight):
            os.makedirs(name_folder_weight)

        if not os.path.exists(name_folder_index):
            os.makedirs(name_folder_index)

        name_weight = os.path.join(name_folder_weight,'weight_{0}LR.npy'.format(str(size.name)))
        name_index = os.path.join(name_folder_index,'index_{0}LR.npy'.format(str(size.name)))
        np.save(name_index,edge_index)
        np.save(name_weight,edge_weight)

    #endregion


class LSTMDataset(DatasetClass):
    r"""
        Class for data manipulation for the LSTM model
    """

    #region Constructor & Properites

    def __init__(self,
                 sigma: int,
                 epsilon: float,
                 size: DatasetSize,
                 datareader: DataReader,
                 device: str = 'cpu',
                 time_start: int = 0,
                 time_stop: float = -1):
        r"""
            Constructor. It also uses base class intialization.
        """
        super().__init__(sigma, epsilon, size, datareader, device, time_start, time_stop)
        self.proccessed_data_path_model = os.path.join(
            self.proccessed_data_path, "LSTM")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.__check_temporal_consistency()
        self.__set_snapshot_count()

    #endregion

    #region Instance Functions

    def __check_temporal_consistency(self):
        r"""
            Function to check if the data and labels have the same number and there are not discrepancies.
        """
        assert len(glob.glob1(os.path.join(self.proccessed_data_path_model, "Data_{0}".format(str(self.size.name))), "X_*.npy")) == len(glob.glob1(
            os.path.join(self.proccessed_data_path_model, "Data_{0}".format(str(self.size.name))), "Y_*.npy")), "Temporal dimension inconsistency."

    def __set_snapshot_count(self):
        r"""
            Sets the dataset length
        """
        self.snapshot_count = len(glob.glob1(os.path.join(
            self.proccessed_data_path_model, "Data_{0}".format(str(self.size.name))), "X_*.npy"))

    def __split_dataset(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        r"""
            Dataset splitter. Splits dataset based on ratios into train validation and test
        """
        assert train_ratio + test_ratio + val_ratio == 1
        time_train = int(train_ratio*self.snapshot_count)
        time_test = time_train + int(test_ratio*self.snapshot_count)

        train_iterator = LSTMDataset(self.sigma,
                                       self.epsilon,
                                       self.size,
                                       self.data_reader,
                                       self.device,
                                       0,
                                       time_train + 1)

        val_iterator = LSTMDataset(self.sigma,
                                     self.epsilon,
                                     self.size,
                                     self.data_reader,
                                     self.device,
                                     time_train + 1,
                                     time_test)

        test_iterator = LSTMDataset(self.sigma,
                                      self.epsilon,
                                      self.size,
                                      self.data_reader,
                                      self.device,
                                      time_test + 1,
                                      self.snapshot_count)

        return train_iterator, val_iterator, test_iterator

    def __arrange_data(data, num_nodes):
        r"""
            Function to arrange data. At a point it represents the temporal state of the graph
        """
        New_Data = []
        for i in range((int)(len(data)/num_nodes)):
            Data = []
            for k in range(num_nodes):
                Data.append(data[(i * num_nodes) + k])
            New_Data.append(Data)
        return New_Data

    def __save_proccess_data(datareader: DataReader, size: DatasetSize):
        r"""
            Function which saves data for easier go through at train time
        """
        print("Saving data with configuration : size = {0}".format(
            str(size.name)))

        X, Y = datareader.get_clean_data_by_nodes(size)

        X = LSTMDataset.__arrange_data(
            X, Graph.get_number_nodes_by_size(size))
        Y = LSTMDataset.__arrange_data(
            Y, Graph.get_number_nodes_by_size(size))
        proccessed_data_path_model = os.path.join(
            Folders.proccessed_data_path, "LSTM")
        if not os.path.exists(proccessed_data_path_model):
            os.makedirs(proccessed_data_path_model)

        name_folder = os.path.join(
            proccessed_data_path_model, 'Data_{0}'.format(str(size.name)))
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

        for index, data in enumerate(X):
            name_x = os.path.join(name_folder, 'X_{0}.npy'.format(str(index)))
            np.save(name_x, data)

        for index, data in enumerate(Y):
            name_y = os.path.join(name_folder, 'Y_{0}.npy'.format(str(index)))
            np.save(name_y, data)

    def __get_features(self, time_index: int):
        r"""
            Function to get data at a time index
        """
        name_x = os.path.join(self.proccessed_data_path_model, "Data_{0}".format(
            str(self.size.name)), 'X_{0}.npy'.format(str(time_index)))
        X = np.load(name_x)
        if X is None:
            return X
        else:
            return torch.FloatTensor(X).to(self.device)

    def __get_target(self, time_index: int):
        r"""
            Function to get labels at a time index
        """
        name_y = os.path.join(self.proccessed_data_path_model, "Data_{0}".format(
            str(self.size.name)), 'Y_{0}.npy'.format(str(time_index)))
        Y = np.load(name_y)
        if Y is None:
            return Y
        else:
            if Y.dtype.kind == 'i':
                return torch.LongTensor(Y).to(self.device)
            elif Y.dtype.kind == 'f':
                return torch.FloatTensor(Y).to(self.device)

    def __getitem__(self, time_index: int):
        r"""
            Function to get featuers and target and a time index
        """
        x = self.__get_features(time_index)
        y = self.__get_target(time_index)
        return x, y

    def __next__(self):
        r"""
            Function to iterate the next item in the collection
        """
        if self.t < self.time_stop:
            snapshot = self.__getitem__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = self.time_start
            raise StopIteration

    def __iter__(self):
        r"""
            Iterator construcor
        """
        self.t = self.time_start
        return self

    def __len__(self):
        """
            Iterator length
        """
        return self.snapshot_count

    #endregion

    #region Class Functions

    def need_load(proccessed_data_path):
        r"""
            Function to determine wheter to start saving data
        """
        return len(LSTMDataset.__get_tuple_to_add(os.path.join(proccessed_data_path, "LSTM"))) > 0

    def get_dataset_LSTM(train_ratio: float, test_ratio: float, val_ratio: float, epsilon: float, sigma: int, nodes_size: DatasetSize, datareader: DataReader, device: str):
        r"""
            Function used in Learn to get train validation and test datasets for training
        """
        DataTraffic = LSTMDataset(
            sigma, epsilon, nodes_size, datareader, device)
        train, val, test = DataTraffic.__split_dataset(
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        return train, val, test

    def __get_tuple_to_add(proccessed_data_path):
        r"""
            Function to retrieve what data is missing in order for it to be saved
        """
        to_create = []
        for size in DatasetSize:
            if size != DatasetSize.All:
                if size != DatasetSize.TinyManual or size != DatasetSize.ExperimentalManual:
                    name_folder = os.path.join(
                        proccessed_data_path, 'Data_{0}'.format(str(size.name)))
                if not os.path.exists(name_folder):
                    to_create.append([size])
        return to_create

    def save_dataset(data_reader: DataReader):
        r"""
            Function to save the dataset
        """
        proccessed_data_path_model = os.path.join(
            Folders.proccessed_data_path, "LSTM")
        to_create = LSTMDataset.__get_tuple_to_add(
            proccessed_data_path_model)
        for tuple in to_create:
            size = tuple[0]
            LSTMDataset.__save_proccess_data(data_reader, size)

    #endregion


class STConvDataset(DatasetClass):

    def __init__(self,
                 sigma: int,
                 epsilon: float,
                 size: DatasetSize,
                 datareader: DataReader,
                 device: str = 'cpu',
                 time_start: int = 0,
                 time_stop: float = -1):

        super().__init__(sigma, epsilon, size, datareader, device, time_start, time_stop)
        self.proccessed_data_path_STCONV = os.path.join(
            self.proccessed_data_path, "STCONV")
        self.__check_temporal_consistency()
        self.__set_snapshot_count()

    def __check_temporal_consistency(self):
        assert len(glob.glob1(os.path.join(self.proccessed_data_path_STCONV, "Data_{0}".format(str(self.size.name))), "X_*.npy")) == len(glob.glob1(
            os.path.join(self.proccessed_data_path_STCONV, "Data_{0}".format(str(self.size.name))), "Y_*.npy")), "Temporal dimension inconsistency."

    def need_load(proccessed_data_path):
        return len(STConvDataset.__get_tuple_to_add(os.path.join(proccessed_data_path, "STCONV"))) > 0

    def get_dataset_STCONV(train_ratio: float, test_ratio: float, val_ratio: float, epsilon: float, sigma: int, nodes_size: DatasetSize, datareader: DataReader, device: str):
        DataTraffic = STConvDataset(sigma, epsilon, nodes_size, datareader, device)
        train, val, test = DataTraffic.__split_dataset(
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        return train, val, test

    def __set_snapshot_count(self):
        self.snapshot_count = len(glob.glob1(os.path.join(self.proccessed_data_path_STCONV, "Data_{0}".format(str(self.size.name))), "X_*.npy"))

    def __split_dataset(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        assert train_ratio + test_ratio + val_ratio == 1
        time_train = int(train_ratio*self.snapshot_count)
        time_test = time_train + int(test_ratio*self.snapshot_count)

        train_iterator = STConvDataset(self.sigma,
                                       self.epsilon,
                                       self.size,
                                       self.data_reader,
                                       self.device,
                                       0,
                                       time_train + 1)

        test_iterator = STConvDataset(self.sigma,
                                      self.epsilon,
                                      self.size,
                                      self.data_reader,
                                      self.device,
                                      time_train + 1,
                                      time_test)

        val_iterator = STConvDataset(self.sigma,
                                     self.epsilon,
                                     self.size,
                                     self.data_reader,
                                     self.device,
                                     time_test + 1,
                                     self.snapshot_count)

        return train_iterator, test_iterator, val_iterator

    def __save_proccess_data(datareader: DataReader, size: DatasetSize):
        batch_size = Constants.batch_size
        time_steps = Constants.time_steps

        print("Saving data with configuration : size = {0}".format(
            str(size.name)))

        interval_per_day = datareader.interval_per_day
        Skip = (int)(time_steps/2) * 2
        interval_per_day -= Skip
        X, Y = datareader.get_clean_data_by_nodes(size)

        nodes_size = Graph.get_number_nodes_by_size(size)

        X = STConvDataset.__arrange_data_for_time_step(
            X, time_steps, nodes_size)
        Y = STConvDataset.__arrange_data_for_time_step(
            Y, time_steps, nodes_size)
        
        new_size = (int)(interval_per_day * datareader.nb_days / batch_size)
        data_size = len(X)
        if new_size * batch_size != data_size:
            difference = data_size-((new_size - 1) * batch_size)
            X = X[:data_size-difference]
            Y = Y[:data_size-difference]
            new_size -= 1
        X = np.array(X).reshape(new_size, batch_size,
                                time_steps, nodes_size, 2)
        Y = np.array(Y).reshape(new_size, batch_size,
                                time_steps, nodes_size, 1)

        name_folder = os.path.join(
            Folders.proccessed_data_path, 'STCONV', 'Data_{0}'.format(str(size.name)))
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

        for index, data in enumerate(X):
            name_x = os.path.join(name_folder, 'X_{0}.npy'.format(str(index)))
            np.save(name_x, data)

        for index, data in enumerate(Y):
            name_y = os.path.join(name_folder, 'Y_{0}.npy'.format(str(index)))
            np.save(name_y, data)

    def __arrange_data_for_time_step(data, time_steps, num_nodes):
        Skip = (int)(time_steps/2)
        New_Data = []
        for i in range(Skip, (int)((len(data) - (num_nodes * Skip))/(num_nodes))):
            Graph = []
            for j in range(-Skip, Skip + 1):
                Data = []
                for k in range(num_nodes):
                    Data.append(data[(i * k) + k + (num_nodes * j)])
                Graph.append(Data)
            New_Data.append(Graph)
        return New_Data

    def __get_tuple_to_add(proccessed_data_path):
        to_create = []
        for size in DatasetSize:
            if size != DatasetSize.All:
                if size != DatasetSize.TinyManual or size != DatasetSize.ExperimentalManual:
                    name_folder = os.path.join(proccessed_data_path, 'Data_{0}'.format(
                        str(size.name)))
                if not os.path.exists(name_folder):
                    to_create.append([size])
        return to_create

    def save_dataset(data_reader: DataReader):
        proccessed_data_path_STCONV = os.path.join(
            Folders.proccessed_data_path, "STCONV")
        to_create = STConvDataset.__get_tuple_to_add(
            proccessed_data_path_STCONV)
        for tuple in to_create:
            size = tuple[0]
            STConvDataset.__save_proccess_data(data_reader, size)

    def __get_features(self, time_index: int):
        name_x = os.path.join(self.proccessed_data_path_STCONV, "Data_{0}".format(
            str(self.size.name)), 'X_{0}.npy'.format(str(time_index)))
        X = np.load(name_x)
        if X is None:
            return X
        else:
            return torch.FloatTensor(X).to(self.device)

    def __get_target(self, time_index: int):
        name_y = os.path.join(self.proccessed_data_path_STCONV, "Data_{0}".format(
            str(self.size.name)), 'Y_{0}.npy'.format(str(time_index)))
        Y = np.load(name_y)
        if Y is None:
            return Y
        else:
            if Y.dtype.kind == 'i':
                return torch.LongTensor(Y).to(self.device)
            elif Y.dtype.kind == 'f':
                return torch.FloatTensor(Y).to(self.device)

    def __getitem__(self, time_index: int):
        x = self.__get_features(time_index)
        y = self.__get_target(time_index)
        return x, y

    def __next__(self):
        if self.t < self.time_stop:
            snapshot = self.__getitem__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = self.time_start
            raise StopIteration

    def __iter__(self):
        self.t = self.time_start
        return self

    def __len__(self):
        return self.snapshot_count


class ARIMADataset():
    def __init__():
        return


class RNNDataset():
    def __init__():
        return


class VARMAXDataset():
    def __init__():
        return
