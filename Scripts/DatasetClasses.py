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
        self.batch_size = Constants.batch_size
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
        self.graph = Graph(self.proccessed_data_path, self.epsilon,
                           self.sigma, self.size, self.data_reader)

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
    
    """

    def __init__(self,
                 datareader: DataReader):
        self.proccessed_data_path = Folders.proccessed_data_path
        self.proccessed_data_path_model = os.path.join(
            self.proccessed_data_path, "LinearRegression")
        self.datareader = datareader
        self.device = Constants.device
        self.__save_dataset()

    def __arrange_data(data, num_nodes):
        r"""

        """
        New_Data = []
        for i in range(num_nodes):
            Data = []
            for k in range((int)(len(data)/num_nodes)):
                Data.append(data[i + (k * num_nodes)])
            New_Data.append(Data)
        return New_Data

    def __save_dataset(datareader):
        r"""
            
        """
        if not LinearRegressionDataset.need_load(Folders.proccessed_data_path):
            return
        proccessed_data_path_model = os.path.join(
            Folders.proccessed_data_path, "LinearRegression")
        X, Y = datareader.get_clean_data_by_nodes(
            DatasetSize.Medium, Folders.proccessed_data_path)

        X = LinearRegressionDataset.__arrange_data(
            X, DatasetSizeNumber.Medium.value)
        Y = LinearRegressionDataset.__arrange_data(
            Y, DatasetSizeNumber.Medium.value)

        nodes_ids = Graph.get_nodes_ids_by_size(
            Folders.proccessed_data_path, DatasetSize.Medium)

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

    def need_load(proccessed_data_path):
        r"""
            
        """
        return not os.path.exists(os.path.join(os.path.join(proccessed_data_path, "LinearRegression"), 'Data'))

    def __getitem__(self, time_index: int):
        r"""
            
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
            
        """
        if self.t < DatasetSizeNumber.Medium.value:
            snapshot = self.__getitem__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        r"""
            
        """
        self.t = 0
        return self

    def get_for_node(self, node: int):
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


class CustomDataset(DatasetClass):

    def __init__(self,
                 sigma: int,
                 epsilon: float,
                 size: DatasetSize,
                 datareader: DataReader,
                 device: str = 'cpu',
                 time_start: int = 0,
                 time_stop: float = -1):

        super().__init__(sigma, epsilon, size, datareader, device, time_start, time_stop)
        self.proccessed_data_path_model = os.path.join(
            self.proccessed_data_path, "Custom")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.__save_dataset()
        self.__check_temporal_consistency()
        self.__set_snapshot_count()

    def __check_temporal_consistency(self):
        assert len(glob.glob1(os.path.join(self.proccessed_data_path_model, "Data_{0}_{1}".format(str(self.batch_size), str(self.size.name))), "X_*.npy")) == len(glob.glob1(
            os.path.join(self.proccessed_data_path_model, "Data_{0}_{1}".format(str(self.batch_size), str(self.size.name))), "Y_*.npy")), "Temporal dimension inconsistency."

    def need_load(proccessed_data_path):
        return len(CustomDataset.__get_tuple_to_add(os.path.join(proccessed_data_path, "Custom"))) > 0

    def get_dataset_Custom(train_ratio: float, test_ratio: float, val_ratio: float, epsilon: float, sigma: int, nodes_size: DatasetSize, datareader: DataReader, device: str):
        DataTraffic = CustomDataset(
            sigma, epsilon, nodes_size, datareader, device)
        train, val, test = DataTraffic.__split_dataset(
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        return train, val, test

    def __set_snapshot_count(self):
        self.snapshot_count = len(glob.glob1(os.path.join(
            self.proccessed_data_path_model, "Data_{0}".format(str(self.size.name))), "X_*.npy"))

    def __split_dataset(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        assert train_ratio + test_ratio + val_ratio == 1
        time_train = int(train_ratio*self.snapshot_count)
        time_test = time_train + int(test_ratio*self.snapshot_count)

        train_iterator = CustomDataset(self.sigma,
                                       self.epsilon,
                                       self.size,
                                       self.data_reader,
                                       self.device,
                                       0,
                                       time_train + 1)

        val_iterator = CustomDataset(self.sigma,
                                     self.epsilon,
                                     self.size,
                                     self.data_reader,
                                     self.device,
                                     time_train + 1,
                                     time_test)

        test_iterator = CustomDataset(self.sigma,
                                      self.epsilon,
                                      self.size,
                                      self.data_reader,
                                      self.device,
                                      time_test + 1,
                                      self.snapshot_count)

        return train_iterator, val_iterator, test_iterator

    def __arrange_data(self, data, num_nodes):
        New_Data = []
        for i in range((int)(len(data)/num_nodes)):
            Data = []
            for k in range(num_nodes):
                Data.append(data[(i * num_nodes) + k])
            New_Data.append(Data)
        return New_Data

    def __save_proccess_data(self, datareader: DataReader, size: DatasetSize):
        print("Saving data with configuration : size = {0}".format(
            str(size.name)))

        X, Y = datareader.get_clean_data_by_nodes(
            size, self.proccessed_data_path)

        X = self.__arrange_data(X, Graph.get_number_nodes_by_size(size))
        Y = self.__arrange_data(Y, Graph.get_number_nodes_by_size(size))

        if not os.path.exists(self.proccessed_data_path_model):
            os.makedirs(self.proccessed_data_path_model)

        name_folder = os.path.join(
            self.proccessed_data_path_model, 'Data_{0}'.format(str(size.name)))
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

        for index, data in enumerate(X):
            name_x = os.path.join(name_folder, 'X_{0}.npy'.format(str(index)))
            np.save(name_x, data)

        for index, data in enumerate(Y):
            name_y = os.path.join(name_folder, 'Y_{0}.npy'.format(str(index)))
            np.save(name_y, data)

    def __get_tuple_to_add(proccessed_data_path):

        to_create = []
        for size in DatasetSize:
            name_folder = os.path.join(
                proccessed_data_path, 'Data_{0}'.format(str(size.name)))
            if not os.path.exists(name_folder):
                to_create.append([size])
        return to_create

    def __save_dataset(data_reader):
        proccessed_data_path_model = os.path.join(
            Folders.proccessed_data_path, "Custom")
        to_create = CustomDataset.__get_tuple_to_add(
            proccessed_data_path_model)
        for tuple in to_create:
            size = tuple[0]
            CustomDataset.__save_proccess_data(data_reader, size)

    def __get_features(self, time_index: int):
        name_x = os.path.join(self.proccessed_data_path_model, "Data_{0}".format(
            str(self.size.name)), 'X_{0}.npy'.format(str(time_index)))
        X = np.load(name_x)
        if X is None:
            return X
        else:
            return torch.FloatTensor(X).to(self.device)

    def __get_target(self, time_index: int):
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


class STConvDataset(DatasetClass):

    time_steps_array = [1]

    def __init__(self,
                 time_steps: int,
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
        self.time_steps = time_steps
        self.__check_timesteps()
        self.__save_dataset()
        self.__check_temporal_consistency()
        self.__set_snapshot_count()

    def __check_temporal_consistency(self):
        assert len(glob.glob1(os.path.join(self.proccessed_data_path_STCONV, "Data_{0}_{1}_{2}".format(str(self.time_steps), str(self.batch_size), str(self.size.name))), "X_*.npy")) == len(glob.glob1(
            os.path.join(self.proccessed_data_path_STCONV, "Data_{0}_{1}_{2}".format(str(self.time_steps), str(self.batch_size), str(self.size.name))), "Y_*.npy")), "Temporal dimension inconsistency."

    def need_load(proccessed_data_path):
        return len(STConvDataset.__get_tuple_to_add(os.path.join(proccessed_data_path, "STCONV"))) > 0

    def get_dataset_STCONV(train_ratio: float, test_ratio: float, val_ratio: float, batch_size: int, time_steps: int, epsilon: float, sigma: int, nodes_size: DatasetSize, datareader: DataReader, device: str):
        DataTraffic = STConvDataset(
            time_steps, batch_size, sigma, epsilon, nodes_size, datareader, device)
        train, val, test = DataTraffic.__split_dataset(
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        return train, val, test

    def __set_snapshot_count(self):
        self.snapshot_count = len(glob.glob1(os.path.join(self.proccessed_data_path_STCONV, "Data_{0}_{1}_{2}".format(
            str(self.time_steps), str(self.batch_size), str(self.size.name))), "X_*.npy"))

    def __check_timesteps(self):
        assert self.time_steps in self.time_steps_array

    def __split_dataset(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        assert train_ratio + test_ratio + val_ratio == 1
        time_train = int(train_ratio*self.snapshot_count)
        time_test = time_train + int(test_ratio*self.snapshot_count)

        train_iterator = STConvDataset(self.time_steps,
                                       self.sigma,
                                       self.epsilon,
                                       self.size,
                                       self.data_reader,
                                       self.device,
                                       0,
                                       time_train + 1)

        test_iterator = STConvDataset(self.time_steps,
                                      self.sigma,
                                      self.epsilon,
                                      self.size,
                                      self.data_reader,
                                      self.device,
                                      time_train + 1,
                                      time_test)

        val_iterator = STConvDataset(self.time_steps,
                                     self.sigma,
                                     self.epsilon,
                                     self.size,
                                     self.data_reader,
                                     self.device,
                                     time_test + 1,
                                     self.snapshot_count)

        return train_iterator, test_iterator, val_iterator

    def __save_proccess_data(proccessed_data_path: str, batch_size: int, time_steps: int, datareader: DataReader, size: DatasetSize):

        print("Saving data with configuration : time_steps = {0}, batch_size = {1}, size = {2}".format(
            str(time_steps), str(batch_size), str(size.name)))

        interval_per_day = datareader.interval_per_day
        Skip = (int)(time_steps/2) * 2
        interval_per_day -= Skip
        X, Y = datareader.get_clean_data_by_nodes(size, proccessed_data_path)

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

        name_folder = os.path.join(proccessed_data_path, 'STCONV', 'Data_{0}_{1}_{2}'.format(
            str(time_steps), str(batch_size), str(size.name)))
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

        for index, data in enumerate(X):
            name_x = os.path.join(name_folder, 'X_{0}.npy'.format(str(index)))
            np.save(name_x, data)

        for index, data in enumerate(Y):
            name_y = os.path.join(name_folder, 'Y_{0}.npy'.format(str(index)))
            np.save(name_y, data)

    def __arrange_data_for_time_step(self, data, time_steps, num_nodes):
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
        for time_step in STConvDataset.time_steps_array:
            for batch_size in STConvDataset.batch_sizes_array:
                for size in DatasetSize:
                    name_folder = os.path.join(proccessed_data_path, 'Data_{0}_{1}_{2}'.format(
                        str(time_step), str(batch_size), str(size.name)))
                    if not os.path.exists(name_folder):
                        to_create.append([time_step, batch_size, size])
        return to_create

    def __save_dataset(data_reader):
        proccessed_data_path_STCONV = os.path.join(
            Folders.proccessed_data_path, "STCONV")
        to_create = STConvDataset.__get_tuple_to_add(
            proccessed_data_path_STCONV)
        for tuple in to_create:
            time_step = tuple[0]
            batch_size = tuple[1]
            size = tuple[2]
            STConvDataset.__save_proccess_data(
                batch_size, time_step, data_reader, size)

    def __get_features(self, time_index: int):
        name_x = os.path.join(self.proccessed_data_path_STCONV, "Data_{0}_{1}_{2}".format(str(
            self.time_steps), str(self.batch_size), str(self.size.name)), 'X_{0}.npy'.format(str(time_index)))
        X = np.load(name_x)
        if X is None:
            return X
        else:
            return torch.FloatTensor(X).to(self.device)

    def __get_target(self, time_index: int):
        name_y = os.path.join(self.proccessed_data_path_STCONV, "Data_{0}_{1}_{2}".format(str(
            self.time_steps), str(self.batch_size), str(self.size.name)), 'Y_{0}.npy'.format(str(time_index)))
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
