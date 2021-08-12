import os
import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data
import glob


Edge_Index = Union[np.ndarray, None] 
Edge_Weight = Union[np.ndarray, None]


def split_dataset(data_iterator, train_ratio: float=0.6, test_ratio: float = 0.2, val_ratio: float = 0.2):
    assert train_ratio + test_ratio + val_ratio == 1
    time_train = int(train_ratio*data_iterator.snapshot_count)
    time_test = time_train + int(test_ratio*data_iterator.snapshot_count)

    train_iterator = CustomStaticGraphTemporalSignal(data_iterator.proccessed_data_path,
                                                data_iterator.time_steps,
                                                data_iterator.batch_size,
                                                data_iterator.lamda,
                                                data_iterator.epsilon,
                                                0,
                                                time_train + 1)

    test_iterator = CustomStaticGraphTemporalSignal(data_iterator.proccessed_data_path,
                                                data_iterator.time_steps,
                                                data_iterator.batch_size,
                                                data_iterator.lamda,
                                                data_iterator.epsilon,
                                                time_train + 1,
                                                time_test)

    val_iterator = CustomStaticGraphTemporalSignal(data_iterator.proccessed_data_path,
                                                data_iterator.time_steps,
                                                data_iterator.batch_size,
                                                data_iterator.lamda,
                                                data_iterator.epsilon,
                                                time_test + 1,
                                                data_iterator.snapshot_count)
    
    return train_iterator, test_iterator, val_iterator

class CustomStaticGraphTemporalSignal(object):

    r""" 
    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
    """
    def __init__(self,proccessed_data_path,time_steps,batch_size,lamda,epsilon,time_start = 0, time_stop = -1):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.lamda = lamda
        self.epsilon = epsilon
        self.time_start = time_start
        self.time_stop = time_stop
        self.proccessed_data_path = proccessed_data_path
        self._set_snapshot_count()
        self._set_edge_index()
        self._set_edge_weight()
        self._check_temporal_consistency()

    def _set_snapshot_count(self):
        self.snapshot_count = len(glob.glob1(os.path.join(self.proccessed_data_path,"Data_{0}_{1}".format(str(self.time_steps),str(self.batch_size))),"X_*.npy"))

    def _check_temporal_consistency(self):
        assert len(glob.glob1(os.path.join(self.proccessed_data_path,"Data_{0}_{1}".format(str(self.time_steps),str(self.batch_size))),"X_*.npy")) == len(glob.glob1(os.path.join(self.proccessed_data_path,"Data_{0}_{1}".format(str(self.time_steps),str(self.batch_size))),"Y_*.npy")) , "Temporal dimension inconsistency."

    def _set_edge_index(self):
        name_index = os.path.join(self.proccessed_data_path,'Data_EdgeIndex','index_{0}_{1}.npy'.format(str(self.epsilon),str(self.lamda)))
        self.edge_index = np.load(name_index)

    def _set_edge_weight(self):
        name_weight = os.path.join(self.proccessed_data_path,'Data_EdgeWeight','weight_{0}_{1}.npy'.format(str(self.epsilon),str(self.lamda)))
        self.edge_weight = np.load(name_weight)

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)

    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

    def _get_features(self, time_index: int):
        name_x = os.path.join(self.proccessed_data_path,"Data_{0}_{1}".format(str(self.time_steps),str(self.batch_size)),'X_{0}.npy'.format(str(time_index))) 
        X = np.load(name_x)
        if X is None:
            return X
        else:       
            return torch.FloatTensor(X)

    def _get_target(self, time_index: int):
        name_y = os.path.join(self.proccessed_data_path,"Data_{0}_{1}".format(str(self.time_steps),str(self.batch_size)),'Y_{0}.npy'.format(str(time_index))) 
        Y = np.load(name_y)
        if Y is None:
            return Y
        else:
            if Y.dtype.kind == 'i':
                return torch.LongTensor(Y)
            elif Y.dtype.kind == 'f':
                return torch.FloatTensor(Y)
         

    def __get_item__(self, time_index: int):
        x = self._get_features(time_index)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        y = self._get_target(time_index)

        snapshot = Data(x = x,
                        edge_index = edge_index,
                        edge_attr = edge_weight,
                        y = y)
        return snapshot

    def __next__(self):
        if self.t < self.time_stop:
            snapshot = self.__get_item__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = self.time_start
            raise StopIteration

    def __iter__(self):
        self.t = self.time_start
        return self