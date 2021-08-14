from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
from Scripts.data_proccess import get_empty_nodes, get_good_nodes, get_number_of_nodes, read_data
import os
import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data
import glob


Edge_Index = Union[np.ndarray, None] 
Edge_Weight = Union[np.ndarray, None]

    
def get_dataset_STCONV(path, path_proccessed_data,graph_info_txt, train_ratio , test_ratio , val_ratio , batch_size,time_steps,epsilon,lamda,nodes_size):
    if nodes_size == "Experimental":
        edge_index,edge_weigth,X,Y = STConvDataset.get_experimental_data_STCONV(path,batch_size,time_steps)
        DataTraffic = StaticGraphTemporalSignal(edge_index,edge_weigth,X,Y)
        train,test = temporal_signal_split(DataTraffic, train_ratio=train_ratio)
        train,val = temporal_signal_split(train, train_ratio=train_ratio)
        num_nodes = 5
    else:
        num_nodes = get_number_of_nodes(os.path.join(path,graph_info_txt))
        empty_nodes = get_empty_nodes(path,num_nodes)
        num_nodes -= len(empty_nodes)
        DataTraffic = STConvDataset(path,graph_info_txt,path_proccessed_data,time_steps,batch_size,lamda,epsilon,nodes_size)
        train,test,val = STConvDataset.split_dataset(DataTraffic, train_ratio=train_ratio,test_ratio = test_ratio,val_ratio = val_ratio)
        return train,val,test,num_nodes

class STConvDataset(object):

    def __init__(self,path_raw_data,graph_info_txt,proccessed_data_path,time_steps,batch_size,lamda,epsilon,nodes_size,time_start = 0, time_stop = -1):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.lamda = lamda
        self.epsilon = epsilon
        self.time_start = time_start
        self.time_stop = time_stop
        self.nodes_size = nodes_size
        self.proccessed_data_path = proccessed_data_path
        self.path_raw_data = path_raw_data
        self.graph_info_txt = graph_info_txt
        self._save_dataset()
        self._set_snapshot_count()
        self._set_edge_index()
        self._set_edge_weight()
        self._check_temporal_consistency()

    def get_experimental_data_STCONV(path,batch_size,time_steps):
        test_nodes = [718292,769496,718291,718290,764567,774279,774278,764671]
        num_nodes = 8
        number_of_days = 1
        nb_days = number_of_days
        edge_index = np.array([[0,1,7,5,4],[1,2,6,4,3]])
        edge_weigth = np.array([1.,1.,1.,1.,1.])
        interval_per_day = (int)(24 * 60 / 5)
        X = []
        Y = []
        txtFiles = os.path.join(path,"*","*.txt")
        for file in glob(txtFiles):
            with open(file) as f:
                content = f.readlines()
                for line in content:
                    line = line.split(',')
                    if int(line[1]) in test_nodes:
                        line = [line1.replace("\n","") for line1 in line]
                        Y.append((float)(line[11]))
                        X.append([(float)(line[8]),(float)(line[9]),(float)(line[10])])
            number_of_days -= 1
            if number_of_days == 0:
                break
        X = STConvDataset._arrange_data_for_time_step(X,time_steps,num_nodes)
        Y = STConvDataset._arrange_data_for_time_step(Y,time_steps,num_nodes)

        new_size = (int)(interval_per_day * nb_days / batch_size)
        data_size = len(X)
        if new_size * batch_size != data_size:
            difference = data_size-((new_size - 1) * batch_size)
            X = X[:data_size-difference]
            Y = Y[:data_size-difference]
            new_size -= 1

        X = np.array(X).reshape(new_size,batch_size,time_steps,num_nodes,3)
        Y = np.array(Y).reshape(new_size,batch_size,time_steps,num_nodes,1)
        return np.array(edge_index),np.array(edge_weigth),X,Y

    def split_dataset(data_iterator, train_ratio: float=0.6, test_ratio: float = 0.2, val_ratio: float = 0.2):
        assert train_ratio + test_ratio + val_ratio == 1
        time_train = int(train_ratio*data_iterator.snapshot_count)
        time_test = time_train + int(test_ratio*data_iterator.snapshot_count)

        train_iterator = STConvDataset(data_iterator.proccessed_data_path,
                                                    data_iterator.time_steps,
                                                    data_iterator.batch_size,
                                                    data_iterator.lamda,
                                                    data_iterator.epsilon,
                                                    0,
                                                    time_train + 1)

        test_iterator = STConvDataset(data_iterator.proccessed_data_path,
                                                    data_iterator.time_steps,
                                                    data_iterator.batch_size,
                                                    data_iterator.lamda,
                                                    data_iterator.epsilon,
                                                    time_train + 1,
                                                    time_test)

        val_iterator = STConvDataset(data_iterator.proccessed_data_path,
                                                    data_iterator.time_steps,
                                                    data_iterator.batch_size,
                                                    data_iterator.lamda,
                                                    data_iterator.epsilon,
                                                    time_test + 1,
                                                    data_iterator.snapshot_count)
        
        return train_iterator, test_iterator, val_iterator

    def _save_proccess_data(self,path_save_data,batch_size,time_steps,x,y,nb_days,num_nodes):

        interval_per_day = (int)(24 * 60 / 5)
        Skip = (int)(time_steps/2) * 2
        interval_per_day -= Skip

        X = x
        Y = y
        
        X = self._arrange_data_for_time_step(X,time_steps,num_nodes)
        Y = self._arrange_data_for_time_step(Y,time_steps,num_nodes)


        new_size = (int)(interval_per_day * nb_days / batch_size)
        data_size = len(X)
        if new_size * batch_size != data_size:
            difference = data_size-((new_size - 1) * batch_size)
            X = X[:data_size-difference]
            Y = Y[:data_size-difference]
            new_size -= 1

        X = np.array(X).reshape(new_size,batch_size,time_steps,num_nodes,3)
        Y = np.array(Y).reshape(new_size,batch_size,time_steps,num_nodes,1)

        name_folder = os.path.join(path_save_data,'Data_{0}_{1}'.format(str(time_steps),str(batch_size)))
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)

        for index,data in enumerate(X):
            name_x = os.path.join(name_folder,'X_{0}.npy'.format(str(index)))
            np.save(name_x, data)

        for index,data in enumerate(Y):
            name_y = os.path.join(name_folder,'Y_{0}.npy'.format(str(index)))
            np.save(name_y, data)

    def _arrange_data_for_time_step(self,data,time_steps,num_nodes):
        Skip = (int)(time_steps/2)
        New_Data = []
        for i in range(Skip,(int)((len(data) - (num_nodes * Skip))/(num_nodes))):
            Graph = []
            for j in range(-Skip,Skip + 1):
                Data = []
                for k in range(num_nodes):
                    Data.append(data[(i * k) + k + (num_nodes * j)])
                Graph.append(Data)
            New_Data.append(Graph)
        return New_Data
        
    def _save_dataset(self):
        total_num_nodes = get_number_of_nodes(os.path.join(self.path_raw_data,self.graph_info_txt))
        empty_nodes = get_empty_nodes(self.path_raw_data,total_num_nodes)
        total_num_nodes -= len(empty_nodes)
        batch_sizes= [8,16,32]
        time_steps= [1,3,5,7]
        to_create = []
        for time_step in time_steps:
            for batch_size in batch_sizes:
                name_folder = os.path.join(self.proccessed_data_path,'Data_{0}_{1}'.format(str(time_steps),str(batch_size)))
                if not os.path.exists(name_folder):
                    os.makedirs(name_folder)
                    to_create.append([time_steps,batch_size])
        X,Y,nb_days = read_data(self.path_raw_data,empty_nodes)
        for tuple in to_create:
            time_step = tuple[0]
            batch_size = tuple[1]
            self._save_proccess_data(self.proccessed_data_path,batch_size,time_step,X,Y,nb_days,total_num_nodes)

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
