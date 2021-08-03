from Scripts.utilities import get_adjency_matrix_weight
import pandas as pd
import os
from glob import glob
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal.static_graph_temporal_signal import Edge_Weight
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split

def get_graph_info(info_data,epsilon,lamda):
    edge_index = []
    edge_weight = []
    nodes_location = []
    skip = True
    with open(info_data) as f:
        content = f.readlines()
        
        for line in content:
            if skip:
                skip = False
            else:
                line = line.split('\t')
                line = line[:-1]
                nodes_location.append([line[0],line[8],line[9]])
    edge_weight = np.zeros((len(nodes_location),len(nodes_location)))
    for i in range(len(nodes_location) - 1):
        for j in range(len(nodes_location) - 1):
            if i != j:
                p1 = (nodes_location[i][1],nodes_location[i][2])
                p2 = (nodes_location[j][1],nodes_location[j][2])
                weight = get_adjency_matrix_weight(p1,p2,epsilon,lamda)
                edge_weight[i][j] = weight
                if weight > 0:
                    edge_index.append([i,j])
    edge_index = np.transpose(edge_index)
    return np.array(edge_index),np.array(edge_weight),len(nodes_location)

def get_dataset(path,train_test_ratio,train_validation_ratio,batch_size,time_steps,epsilon,lamda):
    edge_index,edge_weigth,X,Y,num_nodes = extract_data(path,batch_size,time_steps,epsilon,lamda)
    DataTraffic = StaticGraphTemporalSignal(edge_index,edge_weigth,X,Y)
    train,test = temporal_signal_split(DataTraffic, train_ratio=train_test_ratio)
    train,val = temporal_signal_split(train, train_ratio=train_validation_ratio)
    return train,val,test,num_nodes

def get_dataset_experimental(path,train_test_ratio,train_validation_ratio,batch_size,time_steps):
    edge_index,edge_weigth,X,Y = get_experimental_data(path,batch_size,time_steps)
    DataTraffic = StaticGraphTemporalSignal(edge_index,edge_weigth,X,Y)
    train,test = temporal_signal_split(DataTraffic, train_ratio=train_test_ratio)
    train,val = temporal_signal_split(train, train_ratio=train_validation_ratio)

    return train,val,test

def extract_data(path,batch_size,time_steps,epsilon,lamda):
    txtFiles = os.path.join(path,"*","*.txt")
    X = []
    Y = []
    edge_index,edge_weigth,num_nodes = get_graph_info(os.path.join("Data","d07_text_meta_2021_03_27.txt"),epsilon,lamda)
    interval_per_day = (int)(24 * 60 / 5)
    nb_days = 0
    for file in glob(txtFiles):
        with open(file) as f:
            content = f.readlines()
            for line in content:
                line = line.split(',')
                line = [line1.replace("\n","") for line1 in line]
                Y.append((float)(line[15]))
                X.append([(float)(line[13]),(float)(line[14]),(float)(line[16])])
        nb_days += 1

    X = arrange_data_for_time_step(X,time_steps,num_nodes)
    Y = arrange_data_for_time_step(Y,time_steps,num_nodes)
    X = np.array(X).reshape((int)(interval_per_day * nb_days / batch_size),batch_size,time_steps,num_nodes,3)
    Y = np.array(Y).reshape((int)(interval_per_day * nb_days / batch_size),batch_size,time_steps,num_nodes,1)
    return edge_index,edge_weigth,X,Y,num_nodes

def get_experimental_data(path,batch_size,time_steps):
    # test nodes    0       1      2     3       4      5      6     7
    test_nodes = [718292,769496,718291,718290,764567,774279,774278,764671]
    num_nodes = 8
    # test time
    number_of_days = 1
    nb_days = number_of_days
    # graph connectivity
    edge_index = np.array([[0,1,7,5,4],[1,2,6,4,3]])
    # weigths, to be added distance between them
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
                    Y.append((float)(line[15]))
                    X.append([(float)(line[13]),(float)(line[14]),(float)(line[16])])
        number_of_days -= 1
        if number_of_days == 0:
            break
    X = arrange_data_for_time_step(X,time_steps,num_nodes)
    Y = arrange_data_for_time_step(Y,time_steps,num_nodes)

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

def arrange_data_for_time_step(data,time_steps,num_nodes):
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