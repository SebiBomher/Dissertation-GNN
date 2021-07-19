import pandas as pd
import os
from glob import glob
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal.static_graph_temporal_signal import Edge_Weight
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split

def get_graph(info_data):
    edge_index = []
    edge_weight = []
    with open(info_data) as f:
        content = f.readlines()
        for line in content:
            print(line)
    
    return edge_index,edge_weight

def get_dataset(path,train_test_ratio,train_validation_ratio,batch_size,time_steps):
    X,Y = extract_data(path,batch_size,time_steps)
    edge_index,edge_weigth = get_graph(os.path.join(path,"d07_text_meta_2021_03_27.txt"))
    DataTraffic = StaticGraphTemporalSignal(edge_index,edge_weigth,X,Y)
    train,test = temporal_signal_split(DataTraffic, train_ratio=train_test_ratio)
    train,val = temporal_signal_split(train, train_ratio=train_validation_ratio)
    
    return train,val,test

def get_dataset_experimental(path,train_test_ratio,train_validation_ratio,batch_size,time_steps):
    edge_index,edge_weigth,X,Y = get_experimental_data(path,batch_size,time_steps)
    DataTraffic = StaticGraphTemporalSignal(edge_index,edge_weigth,X,Y)
    train,test = temporal_signal_split(DataTraffic, train_ratio=train_test_ratio)
    train,val = temporal_signal_split(train, train_ratio=train_validation_ratio)

    return train,val,test

def extract_data(path,batch_size,time_steps):
    txtFiles = os.path.join(path,"*","*.txt")
    X = []
    Y = []
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

    X = np.array(X).reshape((int)(interval_per_day * nb_days / batch_size),batch_size,time_steps,8,3)
    Y = np.array(Y).reshape((int)(interval_per_day * nb_days / batch_size),batch_size,time_steps,8,1)
    return X,Y

def get_experimental_data(path,batch_size,time_steps):
    # test nodes    0       1      2     3       4      5      6     7
    test_nodes = [718292,769496,718291,718290,764567,774279,774278,764671]
    # test time
    number_of_days = 1
    nb_days = 1
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

    X = np.array(X).reshape((int)(interval_per_day * nb_days / batch_size),batch_size,time_steps,8,3)
    Y = np.array(Y).reshape((int)(interval_per_day * nb_days / batch_size),batch_size,time_steps,8,1)
    return np.array(edge_index),np.array(edge_weigth),X,Y