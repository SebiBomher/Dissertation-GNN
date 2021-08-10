from Scripts.utilities import get_adjency_matrix_weight
import os
from glob import glob
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
from torch_geometric.data import DataLoader
from Scripts.dataloader import CustomStaticGraphTemporalSignal,split_dataset
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

loader = ChickenpoxDatasetLoader()

dataset = loader.get_dataset()


def get_graph_info(save_data,epsilon,lamda):
    name_weight = os.path.join(save_data,'weight_{0}_{1}.npy'.format(str(epsilon),str(lamda)))
    name_index = os.path.join(save_data,'index_{0}_{1}.npy'.format(str(epsilon),str(lamda)))
    edge_weight = np.load(name_weight)
    edge_index = np.load(name_index)
    return edge_index,edge_weight

def save_graph_info(info_data,save_data,epsilon,lamda,empty_nodes):
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
                if line[0] not in empty_nodes:
                    nodes_location.append([line[0],line[8],line[9]])
    edge_weight = []
    for i in range(len(nodes_location) - 1):
        for j in range(i,len(nodes_location) - 1):
            if i != j:
                p1 = (nodes_location[i][1],nodes_location[i][2])
                p2 = (nodes_location[j][1],nodes_location[j][2])
                weight = get_adjency_matrix_weight(p1,p2,epsilon,lamda)
                if weight > 0:
                    edge_index.append([i,j])
                    edge_weight.append(weight)
    edge_index = np.transpose(edge_index)
    name_weight = os.path.join(save_data,'weight_{0}_{1}.npy'.format(str(epsilon),str(lamda)))
    name_index = os.path.join(save_data,'index_{0}_{1}.npy'.format(str(epsilon),str(lamda)))
    np.save(name_index,edge_index)
    np.save(name_weight,edge_weight)

def get_dataset(path,path_proccessed_data,graph_info_txt,train_test_ratio,train_validation_ratio,batch_size,time_steps,epsilon,lamda):
    num_nodes = get_number_of_nodes(os.path.join(path,graph_info_txt))
    empty_nodes = get_empty_nodes(path,num_nodes)
    num_nodes -= len(empty_nodes)
    DataTraffic = CustomStaticGraphTemporalSignal(path_proccessed_data,time_steps,batch_size,lamda,epsilon)
    train,test = split_dataset(DataTraffic, train_ratio=train_test_ratio)
    train,val = split_dataset(train, train_ratio=train_validation_ratio)
    return train,val,test,num_nodes

def get_dataset_experimental(path,train_test_ratio,train_validation_ratio,batch_size,time_steps):
    edge_index,edge_weigth,X,Y = get_experimental_data(path,batch_size,time_steps)
    DataTraffic = StaticGraphTemporalSignal(edge_index,edge_weigth,X,Y)
    train,test = temporal_signal_split(DataTraffic, train_ratio=train_test_ratio)
    train,val = temporal_signal_split(train, train_ratio=train_validation_ratio)

    return train,val,test

def get_empty_nodes(path,num_nodes):
    empty_nodes = []
    index = num_nodes
    txtFiles = os.path.join(path,"*","*.txt")
    for file in glob(txtFiles):
        with open(file) as f:
            content = f.readlines()
            for line in content:
                line = line.split(',')
                line = [line1.replace("\n","") for line1 in line]
                if line[9] == '' or line[10] == '' or line[8] == '' or line[11] == '':
                    empty_nodes.append(line[1])
                index -=1
                if index == 0:
                    return empty_nodes

def get_number_of_nodes(info_data):
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
    return len(nodes_location)
    
def get_proccess_data(path_save_data,batch_size,time_steps):
    name_x = os.path.join(path_save_data,'X_{0}_{1}.npy'.format(str(time_steps),str(batch_size)))
    name_y = os.path.join(path_save_data,'Y_{0}_{1}.npy'.format(str(time_steps),str(batch_size)))
    X = np.load(name_x)
    Y = np.load(name_y)
    return X,Y

def read_data(path,empty_nodes):
    X = []
    Y = []
    txtFiles = os.path.join(path,"*","*.txt")
    nb_days = 0
    for file in glob(txtFiles):
        print(nb_days)
        with open(file) as f:
            content = f.readlines()
            for line in content:
                line = line.split(',')
                line = [line1.replace("\n","") for line1 in line]
                if line[1] not in empty_nodes:
                    Y.append((float)(line[11]))
                    X.append([(float)(line[8]),(float)(line[9]),(float)(line[10])])
        nb_days += 1
    return X,Y,nb_days

def save_proccess_data(path_save_data,batch_size,time_steps,x,y,nb_days,num_nodes):

    interval_per_day = (int)(24 * 60 / 5)
    Skip = (int)(time_steps/2) * 2
    interval_per_day -= Skip

    X = x
    Y = y
    
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

    name_folder = os.path.join(path_save_data,'Data_{0}_{1}'.format(str(time_steps),str(batch_size)))
    if not os.path.exists(name_folder):
        os.makedirs(name_folder)

    for index,data in enumerate(X):
        name_x = os.path.join(name_folder,'X_{0}.npy'.format(str(index)))
        np.save(name_x, data)

    for index,data in enumerate(Y):
        name_y = os.path.join(name_folder,'Y_{0}.npy'.format(str(index)))
        np.save(name_y, data)

def extract_data(path,path_proccessed_data,graph_info_txt,batch_size,time_steps,epsilon,lamda):
    num_nodes = get_number_of_nodes(os.path.join(path,graph_info_txt))
    empty_nodes = get_empty_nodes(path,num_nodes)
    num_nodes -= len(empty_nodes)
    edge_index,edge_weigth = get_graph_info(path_proccessed_data,epsilon,lamda)
    X,Y = get_proccess_data(path_proccessed_data,batch_size,time_steps)
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
                    Y.append((float)(line[11]))
                    X.append([(float)(line[8]),(float)(line[9]),(float)(line[10])])
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

def create_proccessed_data(path,path_save_proccessed_data,graph_info_txt):
    epsilon_array = [0.1, 0.3, 0.5, 0.7]
    lamda_array = [1, 3, 5, 10]
    batch_sizes= [8,16,32]
    time_steps= [1,3,5,7]
    num_nodes = get_number_of_nodes(os.path.join(path,graph_info_txt))
    empty_nodes = get_empty_nodes(path,num_nodes)
    num_nodes -= len(empty_nodes)
    # for epsilon in epsilon_array:
    #     for lamda in lamda_array:
    #         print("{0} {1}".format(epsilon,lamda))
    #         save_graph_info(os.path.join(path,graph_info_txt),path_save_proccessed_data,epsilon,lamda,empty_nodes)
    X,Y,nb_days = read_data(path,empty_nodes)
    for time_step in time_steps:
        for batch_size in batch_sizes:
            print("{0} {1}".format(time_step,batch_size))
            if time_step == 1 and (batch_size == 16 or batch_size == 8):
                continue
            save_proccess_data(path_save_proccessed_data,batch_size,time_step,X,Y,nb_days,num_nodes)