from random import sample
from Scripts.utilities import get_adjency_matrix_weight
import os
from glob import glob
import numpy as np

class DataReader():
    def __init__(self,path_raw_data,graph_info_txt):
        self.path_raw_data = path_raw_data
        self.graph_info_txt = graph_info_txt

    def __get_number_of_nodes(self):
        info_data =  glob(os.path.join(self.path_raw_data,self.graph_info_txt))
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

        self.__num_nodes = len(nodes_location)

    def get_good_empty_nodes(self):
        self.__get_number_of_nodes()
        index = self.__num_nodes
        empty_nodes = []
        good_nodes = []
        txtFiles = os.path.join(self.path_raw_data,"*","*.txt")
        for file in glob(txtFiles):
            with open(file) as f:
                content = f.readlines()
                for line in content:
                    line = line.split(',')
                    line = [line1.replace("\n","") for line1 in line]
                    line[0]
                    if not (line[9] == '' or line[10] == '' or line[8] == '' or line[11] == ''):
                        good_nodes.append(line[1])
                    index -=1
                    if index == 0:
                        return good_nodes,empty_nodes

    def read_data(self):
        X = []
        Y = []
        empty_nodes = []
        good_nodes = []
        txtFiles = os.path.join(self.path_raw_data,"*","*.txt")
        nb_days = 0
        for file in glob(txtFiles):
            with open(file) as f:
                content = f.readlines()
                for line in content:
                    line = line.split(',')
                    line = [line1.replace("\n","") for line1 in line]
                    if line[9] == '' or line[10] == '' or line[8] == '' or line[11] == '':
                        empty_nodes.append(line[1])
                    else:
                        good_nodes.append(line[1])
                        Y.append((float)(line[11]))
                        X.append([(float)(line[8]),(float)(line[9]),(float)(line[10])])
            nb_days += 1
        return X,Y,nb_days
  

    def read_nodes_data(self):
        info_data =  glob(os.path.join(self.path_raw_data,self.graph_info_txt))
        nodes_location = []
        with open(info_data) as f:
            content = f.readlines()
            for line in content:
                if skip:
                    skip = False
                else:
                    line = line.split('\t')
                    line = line[:-1]
                    if line[0] not in self.empty_nodes:
                        nodes_location.append([line[0],line[8],line[9]])
        self.nodes_location = nodes_location

class Graph():
    def __init__(self,path_raw_data,graph_info_txt,path_processed_data,epsilon,lamda,size):
        self.path_raw_data = path_raw_data
        self.graph_info_txt = graph_info_txt
        self.path_processed_data = path_processed_data
        self.epsilon = epsilon
        self.lamda = lamda
        self.size = size
        self.data_reader = DataReader(self.path_raw_data,self.graph_info_txt)
        self._save_graph_info()

    def get_graph_info(self):
        name_weight = os.path.join(self.path_processed_data,'weight_{0}_{1}_{2}.npy'.format(str(self.epsilon),str(self.lamda),str(self.size)))
        name_index = os.path.join(self.path_processed_data,'index_{0}_{1}_{2}.npy'.format(str(self.epsilon),str(self.lamda),str(self.size)))
        edge_weight = np.load(name_weight)
        edge_index = np.load(name_index)
        return edge_index,edge_weight

    def get_nodes_by_size(self,size):
        good_nodes = self.data_reader.get_good_empty_nodes()
        if size == "Full":
            return good_nodes
        elif size == "Medium":
            return sample(good_nodes, 1200)
        elif size == "Small":
            return sample(good_nodes, 120)
        elif size == "Experimental":
            return [718292,769496,718291,718290,764567,774279,774278,764671]

    def _save_graph_info(self):
        edge_index = []
        edge_weight = []

        epsilon_array = [0.1, 0.3, 0.5, 0.7]
        lamda_array = [1, 3, 5, 10]
        size_array = ["Full","Medium","Small","Experimental"]
        # for epsilon in epsilon_array:
        #     for lamda in lamda_array:
        #         for size in size_array:
        nodes_location = self.data_reader.nodes_location
        all_nodes = self.get_nodes_by_size(size)
        nodes_location = [node for node in nodes_location if node in all_nodes]

        for i in range(len(nodes_location) - 1):
            for j in range(i,len(nodes_location) - 1):
                if i != j:
                    p1 = (nodes_location[i][1],nodes_location[i][2])
                    p2 = (nodes_location[j][1],nodes_location[j][2])
                    weight = get_adjency_matrix_weight(p1,p2,self.epsilon,self.lamda)
                    if weight > 0:
                        edge_index.append([i,j])
                        edge_weight.append(weight)

        edge_index = np.transpose(edge_index)
        name_weight = os.path.join(self.path_processed_data,'weight_{0}_{1}_{2}.npy'.format(str(self.epsilon),str(self.lamda),str(self.size)))
        name_index = os.path.join(self.path_processed_data,'index_{0}_{1}_{2}.npy'.format(str(self.epsilon),str(self.lamda),str(self.size)))
        np.save(name_index,edge_index)
        np.save(name_weight,edge_weight)

    def get_empty_and_good_nodes(path,num_nodes):
        empty_nodes = []
        good_nodes = []
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
                    else:
                        good_nodes.append(line[1])
                    index -=1
                    if index == 0:
                        return empty_nodes,good_nodes

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

    

    def create_proccessed_data_graph(path,path_save_proccessed_data,graph_info_txt):
        epsilon_array = [0.1, 0.3, 0.5, 0.7]
        lamda_array = [1, 3, 5, 10]
        num_nodes = get_number_of_nodes(os.path.join(path,graph_info_txt))
        empty_nodes = get_empty_nodes(path,num_nodes)
        num_nodes -= len(empty_nodes)
        for epsilon in epsilon_array:
            for lamda in lamda_array:
                print("{0} {1}".format(epsilon,lamda))
                save_graph_info(os.path.join(path,graph_info_txt),path_save_proccessed_data,epsilon,lamda,empty_nodes)
    