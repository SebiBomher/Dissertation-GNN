import math
from random import sample

from geopy.distance import geodesic
import os
from glob import glob
import numpy as np
from enum import Enum

class DatasetSizeNumber(Enum):
    r"""
        Number of DatasetSizes, the total number of nodes per each dataset type.
            Full = 2800
            Medium = 120
            Small = 120
            Experimental = 5
    """

    Full = 2800
    Medium = 1200
    Small = 120
    Experimental = 5


class DatasetSize(Enum):
    r"""
        Types of Dataset Sizes.
            Full = 0
            Medium = 1
            Small = 2
            Experimental = 3
    """

    Full = 0
    Medium = 1
    Small = 2
    Experimental = 3

class DataReader():
    r"""
        DataReader Class, Once initialized it will read data once and will be able to access data without re-reading aditional data

        Args : 
            path_raw_data , string :  Path of raw data
            graph_info_txt , string :  Txt file for the nodes metadata

        Variables:
            __path_raw_data , string : Path of raw data - private
            __graph_info_txt , string : Txt file for the nodes metadata
            X , List : Data - public
            Y , List : Labels - public
            nb_days , integer: Total number of days for the given data - public
            nodes_location , list[int, float, float] : geographic location of each node containing the id, x coordinate and y coordinate (note : it can contain empty nodes, must be processed) - public
            empty_nodes , list : Nodes which do not contain data - public
            good_nodes , list : Nodes which contain data - public

        Instance Functions: 
            __get_number_of_nodes : set total number of nodes (nb_days) from Metadata (may contain empty nodes) -> None - private
            __read_data : set data and labes (X and Y) from Data (may contain data from empty nodes) -> None - private
            __get_good_empty_nodes : set empty_nodes and good_nodes from Data (may contain data from empty nodes) -> None - private
            __read_nodes_data : set nodes_location from Metadata (may contain data from empty nodes) -> None - private
    """

    def __init__(self,__path_raw_data : str,graph_info_txt : str) -> None:
        r"""
            Constructor, reads all the necessary data.
            Args : 
                __path_raw_data , string :  Path of raw data
                graph_info_txt , string :  Txt file for the nodes metadata
        """

        self.__path_raw_data = __path_raw_data
        self.__graph_info_txt = graph_info_txt
        self.__get_good_empty_nodes()
        self.__read_data()
        self.__read_nodes_data()

    def __get_number_of_nodes(self) -> None:
        r"""
            Instance function.
            Set total number of nodes (nb_days) from Metadata (may contain empty nodes).
            Returns Nothing.
        """

        info_data =  glob(os.path.join(self.__path_raw_data,self.__graph_info_txt))
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

        self.__total_num_nodes = len(nodes_location)

    def __get_good_empty_nodes(self) -> None :
        r"""
            Instance function.
            Set empty_nodes and good_nodes from Data (may contain data from empty nodes).
            Returns Nothing.
        """

        self.__get_number_of_nodes()
        index = self.__total_num_nodes
        empty_nodes = []
        good_nodes = []
        txtFiles = os.path.join(self.__path_raw_data,"*","*.txt")
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
                        self.good_nodes = good_nodes
                        self.empty_nodes = empty_nodes
                        return

    def __read_data(self) -> None :
        r"""
            Instance function.
            Set data and labes (X and Y) from Data (may contain data from empty nodes).
            Returns Nothing.
        """

        X = []
        Y = []
        empty_nodes = []
        good_nodes = []
        txtFiles = os.path.join(self.__path_raw_data,"*","*.txt")
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
        self.X = X
        self.Y = Y
        self.nb_days = nb_days

    def __read_nodes_data(self) -> None :
        r"""
            Instance function.
            Set nodes_location from Metadata (may contain data from empty nodes).
            Returns Nothing.
        """
        
        info_data =  glob(os.path.join(self.__path_raw_data,self.__graph_info_txt))
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
    r"""
        Graph class, contains data for graph construction in pytorch.
        For the first time instancing this class all data will be saved in path_processed_data for further use, such that there wont be any redundant processing

        Args: 
            path_processed_data , string : Path where the processed data will be saved
            epsilon , float : epsilon for weight calculation (see documentation for further information)
            lamda , integer : lamda for weight calculation (see documentation for further information)
            size , DatasetSize class : Size of the graph
            data_reader , DataReader class : Helper class for data reading

        Static Variables:
            lamda_array = [0.1, 0.3, 0.5, 0.7] - lamda posibilities
            epsilon_array = [1, 3, 5, 10] - epsilon posibilities

        Variables:
            __path_processed_data , string : Path where the processed data will be saved - private
            __epsilon , float : epsilon for weight calculation (see documentation for further information) - private
            __lamda , integer : lamda for weight calculation (see documentation for further information) - private
            __size , DatasetSize class : Size of the graph - private
            __data_reader , DataReader class : Helper class for data reading - private
            num_nodes , integer : Total number of nodes for the graph per size given - public
            edge_index , list (2,num_nodes) : list of graph edges for pytorch
            edge_weight , list (num_nodes) : list of graph weights for pytorch

        Instance Functions:
            __check___lamda___epsilon : checks if lamda and epsilon are valid -> None - private
            __set_nodes : sets nodes array of ids for each size, so the ids will always be the same -> None - private
            __get_nodes_ids_by_size : returns graph nodes ids based by size -> list - private
            __process_graph_info : checks if the data required is available, if not it creates it -> None - private
            __save_graph : save a graph by a configuration -> None - private
            __set_graph_info : sets the edge_index and edge_weight -> None- private

        Class Functions:
            get_number_nodes_by_size : returns the number of nodes by size -> int: - public
            __get_adjency_matrix_weight : gets the weight of 2 nodes based on lamda and epsilon (see documentation for further information) -> float : - private
    """

    lamda_array = [0.1, 0.3, 0.5, 0.7]
    epsilon_array = [1, 3, 5, 10]

    def __init__(self,path_processed_data : str, epsilon : float,lamda : int ,size : DatasetSize,data_reader : DataReader) -> None :
        r"""
            Constructor, makes the processing and data saving.

            Args: 
                path_processed_data , string : Path where the processed data will be saved
                epsilon , float : epsilon for weight calculation (see documentation for further information)
                lamda , integer : lamda for weight calculation (see documentation for further information)
                size , DatasetSize class : Size of the graph
                data_reader , DataReader class : Helper class for data reading
        """
        self.__path_processed_data = path_processed_data
        self.__epsilon = epsilon
        self.__lamda = lamda
        self.__size = size
        self.__data_reader = data_reader
        self.edge_index = []
        self.edge_weight = []
        self.num_nodes = 0
        self.nodes_id_size_array = []
        self.__check___lamda___epsilon()
        self.__set_nodes()
        self.__process_graph_info()
        self.__set_graph_info()
        

    def __check___lamda___epsilon(self) -> None:
        r"""
            Instance function. 
            Checks if lamda and epsilon are valid.
            Returns Nothing.
        """
        assert self.__lamda in self.lamda_array and self.__epsilon in self.__epsilon_array

    def __set_nodes(self,size) -> None:
        r"""
            Instance function.
            Sets nodes array of ids for each size, so the ids will always be the same.
            Returns Nothing.
        """

        good_nodes = self.__data_reader.good_nodes
        sizes_to_add = []
        for size in DatasetSize:
            name_nodes = os.path.join(self.__path_processed_data,'nodes_{0}.npy'.format(str(size)))
            if os.path.isfile(name_nodes):
                sizes_to_add.append(size)

        for size in sizes_to_add:
            nodes = sample(good_nodes, self.get_number_nodes_by_size(size))
            name_nodes = os.path.join(self.__path_processed_data,'nodes_{0}.npy'.format(str(size)))
            np.save(nodes,name_nodes)

    def __get_nodes_ids_by_size(self,size) -> list:
        r"""
            Instance function.
            Returns graph nodes ids based by size.
            Returns list.
        """
        name_nodes = os.path.join(self.__path_processed_data,'nodes_{0}.npy'.format(str(size)))
        return np.load(name_nodes)

    def __process_graph_info(self) -> None:
        r"""
            Instance function.
            Checks if the data required is available, if not it creates it.
            Returns Nothing.
        """
        nodes_location = self.__data_reader.nodes_location
        list_to_add = []

        for epsilon in Graph.epsilon_array:
            for lamda in Graph.lamda_array:
                for size in DatasetSize:
                    name_weight = os.path.join(self.__path_processed_data,'weight_{0}_{1}_{2}.npy'.format(str(epsilon),str(lamda),str(size)))
                    name_index = os.path.join(self.__path_processed_data,'index_{0}_{1}_{2}.npy'.format(str(epsilon),str(lamda),str(size)))
                    if os.path.isfile(name_index) and os.path.isfile(name_weight):
                        list_to_add.append([epsilon,lamda,size])

        for info in list_to_add:
            epsilon = info[0]           
            lamda = info[1]           
            size = info[2]
            self.__save_graph(self.__get_nodes_ids_by_size(size),nodes_location,epsilon,lamda,size)  


    def __save_graph(self,nodes,nodes_location,epsilon,lamda,size) -> None:
        r"""
            Instance function.
            Save a graph by a configuration.
            Returns Nothing.
        """
        edge_index = []
        edge_weight = []
        nodes_location = [node for node in nodes_location if node in nodes]
        self.num_nodes = len(nodes_location)

        for i in range(len(nodes_location) - 1):
            for j in range(i,len(nodes_location) - 1):
                if i != j:
                    p1 = (nodes_location[i][1],nodes_location[i][2])
                    p2 = (nodes_location[j][1],nodes_location[j][2])
                    weight = self.__get_adjency_matrix_weight(p1,p2,epsilon,lamda)
                    if weight > 0:
                        edge_index.append([i,j])
                        edge_weight.append(weight)

        edge_index = np.transpose(edge_index)
        name_weight = os.path.join(self.__path_processed_data,'weight_{0}_{1}_{2}.npy'.format(str(epsilon),str(lamda),str(size)))
        name_index = os.path.join(self.__path_processed_data,'index_{0}_{1}_{2}.npy'.format(str(epsilon),str(lamda),str(size)))
        np.save(name_index,edge_index)
        np.save(name_weight,edge_weight)

    def __set_graph_info(self) -> None:
        r"""
            Instance function.
            Sets the edge_index, edge_weight and num_nodes.
            Returns Nothing.
        """
        name_weight = os.path.join(self.__path_processed_data,'Data_EdgeWeight','weight_{0}_{1}_{2}.npy'.format(str(self.__epsilon),str(self.__lamda),str(self.__size)))
        self.edge_weight = np.load(name_weight)

        name_index = os.path.join(self.__path_processed_data,'Data_EdgeWeight','index_{0}_{1}_{2}.npy'.format(str(self.__epsilon),str(self.__lamda),str(self.__size)))
        self.edge_index = np.load(name_index)

    def get_number_nodes_by_size(size : DatasetSize) -> int:
        r"""
            Class function.
            Returns the number of nodes by size
            Returns Integer
        """
        if size == DatasetSize.Experimental:
            return DatasetSizeNumber.Experimental
        elif size == DatasetSize.Medium:
            return DatasetSizeNumber.Medium
        elif size == DatasetSize.Small:
            return DatasetSizeNumber.Small
        elif size == DatasetSize.Experimental:
            return DatasetSizeNumber.Small

    def __get_adjency_matrix_weight(p1,p2,epsilon,lamda) -> float:
        r"""
            Class Function
            Gets the weight of 2 nodes based on lamda and epsilon (see documentation for further information)
            Returns Float
        """
        distance = geodesic(p1,p2).km
        weight = math.exp(-((distance ** 2)/(lamda ** 2)))
        if weight >= epsilon:
            return weight
        else:
            return 0