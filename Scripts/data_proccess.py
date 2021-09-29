import math
import os
import numpy as np
from random import sample
from typing import Tuple, Union
from geopy.distance import geodesic
from glob import glob
from enum import Enum
import pandas as pd
from sklearn.preprocessing import normalize
class DatasetSizeNumber(Enum):
    r"""
        Number of DatasetSizes, the total number of nodes per each dataset type.

            Medium = 480
            Small = 120
            Experimental = 5
    """

    Medium = 480
    Small = 120
    Experimental = 8


class DatasetSize(Enum):
    r"""
        Types of Dataset Sizes.
            Experimental = 0
            Small = 1
            Medium = 2
            Full = 3
    """

    Experimental = 0
    Small = 1
    Medium = 2
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

    interval_per_day = 288
    def __init__(self,path_raw_data : str,graph_info_txt : str) -> None:
        r"""
            Constructor, reads all the necessary data.
            Args : 
                __path_raw_data , string :  Path of raw data
                graph_info_txt , string :  Txt file for the nodes metadata
        """

        self.__path_raw_data = path_raw_data
        self.__graph_info_txt = graph_info_txt
        self.good_nodes = []
        self.empty_nodes = []
        self.X = []
        self.Y = []
        self.nodes_location = []
        self.nb_days = 0

    def results(self, results_path)-> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        dfLR = pd.read_csv(os.path.join(results_path,"LinearRegression.csv"), header=False)
        dfSTCONV = pd.read_csv(os.path.join(results_path,"STCONV.csv"), header=False)
        dfCUSTOM = pd.read_csv(os.path.join(results_path,"Customm.csv"), header=False)
        return dfLR,dfSTCONV,dfCUSTOM

    def visualization(self) -> Tuple[pd.DataFrame,pd.DataFrame]:
        return self.__read_visualization()

    def start(self):
        self.__get_good_empty_nodes()
        self.__read_data()
        self.__read_nodes_data()

    def __read_visualization(self) -> Tuple[pd.DataFrame,pd.DataFrame]:
        r"""
            Instance function.
            Set data and labes (X and Y) from Data (may contain data from empty nodes).
            Returns Nothing.
        """
        # 52 columns
        columnsInfo = ['Timestamp',
                    'Station',
                    'District',
                    'Freeway',
                    'DirOfTravel',
                    'LaneType',
                    'Length',
                    'Samples',
                    'Observed',
                    'Flow',
                    'Occupancy',
                    'Speed']
        for i in range(1,9):
            columnsInfo.extend([str(i) + '_Samples',
                            str(i) + '_Flow',
                            str(i) + '_Occupancy',
                            str(i) + '_Speed',
                            str(i) + '_Observed'])
        columnsMetadata = ['ID',
                            'Fwy',
                            'Dir',
                            'District',
                            'County',
                            'City',
                            'State_PM',
                            'Abs_PM',
                            'Latitude',
                            'Longitude',
                            'Length',
                            'Type',
                            'Lanes',
                            'Name',
                            'User_ID_1',
                            'User_ID_2',
                            'User_ID_3',
                            'User_ID_4']
        txtFiles = os.path.join(self.__path_raw_data,"*","*.txt")
        info_data =  os.path.join(self.__path_raw_data,self.__graph_info_txt)
        print("Reading Metadata")
        dataframeMetadata = pd.read_csv(info_data,sep = '\t', skiprows=1, header=None, names = columnsMetadata)
        print("Finished reading Metadata")
        print("Reading Information")
        nb_days = 0
        dataframeInfo = pd.DataFrame(columns = columnsInfo)
        for file in glob(txtFiles):
            print("Reading day {0}".format(nb_days + 1))
            with open(file) as f:
                dataframeInfo=dataframeInfo.append(pd.read_csv(file, sep = ',', header=None,names = columnsInfo),ignore_index=True)
                nb_days += 1
            if nb_days == 5:
                break
        print("Finished Reading Information")
        return dataframeInfo,dataframeMetadata

    def __get_number_of_nodes(self) -> None:
        r"""
            Instance function.
            Set total number of nodes (nb_days) from Metadata (may contain empty nodes).
            Returns Nothing.
        """

        info_data = os.path.join(self.__path_raw_data,self.__graph_info_txt)
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
                    if not (line[9] == '' or line[10] == '' or line[11] == ''):
                        good_nodes.append((int)(line[1]))
                    else:
                        empty_nodes.append((int)(line[1]))
                    index -=1
                    if index == 0:
                        self.good_nodes = good_nodes
                        self.empty_nodes = empty_nodes
                        return

    def get_clean_data_by_nodes(self, size : DatasetSize,path_proccessed_data : str) -> Union[list,list]:
        assert len(self.X) > Graph.get_number_nodes_by_size(size)
        new_X = []
        new_Y = []
        nodes_index = 0
        nodes_ids = Graph.get_nodes_ids_by_size(path_proccessed_data,size)
        for index,tuple in enumerate(zip(self.X,self.Y)):
            if int(self.nodes_location[nodes_index][0]) in nodes_ids:
                new_X.append(tuple[0])
                new_Y.append([tuple[1]])
            nodes_index +=1
            if nodes_index == len(self.nodes_location):
                nodes_index = 0
        return new_X,new_Y

    def __read_data(self) -> None :
        r"""
            Instance function.
            Set data and labes (X and Y) from Data (may contain data from empty nodes).
            Returns Nothing.
        """

        X = []
        Y = []
        txtFiles = os.path.join(self.__path_raw_data,"*","*.txt")
        nb_days = 0
        for file in glob(txtFiles):
            with open(file) as f:
                print("Reading day {0}".format(nb_days + 1))
                content = f.readlines()
                for line in content:
                    line = line.split(',')
                    line = [line1.replace("\n","") for line1 in line]
                    if not(line[9] == '' or line[10] == '' or line[11] == ''):
                        Y.append((float)(line[11]))
                        X.append([(float)(line[9]),(float)(line[10])])
            nb_days += 1
        self.X = normalize(np.array(X))
        self.Y = Y
        self.nb_days = nb_days

    def __read_nodes_data(self) -> None :
        r"""
            Instance function.
            Set nodes_location from Metadata (may contain data from empty nodes).
            Returns Nothing.
        """
        
        info_data =  os.path.join(self.__path_raw_data,self.__graph_info_txt)
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
                    if (int)(line[0]) in self.good_nodes:
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

    epsilon_array = [0.1, 0.3, 0.5, 0.7]
    lamda_array = [1, 3, 5, 10]

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
        assert self.__lamda in self.lamda_array and self.__epsilon in self.epsilon_array

    def need_load(path_processed_data):
        return len(Graph.__get_tuple_to_add_nodes(path_processed_data)) > 0 and len(Graph.__get_tuple_to_add_graph(path_processed_data)) > 0

    def __get_tuple_to_add_nodes(path_processed_data):
        sizes_to_add = []
        for size in DatasetSize:
            name_nodes = os.path.join(path_processed_data,'nodes_{0}.npy'.format(str(size.name)))
            if not os.path.isfile(name_nodes):
                sizes_to_add.append(size)
        return sizes_to_add

    def __get_tuple_to_add_graph(path_processed_data):
        list_to_add = []

        for epsilon in Graph.epsilon_array:
            for lamda in Graph.lamda_array:
                for size in DatasetSize:
                    name_weight = os.path.join(path_processed_data,'Data_EdgeWeight','weight_{0}_{1}_{2}.npy'.format(str(epsilon),str(lamda),str(size.name)))
                    name_index = os.path.join(path_processed_data,'Data_EdgeIndex','index_{0}_{1}_{2}.npy'.format(str(epsilon),str(lamda),str(size.name)))
                    if not(os.path.isfile(name_index) and os.path.isfile(name_weight)):
                        list_to_add.append([epsilon,lamda,size])
        return list_to_add

    def __set_nodes(self) -> None:
        r"""
            Instance function.
            Sets nodes array of ids for each size, so the ids will always be the same.
            Returns Nothing.
        """
        good_nodes = self.__data_reader.good_nodes
        sizes_to_add = Graph.__get_tuple_to_add_nodes(self.__path_processed_data)
        for size in sizes_to_add:
            if size != DatasetSize.Experimental:
                number_of_nodes =  Graph.get_number_nodes_by_size(size)
                nodes = sample(good_nodes, number_of_nodes)
            else:
                nodes = [718292,769496,718291,718290,764567,774279,774278,764671]
            name_nodes = os.path.join(self.__path_processed_data,'nodes_{0}.npy'.format(str(size.name)))
            np.save(name_nodes,nodes)

    def get_nodes_ids_by_size(path_processed_data : str,size : DatasetSize) -> list:
        r"""
            Instance function.
            Returns graph nodes ids based by size.
            Returns list.
        """
        name_nodes = os.path.join(path_processed_data,'nodes_{0}.npy'.format(str(size.name)))
        return np.load(name_nodes)

    def __process_graph_info(self) -> None:
        r"""
            Instance function.
            Checks if the data required is available, if not it creates it.
            Returns Nothing.
        """
        nodes_location = self.__data_reader.nodes_location
        list_to_add = Graph.__get_tuple_to_add_graph(self.__path_processed_data)
        for info in list_to_add:
            epsilon = info[0]           
            lamda = info[1]           
            size = info[2]
            self.__save_graph(nodes_location,epsilon,lamda,size)  


    def __save_graph(self,nodes_location,epsilon,lamda,size : DatasetSize) -> None:
        r"""
            Instance function.
            Save a graph by a configuration.
            Returns Nothing.
        """

        nodes = Graph.get_nodes_ids_by_size(self.__path_processed_data,size)
        edge_index = []
        edge_weight = []
        nodes_location = [node for node in nodes_location if (int)(node[0]) in nodes]
        self.num_nodes = len(nodes_location)
        print("Saving graph with configuration : epsilon = {0}, lamda = {1}, size = {2}".format(str(epsilon),str(lamda),str(size.name)))
        for i in range(len(nodes_location) - 1):
            for j in range(i,len(nodes_location) - 1):
                if i != j:
                    p1 = (nodes_location[i][1],nodes_location[i][2])
                    p2 = (nodes_location[j][1],nodes_location[j][2])
                    weight = Graph.__get_adjency_matrix_weight(p1,p2,epsilon,lamda)
                    if weight > 0:
                        edge_index.append([i,j])
                        edge_weight.append(weight)

        edge_index = np.transpose(edge_index)
        name_folder_weight = os.path.join(self.__path_processed_data,'Data_EdgeWeight')
        name_folder_index = os.path.join(self.__path_processed_data,'Data_EdgeIndex')
        if not os.path.exists(name_folder_weight):
            os.makedirs(name_folder_weight)

        if not os.path.exists(name_folder_index):
            os.makedirs(name_folder_index)
        
        name_weight = os.path.join(name_folder_weight,'weight_{0}_{1}_{2}.npy'.format(str(epsilon),str(lamda),str(size.name)))
        name_index = os.path.join(name_folder_index,'index_{0}_{1}_{2}.npy'.format(str(epsilon),str(lamda),str(size.name)))
        np.save(name_index,edge_index)
        np.save(name_weight,edge_weight)

    def __set_graph_info(self) -> None:
        r"""
            Instance function.
            Sets the edge_index, edge_weight and num_nodes.
            Returns Nothing.
        """
        name_weight = os.path.join(self.__path_processed_data,'Data_EdgeWeight','weight_{0}_{1}_{2}.npy'.format(str(self.__epsilon),str(self.__lamda),str(self.__size.name)))
        self.edge_weight = np.load(name_weight)

        name_index = os.path.join(self.__path_processed_data,'Data_EdgeIndex','index_{0}_{1}_{2}.npy'.format(str(self.__epsilon),str(self.__lamda),str(self.__size.name)))
        self.edge_index = np.load(name_index)

    def get_number_nodes_by_size(size : DatasetSize) -> int:
        r"""
            Class function.
            Returns the number of nodes by size
            Returns Integer
        """
        if size == DatasetSize.Medium:
            return DatasetSizeNumber.Medium.value
        elif size == DatasetSize.Small:
            return DatasetSizeNumber.Small.value
        elif size == DatasetSize.Experimental:
            return DatasetSizeNumber.Experimental.value

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