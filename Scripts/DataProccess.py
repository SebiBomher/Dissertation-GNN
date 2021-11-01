#region Imports

import math
import os
import numpy as np
import pandas as pd
from random import sample
from typing import Tuple, Union
from geopy.distance import geodesic
from glob import glob
from sklearn.preprocessing import normalize
from Scripts.Utility import DatasetSize, DatasetSizeNumber, Folders

#endregion

class DataReader():
    r"""
        DataReader Class, Once initialized it will read data once and will be able to access data without re-reading additional data

        Variables:
            __path_raw_data , string : Path of raw data
            __graph_info_txt , string : Txt file for the nodes metadata
            X , List : Data
            Y , List : Labels
            nb_days , integer: Total number of days for the given data
            nodes_location , list[int, float, float] : geographic location of each node containing the id, x coordinate and y coordinate (note : it can contain empty nodes, must be processed)
            empty_nodes , list : Nodes which do not contain data
            good_nodes , list : Nodes which contain data

        Instance Functions:
            start(), Starts the reading procedure
            results(), Reads the csv results and combines the Custom and STCONV results into 2 csv files
            visualization(), Reads the Metadata information and Data information for data visualization
            get_clean_data_by_nodes(),
            __get_number_of_nodes(), set total number of nodes (nb_days) from Metadata (may contain empty nodes)
            __read_data(), set data and labels (X and Y) from Data (may contain data from empty nodes)
            __get_good_empty_nodes(), set empty_nodes and good_nodes from Data (may contain data from empty nodes)
            __read_nodes_data(), set nodes_location from Metadata (may contain data from empty nodes)
    """

    #region Constructors & Properties

    interval_per_day = 288

    def __init__(self) -> None:
        r"""
            Constructor
        """

        self.__path_raw_data = Folders.path_data
        self.__graph_info_txt = Folders.graph_info_path
        self.__results_path = Folders.results_path
        self.__path_proccessed_data = Folders.proccessed_data_path
        self.good_nodes = []
        self.empty_nodes = []
        self.X = []
        self.Y = []
        self.nodes_location = []
        self.nb_days = 0
        if (not os.path.isdir(os.path.join(self.__path_proccessed_data,"STCONV")) or 
        not os.path.isdir(os.path.join(self.__path_proccessed_data,"Custom")) or 
        not os.path.isdir(os.path.join(self.__path_proccessed_data,"Data_EdgeWeight")) or
        not os.path.isdir(os.path.join(self.__path_proccessed_data,"Data_EdgeIndex")) or
        not os.path.isdir(os.path.join(self.__path_proccessed_data,"LinearRegression"))):
            self.start()
            Graph(epsilon=0.1,sigma=3,size=DatasetSize.Medium,data_reader=self)
    #endregion
    
    #region Instance Functions

    def results(self,experiment_name : str)-> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        r"""
            Reads the csv results and combines the Custom and STCONV results into 2 csv files.
            Instance Function.
            No Arguments.
            Returns a Tuple of 3 pandas Dataframe, one for each model.
        """
        experiment_path = os.path.join(self.__results_path,experiment_name)
        dfLR = pd.read_csv(os.path.join(experiment_path,"LinearRegression.csv"))
        columnsInfo = ["Model", "Epsilon", "Sigma", "Size", "Criterion", "Loss", "Epoch", "OptimizerType", "Trial", "TestOrVal"]

        STCONVFile = os.path.join(experiment_path,"STCONV.csv")
        if not(os.path.exists(STCONVFile)):
            dataframeInfo = pd.DataFrame(columns = columnsInfo)
            STConvFiles = os.path.join(experiment_path,"STCONV_*_*.csv")
            for file in glob(STConvFiles):
                with open(file) as f:
                    dataframeInfo = dataframeInfo.append(pd.read_csv(file, sep = ',', header=None,names = columnsInfo,  skiprows=1),ignore_index=True)
            dataframeInfo.to_csv(STCONVFile)

        CustomFile = os.path.join(experiment_path,"CUSTOM.csv")
        if not(os.path.exists(CustomFile)):
            dataframeInfo = pd.DataFrame(columns = columnsInfo)
            CustomFiles = os.path.join(experiment_path,"CUSTOM*_*.csv")
            for file in glob(CustomFiles):
                with open(file) as f:
                    dataframeInfo = dataframeInfo.append(pd.read_csv(file, sep = ',', header=None,names = columnsInfo,  skiprows=1),ignore_index=True)
            dataframeInfo.to_csv(CustomFile)

        dfSTCONV = pd.read_csv(os.path.join(experiment_path,"STCONV.csv"))
        dfCUSTOM = pd.read_csv(os.path.join(experiment_path,"CUSTOM.csv"))
        return dfLR,dfSTCONV,dfCUSTOM

    def start(self) -> None:
        r"""
            Starts the reading procedure.
            Instance Function.
            No arguments.
            Returns None.
        """

        self.__get_good_empty_nodes()
        self.__read_data()
        self.__read_nodes_data()

    def visualization(self) -> Tuple[pd.DataFrame,pd.DataFrame]:
        r"""
            Reads the Metadata information and Data information for data visualization.
            Instance Function.
            No Arguments.
            Returns a Tuple of 2 Dataframes with Metadata information and Data information.
        """

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
        print("Finished Reading Metadata")
        print("Reading Information")
        nb_days = 0
        dataframeInfo = pd.DataFrame(columns = columnsInfo)

        for file in glob(txtFiles):
            print("Reading day {0}".format(nb_days + 1))
            with open(file) as f:
                dataframeInfo = dataframeInfo.append(pd.read_csv(file, sep = ',', header=None,names = columnsInfo),ignore_index=True)
                day = None
                nb_days += 1
            
            if nb_days == 5:
                break

        print("Finished Reading Information")
        return dataframeInfo ,dataframeMetadata

    def __get_number_of_nodes(self) -> None:
        r"""
            Set total number of nodes (nb_days) from Metadata (may contain empty nodes).
            Instance Function.
            No Arguments.
            Returns None.
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
            Set empty_nodes and good_nodes from Data (may contain data from empty nodes).
            Instance Function.
            No Arguments.
            Returns None.
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

    def get_clean_data_by_nodes(self, size : DatasetSize) -> Tuple[list,list]:
        r"""
            Returns data for a specific datasize.
            Instance Function.
            Args:
                size : DatasetSize
            Returns a tuple of 2 lists with the Data and Labels
        """

        new_X = []
        new_Y = []
        nodes_index = 0
        nodes_ids = Graph.get_nodes_ids_by_size(size)
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
            Set data and labels (X and Y) from Data (may contain data from empty nodes).
            Instance Function.
            No Arguments.
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
            if nb_days == 5:
                break
        self.X = normalize(np.array(X))
        self.Y = Y
        self.nb_days = nb_days

    def __read_nodes_data(self) -> None :
        r"""
            Set nodes_location from Metadata (may contain data from empty nodes).
            Instance Function.
            No Arguments.
            Returns None.
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

    #endregion

class Graph():
    r"""
        Graph class, contains data for graph construction in pytorch.
        For the first time instancing this class all data will be saved in path_processed_data for further use, such that there wont be any redundant processing

        Args: 
            path_processed_data : str , Path where the processed data will be saved
            epsilon : float , epsilon for weight calculation (see documentation for further information)
            sigma : int , sigma for weight calculation (see documentation for further information)
            size : DatasetSize , Size of the graph
            data_reader : DataReader , Helper class for data Reading

        Static Variables:
            sigma_array : list = [0.1, 0.3, 0.5, 0.7] - sigma possibilities
            epsilon_array : list = [1, 3, 5, 10] - epsilon possibilities

        Variables:
            __path_processed_data : str , Path where the processed data will be saved
            __epsilon : float , epsilon for weight calculation (see documentation for further information)
            __sigma : integer , sigma for weight calculation (see documentation for further information)
            __size : DatasetSize class , Size of the graph
            __data_reader : DataReader class , Helper class for data Reading
            num_nodes : integer , Total number of nodes for the graph per size given
            edge_index : list (2,num_nodes) , list of graph edges for pytorch
            edge_weight : list (num_nodes) , list of graph weights for pytorch

        Instance Functions:
            __check_sigma_epsilon(), Checks if sigma and epsilon are valid.
            __set_nodes(), Sets nodes array of ids for each size, so the ids will always be the same.
            __process_graph_info(), Checks if the data required is available, if not it creates it.
            __save_graph(), Save a graph by a configuration.
            __set_graph_info(), Sets the edge_index, edge_weight and num_nodes.

        Class Functions:
            need_load(), Checks if Graph information has been proccessed.
            __get_tuple_to_add_nodes(), Function which retrieves which datasets need nodes to be implemented and saved.
            __get_tuple_to_add_graph(), Function which retrieves which graphs need preparing to be implemented and saved.
            get_nodes_ids_by_size(),  Returns graph nodes ids based by size.
            get_number_nodes_by_size(), Returns the number of nodes by size.
            __get_adjency_matrix_weight(), Gets the weight of 2 nodes based on sigma and epsilon (see documentation for further information).
    """
    #region Constructors & Properties

    epsilon_array = [0.1, 0.3, 0.5, 0.7]
    sigma_array = [1, 3, 5, 10]

    def __init__(self, epsilon : float, sigma : int ,size : DatasetSize, data_reader : DataReader) -> None :
        r"""
            Constructor, makes the processing and data saving.

            Args: 
                epsilon , float : epsilon for weight calculation (see documentation for further information)
                sigma , integer : sigma for weight calculation (see documentation for further information)
                size , DatasetSize class : Size of the graph
                data_reader , DataReader class : Helper class for data Reading
        """
        self.__path_processed_data = Folders.proccessed_data_path
        self.__epsilon = epsilon
        self.__sigma = sigma
        self.__size = size
        self.__data_reader = data_reader
        self.edge_index = []
        self.edge_weight = []
        self.num_nodes = 0
        self.nodes_id_size_array = []
        self.__check_sigma_epsilon()
        self.__set_nodes()
        self.__process_graph_info()
        self.__set_graph_info()
        
    #endregion

    #region Instance Functions

    def __check_sigma_epsilon(self) -> None:
        r"""
            Checks if sigma and epsilon are valid.
            Instance function.
            No Arguments.
            Returns Nothing.
        """
        assert self.__sigma in self.sigma_array and self.__epsilon in self.epsilon_array

    def __set_nodes(self) -> None:
        r"""
            Sets nodes array of ids for each size, so the ids will always be the same.
            Instance function.
            No Arguments.
            Returns None.
        """
        good_nodes = self.__data_reader.good_nodes
        sizes_to_add = Graph.__get_tuple_to_add_nodes()
        for size in sizes_to_add:
            if size != DatasetSize.Experimental:
                number_of_nodes =  Graph.get_number_nodes_by_size(size)
                nodes = sample(good_nodes, number_of_nodes)
            else:
                nodes = [718292,769496,718291,718290,764567,774279,774278,764671]
            name_nodes = os.path.join(self.__path_processed_data,'nodes_{0}.npy'.format(str(size.name)))
            np.save(name_nodes,nodes)

    def __process_graph_info(self) -> None:
        r"""
            Checks if the data required is available, if not it creates it.
            Instance function.
            No Arguments.
            Returns Nothing.
        """
        nodes_location = self.__data_reader.nodes_location
        list_to_add = Graph.__get_tuple_to_add_graph()
        for info in list_to_add:
            epsilon = info[0]           
            sigma = info[1]           
            size = info[2]
            self.__save_graph(nodes_location,epsilon,sigma,size)

    def __save_graph(self,nodes_location : list ,epsilon : float ,sigma : int ,size : DatasetSize) -> None:
        r"""
            Save a graph by a configuration.
            Instance function.
            Args:
                nodes_location : list, list of geographic location for each node
                epsilon : float, epsilon from the epsilon array for which to save
                sigma : int, sigma from sigma array for which to save
                size : DatasetSize, Dataset size for which to save
            Returns None.
        """

        nodes = Graph.get_nodes_ids_by_size(size)
        edge_index = []
        edge_weight = []
        nodes_location = [node for node in nodes_location if (int)(node[0]) in nodes]
        self.num_nodes = len(nodes_location)
        print("Saving graph with configuration : epsilon = {0}, sigma = {1}, size = {2}".format(str(epsilon),str(sigma),str(size.name)))
        for i in range(len(nodes_location) - 1):
            for j in range(i,len(nodes_location) - 1):
                if i != j:
                    p1 = (nodes_location[i][1],nodes_location[i][2])
                    p2 = (nodes_location[j][1],nodes_location[j][2])
                    weight = Graph.__get_adjency_matrix_weight(p1,p2,epsilon,sigma)
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
        
        name_weight = os.path.join(name_folder_weight,'weight_{0}_{1}_{2}.npy'.format(str(epsilon),str(sigma),str(size.name)))
        name_index = os.path.join(name_folder_index,'index_{0}_{1}_{2}.npy'.format(str(epsilon),str(sigma),str(size.name)))
        np.save(name_index,edge_index)
        np.save(name_weight,edge_weight)

    def __set_graph_info(self) -> None:
        r"""
            Sets the edge_index, edge_weight and num_nodes.
            Instance function.
            No Arguments.
            Returns Nothing.
        """
        name_weight = os.path.join(self.__path_processed_data,'Data_EdgeWeight','weight_{0}_{1}_{2}.npy'.format(str(self.__epsilon),str(self.__sigma),str(self.__size.name)))
        self.edge_weight = np.load(name_weight)

        name_index = os.path.join(self.__path_processed_data,'Data_EdgeIndex','index_{0}_{1}_{2}.npy'.format(str(self.__epsilon),str(self.__sigma),str(self.__size.name)))
        self.edge_index = np.load(name_index)

    #endregion

    #region Class Functions

    def need_load() -> bool:
        r"""
            Checks if Graph information has been proccessed
            Class Function.
            No Arguments.
            Returns bool.
        """
        return len(Graph.__get_tuple_to_add_nodes()) > 0 and len(Graph.__get_tuple_to_add_graph()) > 0
    
    def __get_tuple_to_add_nodes() -> list:
        r"""
            Function which retrieves which datasets need nodes to be implemented and saved.
            Class Function.
            No Arguments.
            Returns list.
        """
        sizes_to_add = []
        for size in DatasetSize:
            name_nodes = os.path.join(Folders.proccessed_data_path,'nodes_{0}.npy'.format(str(size.name)))
            if not os.path.isfile(name_nodes):
                sizes_to_add.append(size)
        return sizes_to_add

    def __get_tuple_to_add_graph() -> list:
        r"""
            Function which retrieves which graphs need preparing to be implemented and saved.
            Class Function.
            No Arguments.
            Returns list.
        """
        list_to_add = []

        for epsilon in Graph.epsilon_array:
            for sigma in Graph.sigma_array:
                for size in DatasetSize:
                    name_weight = os.path.join(Folders.proccessed_data_path,'Data_EdgeWeight','weight_{0}_{1}_{2}.npy'.format(str(epsilon),str(sigma),str(size.name)))
                    name_index = os.path.join(Folders.proccessed_data_path,'Data_EdgeIndex','index_{0}_{1}_{2}.npy'.format(str(epsilon),str(sigma),str(size.name)))
                    if not(os.path.isfile(name_index) and os.path.isfile(name_weight)):
                        list_to_add.append([epsilon,sigma,size])
        return list_to_add

    def get_nodes_ids_by_size(size : DatasetSize) -> list:
        r"""
            Instance function.
            Returns graph nodes ids based by size.
            Returns list.
        """
        name_nodes = os.path.join(Folders.proccessed_data_path,'nodes_{0}.npy'.format(str(size.name)))
        return np.load(name_nodes)

    def get_number_nodes_by_size(size : DatasetSize) -> int:
        r"""
            Returns the number of nodes by size.
            Class function.
            Args:
                size : DatasetSize, the dataset size for which to return the number of nodes
            Returns Integer.
        """
        if size == DatasetSize.Medium:
            return DatasetSizeNumber.Medium.value
        elif size == DatasetSize.Small:
            return DatasetSizeNumber.Small.value
        elif size == DatasetSize.Experimental:
            return DatasetSizeNumber.Experimental.value

    def __get_adjency_matrix_weight(p1 : tuple,p2 : tuple,epsilon : float ,sigma : int) -> float:
        r"""
            Gets the weight of 2 nodes based on sigma and epsilon (see documentation for further information).
            Class Function.
            Args:
                p1 : tuple, First point which contains the x and y geographical coordinates
                p2 : tuple, Second point which contains the x and y geographical coordinates
                epsilon : float, epsilon from epsilon array
                sigma : int, sigma from epsilon array
            Returns Float.
        """
        distance = geodesic(p1,p2).km
        weight = math.exp(-((distance ** 2)/(sigma ** 2)))
        if weight >= epsilon:
            return weight
        else:
            return 0

    #endregion

    