from datetime import datetime
import os
from enum import Enum
import numpy as np

class Constants():
    r"""
        Class for constant variables
    """

    num_features: int = 2
    device: str = "cpu"

    # Training parameters
    learning_rate: float = 0.01
    EarlyStoppingPatience: int = 5
    nb_epoch: int = 30
    batch_size: int = 8
    time_steps = 1

    #Hyper Parameter Tuning Parameters
    num_samples: int = 16
    grace_period: int = 30
    reduction_factor: int = 3

    # Dataset split
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # Folders
    data_folder: str = "Data"
    proccessed_data_folder: str = "Proccessed"
    checkpoint_LR_folder: str = "Checkpoint_LR"
    graph_info_txt: str = "d07_text_meta_2021_03_27.txt"
    results_folder: str = "Results"
    results_ray_folder: str = "Results-RAY"
    folder_save_plots: str = "Plots"
    checkpoin_ARIMA_folder: str = "Checkpoint_ARIMA"
    checkpoin_SARIMA_folder: str = "Checkpoint_SARIMA"
    nodes_Experimental: list = [718292, 769496,
                                718291, 718290, 764567, 774279, 774278, 764671]
    edge_index_Experimental_manual = [[0, 1, 7, 4, 7, 5], [1, 2, 4, 3, 6, 4]]
    edge_weight_Experimental_manual = np.ones(len(edge_index_Experimental_manual[0]))
    nodes_Tiny: list = [775637, 718165, 776986, 759289, 774672, 760643, 774671, 717046, 718419, 769105, 764026, 759280, 775636, 759385, 760635, 718166, 774685, 774658, 716938, 776177, 763453, 718421, 717045, 768598,
                        717043, 716063, 717041, 717040, 717039, 737184, 717042, 718335, 763458, 776981, 737158, 737313, 769118, 772501, 718173, 764037, 763447, 763246, 718041, 763251, 763424, 763429, 763434, 763439, 764032, 764418]
    edge_index_Tiny_manual: list = [[0, 0, 0, 1, 5, 5, 5, 9, 9, 9, 10, 10, 10, 6, 14, 15, 7, 13, 10, 11, 8, 9, 4, 16, 5, 2, 20, 3, 22, 23, 24, 25, 26, 28, 29, 30, 31, 21, 32, 33, 34, 36, 17, 38, 12, 39, 40, 41, 42, 44, 45, 46, 47, 48, 27, 35, 9], [
        1, 2, 3, 4, 6, 7, 2, 4, 3, 12, 12, 11, 4, 14, 15, 12, 13, 10, 11, 8, 2, 3, 16, 17, 6, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 7, 32, 33, 34, 35, 9, 37, 5, 39, 40, 41, 42, 43, 45, 46, 47, 48, 0, 19, 18, 36]]
    edge_weight_Tiny_manual: list = np.ones(len(edge_index_Tiny_manual[0]))


class Folders():
    r"""
        Class for folders and paths
    """

    __current_directory = os.getcwd()
    path_data = os.path.join(__current_directory, Constants.data_folder)
    proccessed_data_path = os.path.join(
        __current_directory, Constants.proccessed_data_folder)
    graph_info_path = os.path.join(
        __current_directory, Constants.data_folder, Constants.graph_info_txt)
    results_path = os.path.join(__current_directory, Constants.results_folder)
    results_ray_path = os.path.join(
        __current_directory, Constants.results_ray_folder)
    path_save_plots = os.path.join(
        __current_directory, Constants.folder_save_plots)

    def CreateFolders(self) -> None:
        if not os.path.exists(self.path_data):
            os.makedirs(self.path_data)

        if not os.path.exists(self.proccessed_data_path):
            os.makedirs(self.proccessed_data_path)

        if not os.path.exists(self.graph_info_path):
            os.makedirs(self.graph_info_path)

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        if not os.path.exists(self.results_ray_path):
            os.makedirs(self.results_ray_path)

        if not os.path.exists(self.path_save_plots):
            os.makedirs(self.path_save_plots)


class DatasetSizeNumber(Enum):
    r"""
        Number of DatasetSizes, the total number of nodes per each dataset type.

            Medium = 480
            Small = 120
            Experimental = 5
    """
    All = 2789
    Medium = 480
    Small = 120
    Tiny = 50
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
    ExperimentalManual = 1
    ExperimentalLR = 2
    Tiny = 3
    TinyManual = 4
    TinyLR = 5
    Small = 6
    Medium = 7
    All = 8

class ModelType(Enum):
    r"""
        Enumeration for each model type.
            LSTM = 0
            STCONV = 1
            LinearRegression = 2
    """
    LSTM = 0
    DCRNN = 1
    STCONV = 2
    LinearRegression = 3
    ARIMA = 4
    SARIMA = 5


class OptimizerType(Enum):
    r"""
        Enumeration for each optimizer type
            Adam = 0
            RMSprop = 1
            Adamax = 2
            AdamW = 3
    """
    Adam = 0
    RMSprop = 1
    Adamax = 2
    AdamW = 3
