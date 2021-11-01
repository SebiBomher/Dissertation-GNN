from datetime import datetime
import os
from enum import Enum

class Constants():
    r"""
        Class for constant variables
    """
    
    num_features : int = 2
    device : str = "cpu"

    # Training parameters
    learning_rate : float = 0.01
    EarlyStoppingPatience : int = 10
    nb_epoch : int = 300
    batch_size : int = 8
    hidden_channels : int = 8
    time_steps = 1
    
    #Hyper Parameter Tuning Parameters
    num_samples : int = 16
    grace_period : int = 100
    reduction_factor : int = 3

    # Dataset split
    train_ratio : float = 0.6
    val_ratio : float =  0.2
    test_ratio : float = 0.2

    # Folders
    data_folder : str = "Data"
    proccessed_data_folder : str = "Proccessed"
    checkpoint_LR_folder : str = "Checkpoint_LR"
    graph_info_txt : str = "d07_text_meta_2021_03_27.txt"
    results_folder : str = "Results"
    results_ray_folder : str = "Results-RAY"
    folder_save_plots : str = "Plots"
    checkpoin_ARIMA_folder : str = "Checkpoint_ARIMA"
    checkpoin_VARMAX_folder : str = "Checkpoint_VARMAX"

class Folders():
    r"""
        Class for folders and paths
    """
    
    __current_directory = os.getcwd()
    path_data = os.path.join(__current_directory,Constants.data_folder)
    proccessed_data_path = os.path.join(__current_directory,Constants.proccessed_data_folder)
    graph_info_path = os.path.join(__current_directory,Constants.data_folder,Constants.graph_info_txt)
    results_path = os.path.join(__current_directory,Constants.results_folder)
    results_ray_path = os.path.join(__current_directory,Constants.results_ray_folder)
    path_save_plots = os.path.join(__current_directory,Constants.folder_save_plots)

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


class ModelType(Enum):
    r"""
        Enumeration for each model type.
            Custom = 0
            STCONV = 1
            LinearRegression = 2
    """
    Custom = 0
    STCONV = 1
    LinearRegression = 2
    ARIMA = 3
    VARMAX = 4
    RNN = 5
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