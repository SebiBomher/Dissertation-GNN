#region Imports

from Scripts.Utility import Constants, DatasetSize, Folders, ModelType
from Scripts.Learn import Learn
from Scripts.DataProccess import DataReader
from Scripts.DataVisualization import DataViz

#endregion

if __name__ == '__main__':
    
    Folders().CreateFolders()
    datareader = DataReader(Folders.path_data, Folders.graph_info_path)

    Learn.set_data(
        proccessed_data_path=Folders.proccessed_data_path,
        datareader=DataReader,
        device=Constants.device)

    Learn.startLR(
        path_data=Folders.path_data,
        proccessed_data_path=Folders.proccessed_data_path,
        graph_info_txt=Folders.graph_info_path,
        datareader=datareader,
        checkpoint_LR=Folders.checkpoint_LR_path)

    for datasize in DatasetSize:
        for model in ModelType:
            if model != ModelType.LinearRegression:
                Learn.HyperParameterTuning(
                    datasetsize=datasize, model=model, datareader=datareader)

    DataViz.Run()
