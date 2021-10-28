#region Imports

from Scripts.Utility import Constants, DatasetSize, Folders, ModelType
from Scripts.Learn import Learn
from Scripts.DataProccess import DataReader
from Scripts.DataVisualization import DataViz

#endregion

if __name__ == '__main__':

    Folders().CreateFolders()
    datareader = DataReader()

    Learn.set_data(datareader=DataReader)

    Learn.startLR(datareader=datareader)

    for datasize in DatasetSize:
        for model in ModelType:
            if model != ModelType.LinearRegression:
                Learn.HyperParameterTuning(
                    datasetsize=datasize, model=model, datareader=datareader)

    DataViz.Run()
