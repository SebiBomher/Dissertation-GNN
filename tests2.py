from Scripts.datasetsClasses import STConvDataset
from Scripts.data_proccess import DataReader, DatasetSize, Graph
from Scripts.learn import Learn, LossFunction,  ModelType

if __name__ == '__main__':

    path_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    path_processed_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed"
    graph_info_txt = "d07_text_meta_2021_03_27.txt"
    datareader = DataReader(path_data,graph_info_txt)
    if Graph.need_load(path_processed_data) or STConvDataset.need_load(path_processed_data):
        datareader.start()

    config_partial = {
        "batch_size": 16,
        "hidden_channels": 32,
        "K" : 1,
        "epsilon" : 0.5,
        "optimizer_type" : "Adam",
        "lamda" : 5}
    
    param = {
            "learning_rate" : 0.01,
            "num_nodes": 8,
            "num_features" : 3,
            "EarlyStoppingPatience" : 10,
            "path_data" : path_data,
            "path_processed_data" : "D:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed",
            "graph_info_txt" : graph_info_txt,
            "nb_epoch" : 200,
            "datareader" : datareader,
            "nodes_size" : DatasetSize.Experimental
        }

    info = {
        "criterion": LossFunction.MAE,
        "model_type" : ModelType.STCONV
    }

    Learn.start(config_partial,info,param)
