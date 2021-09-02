from Scripts.datasetsClasses import CustomDataset, LinearRegressionDataset, STConvDataset
from Scripts.data_proccess import DataReader, DatasetSize, Graph
from Scripts.learn import Learn, LossFunction,  ModelType, OptimiserType

if __name__ == '__main__':
    
    path_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    path_processed_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed"
    checkpoint_LR = "E:\\FacultateMasterAI\\Dissertation-GNN\\Checkpoint_LR"
    graph_info_txt = "d07_text_meta_2021_03_27.txt"
    datareader = DataReader(path_data,graph_info_txt)
    

    config_partial = {
        "batch_size": 32,
        "hidden_channels": 64,
        "K" : 1,
        "epsilon" : 0.5,
        "optimizer_type" : OptimiserType.Adam,
        "lamda" : 5}
    
    param = {
            "learning_rate" : 0.01,
            "num_features" : 2,
            "EarlyStoppingPatience" : 10,
            "path_data" : path_data,
            "proccessed_data_path" : path_processed_data,
            "graph_info_txt" : graph_info_txt,
            "nb_epoch" : 200,
            "datareader" : datareader,
            "nodes_size" : DatasetSize.Medium,
            "train_ratio" : 0.6,
            "val_ratio" : 0.2,
            "test_ratio" : 0.2,
            "checkpoint_LR" : "E:\\FacultateMasterAI\\Dissertation-GNN\\Checkpoint_LR",
            "checkpoint_dir" : None
        }

    info = {
        "criterion": LossFunction.MAE,
        "model_type" : ModelType.LinearRegression
    }

    if Graph.need_load(path_processed_data) or STConvDataset.need_load(path_processed_data) or CustomDataset.need_load(path_processed_data) or LinearRegressionDataset.need_load(path_processed_data):
        datareader.start()
        Learn.set_data(config_partial,info,param)
        
    Learn.start(config_partial,info,param)