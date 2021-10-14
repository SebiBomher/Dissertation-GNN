from Scripts.data_proccess import DataReader, DatasetSize
from Scripts.learn import Learn, LossFunction, ModelType, OptimiserType
from ray import tune
from ray.tune.schedulers import ASHAScheduler

if __name__ == '__main__':
    num_samples = 16
    path_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    path_processed_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed"
    checkpoint_LR = "D:\\FacultateMasterAI\\Dissertation-GNN\\Checkpoint_LR"
    graph_info_txt = "d07_text_meta_2021_03_27.txt"
    datareader = DataReader(path_data,graph_info_txt)
    
    config = {
        "batch_size": tune.choice([8]),
        "hidden_channels": tune.choice([8]),
        "K" : tune.choice([1,3,5,7]),
        "epsilon" : tune.choice([0.1, 0.3, 0.5, 0.7]),
        "optimizer_type" : tune.choice([OptimiserType.Adam,OptimiserType.AdamW,OptimiserType.Adamax,OptimiserType.RMSprop]),
        "lamda" : tune.choice([1, 3, 5, 10])
        }
    
    nb_epoch = 300
    param = {
            "learning_rate" : 0.01,
            "num_features" : 2,
            "EarlyStoppingPatience" : 10,
            "path_data" : path_data,
            "proccessed_data_path" : path_processed_data,
            "graph_info_txt" : graph_info_txt,
            "nb_epoch" : nb_epoch,
            "datareader" : datareader,
            "nodes_size" : DatasetSize.Medium,
            "train_ratio" : 0.6,
            "val_ratio" : 0.2,
            "test_ratio" : 0.2,
            "checkpoint_LR" : checkpoint_LR,
            "checkpoint_dir" : None
        }

    config_partial = {
        "batch_size": 32,
        "hidden_channels": 64,
        "K" : 1,
        "epsilon" : 0.5,
        "optimizer_type" : OptimiserType.Adam,
        "lamda" : 5
    }
    info = {
        "criterion": LossFunction.MAE,
        "model_type" : ModelType.LinearRegression
    }

    Learn.startCUSTOM(config_partial,info,param)
    
    # for datasize in DatasetSize:

    # param["nodes_size"] = DatasetSize.Medium

    # scheduler = ASHAScheduler(
    #     max_t=nb_epoch,
    #     grace_period=100,
    #     reduction_factor=3)

    # info = {
    #     "criterion": LossFunction.MAE,
    #     "model_type" : ModelType.Custom
    # }

    # config["hidden_channels"] = tune.choice([0])
    # result = tune.run(
    #     tune.with_parameters(Learn.startCUSTOM, info = info, param = param),
    #     resources_per_trial={"cpu": 8, "gpu": 1},
    #     config=config,
    #     metric="loss",
    #     mode="min",
    #     num_samples=num_samples,
    #     scheduler=scheduler
    # )

    # best_trial = result.get_best_trial("loss", "min", "last")
    # print("Best trial config: {}".format(best_trial.config))
    # print("Best trial for Custom model final validation loss: {}".format(best_trial.last_result["loss"]))


    # scheduler = ASHAScheduler(
    #     max_t=nb_epoch,
    #     grace_period=100,
    #     reduction_factor=3)

    # info = {
    #     "criterion": LossFunction.MAE,
    #     "model_type" : ModelType.STCONV
    # }

    # config["hidden_channels"] = tune.choice([8])
    # result = tune.run(
    #     tune.with_parameters(Learn.startSTCONV, info = info, param = param),
    #     resources_per_trial={"cpu": 8, "gpu": 1},
    #     config=config,
    #     metric="loss",
    #     mode="min",
    #     num_samples=num_samples,
    #     scheduler=scheduler
    # )

    # best_trial = result.get_best_trial("loss", "min", "last")
    # print("Best trial config: {}".format(best_trial.config))
    # print("Best trial for Custom model final validation loss: {}".format(best_trial.last_result["loss"]))