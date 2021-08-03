import os
import numpy as np
from Scripts.models import ST_GCN
from Scripts.learn import learn, train_val_and_test,MSE,MAPE,MAE,RMSE
from Scripts.data_proccess import get_dataset, get_dataset_experimental, get_experimental_data, get_graph_info
import torch
from torch.optim import Adam,SGD,RMSprop,Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
import skopt
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pytorch_lightning as pl
import torch.nn as nn
from functools import partial
from ray.tune.schedulers import ASHAScheduler

if __name__ == '__main__':
    checkpoint_dir = "E:\\FacultateMasterAI\\Dissertation-GNN\\checkpoints"
    num_samples = 16
    config = {
        "batch_size": tune.choice([8,16,32,64,128]),
        "hidden_channels": tune.choice([8,16,32,64,128]),
        "K" : tune.choice([1,3,5,7]),
        "epsilon" : tune.choice([0.1, 0.3, 0.5, 0.7]),
        "optimizer_type" : tune.choice(["Adam","SGD","RMSprop","Adamax"]),
        "lamda" : tune.choice([1, 3, 5, 10])
        }
    nb_epoch = 200
    time_steps = [1,3,5,7]
    criterions = [MSE,MAE,MAPE,RMSE]
    # for crit in criterions:
        # for time_step in time_steps:

    scheduler = ASHAScheduler(
        max_t=nb_epoch,
        grace_period=1,
        reduction_factor=2)

    result = tune.run(
        tune.with_parameters(learn, checkpoint_dir=checkpoint_dir, time_step = 1, criterion = MSE),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
            # print("time step : {0} ; criterion : {1} ; loss : {2}".format(str(time_step),str(criterions.__name__),str(score)))