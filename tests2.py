from Scripts.learn import learn,MSE,MAPE,MAE,RMSE, test
from ray import tune
from ray.tune.schedulers import ASHAScheduler

if __name__ == '__main__':
    num_samples = 16
    config_full = {
        "batch_size": tune.choice([8,16,32]),
        "hidden_channels": tune.choice([8,16,32,64]),
        "K" : tune.choice([1,3,5,7]),
        "epsilon" : tune.choice([0.1, 0.3, 0.5, 0.7]),
        "optimizer_type" : tune.choice(["Adam","RMSprop","Adamax"]),
        "lamda" : tune.choice([1, 3, 5, 10])
        }
    config_partial = {
        "batch_size": 16,
        "hidden_channels": 32,
        "K" : 1,
        "epsilon" : 0.5,
        "optimizer_type" : "Adam",
        "lamda" : 5}
    nb_epoch = 200
    time_steps = [1,3,5,7]
    criterions = [MSE,MAE,MAPE,RMSE]
    nodes_size = ["Full","Medium","Small","Experimental"]
    model_type = ["STCONV","ASTGCN","MSTGCN","GMAN","",""]
    learn(config_partial, time_step = 1, criterion = MAE, nb_epoch = nb_epoch)
