import ray
from Scripts.learn import learn,MSE,MAPE,MAE,RMSE, test
from ray import tune
from ray.tune.schedulers import ASHAScheduler

if __name__ == '__main__':
    num_samples = 16
    config = {
        "batch_size": tune.choice([8,16,32]),
        "hidden_channels": tune.choice([8,16,32,64]),
        "K" : tune.choice([1,3,5,7]),
        "epsilon" : tune.choice([0.1, 0.3, 0.5, 0.7]),
        "optimizer_type" : tune.choice(["Adam","RMSprop","Adamax"]),
        "lamda" : tune.choice([1, 3, 5, 10])
        }
    nb_epoch = 200
    time_steps = [1,3,5,7]
    criterions = [MSE,MAE,MAPE,RMSE]
    for crit in criterions:
        for time_step in time_steps:

            scheduler = ASHAScheduler(
                max_t=nb_epoch,
                grace_period=nb_epoch,
                reduction_factor=3)

            result = tune.run(
                tune.with_parameters(learn, time_step = 1, criterion = MSE),
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
