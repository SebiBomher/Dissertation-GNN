import torch


def MSE(y_true,y_pred):
    return torch.mean((y_pred-y_true)**2)