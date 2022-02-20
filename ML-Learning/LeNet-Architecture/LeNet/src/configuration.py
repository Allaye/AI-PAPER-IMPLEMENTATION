import torch


def configure_device() -> torch.device:
    """
    check if GPU is available and use it else use CPU
    :return: device cuda or cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    return device


def hyperparameter() -> tuple:
    """
    static configuration of hyperparameters
    :return: hyper parameters of type tuple
    """
    lr = 0.001
    epochs = 20
    batch_size = 100
    return lr, epochs, batch_size
