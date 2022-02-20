import torch


def configure_device():
    """
    check if GPU is available and use it
    :return: device cuda or cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    return device


def hyperparameter():
    """
    define hyper parameters
    :return: hyper parameters of type tuple
    """
    lr = 0.001
    epochs = 20
    batch_size = 100
    return lr, epochs, batch_size
