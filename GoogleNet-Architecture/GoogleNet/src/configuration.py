import torch
from typing import Dict


def configureDevice() -> dict:
    """
    Check if GPU is available and use it else use CPU
    :return: device cuda or cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    return {'device': device}


def hyperParameter() -> dict:
    """
    static configuration of hyper parameters
    :return: hyper parameters of type tuple
    """
    lr = 0.001
    epochs = 20
    batch_size = 100
    return {'lr': lr, 'epochs': epochs, 'batch_size': batch_size}
