import torch


def config_device() -> torch.device:
    """
    Set device for training and testing, choose GPU if available else default to CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{device}device selected')
    return device


def hyperparameter() -> tuple:
    """
    static Hyperparameters setting for training
    """
    # hyperparameters
    epochs = 50
    lr = 0.001
    batch_size = 64
    momentum = 0.9
    return epochs, lr, batch_size, momentum
