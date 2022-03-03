import torch


def config_device():
    """
    Set device for training and testing, choose GPU if available else default to CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{device}device selected')
    return device

