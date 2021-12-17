import torch
from lenet import LeNet



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = LeNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model