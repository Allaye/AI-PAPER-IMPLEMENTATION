import torch
from lenet import LeNet



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = LeNet()
    