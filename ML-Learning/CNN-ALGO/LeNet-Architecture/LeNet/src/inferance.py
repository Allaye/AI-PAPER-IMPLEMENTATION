import os
import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import vision
from lenet import LeNet



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = LeNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model




def make_inference(model, test_loader):
    pass

d = prepare_test_set("./img/")
print(d)