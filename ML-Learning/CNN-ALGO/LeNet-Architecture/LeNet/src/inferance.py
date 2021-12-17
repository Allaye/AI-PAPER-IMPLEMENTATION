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

def prepare_test_set(imagepath, exten=(".jpg", ".png", ".jpeg")):
    # dataset = datasets.ImageFolder(root=imagepath, transform=transforms.Compose([transforms.ToTensor()]))
    filename = []
    for file in os.scandir(imagepath):
        if (file.is_file() and file.name.endswith(exten)):
            ## print(file.path)
            filename.append(file.path)
    batch_size = len(filename)
    # dataset = torchvision.io.read_image(imagepath)
    return None


def make_inference(model, test_loader):
    pass

d = prepare_test_set("./img/")
print(d)