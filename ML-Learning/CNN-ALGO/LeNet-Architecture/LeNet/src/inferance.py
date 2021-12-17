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
    filenames = []
    for file in os.scandir(imagepath):
        if (file.is_file() and file.name.endswith(exten)):
            ## print(file.path)
            filenames.append(file.path)
    batch_size = len(filenames)
    batches = torch.zeros(batch_size, 3, 32, 32, dtype=torch.uint8)
    for i, filename in enumerate(filenames):
        batches[i] = transforms.transforms.Resize((32, 32))(torchvision.io.read_image(filename))
    
    return batches


def make_inference(model, test_loader):
    pass

d = prepare_test_set("./img/")
print(d)