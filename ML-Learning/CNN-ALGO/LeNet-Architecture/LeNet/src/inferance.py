import torch
from torchvision import transforms, datasets
from lenet import LeNet



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = LeNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def prepare_test_set(imagepath):
    dataset = datasets.ImageFolder(root=imagepath, transform=transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((32, 32))]))
    return dataset
def make_inference(model, test_loader):
    pass