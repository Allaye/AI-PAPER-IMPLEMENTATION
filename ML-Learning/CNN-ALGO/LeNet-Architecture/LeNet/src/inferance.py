import torch
from lenet import LeNet



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # model = LeNet()
    # model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

m = load_checkpoint("checkpoint.pth")
print(m["model_state_dict"])
print(m["best_accuracy"])
print(m["epoch"])


def make_inference(model):
    pass
