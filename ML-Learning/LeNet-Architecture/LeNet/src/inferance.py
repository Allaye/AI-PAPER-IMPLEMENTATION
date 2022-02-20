import torch
import numpy as np
from lenet import LeNet
from dataset_loader import prepare_testset


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = LeNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


m = load_checkpoint("checkpoint.pth")
# print(m["model_state_dict"])
# print(m["best_accuracy"])
# print(m["epoch"])

data = prepare_testset("img")
pred = m(data[2::])
predicted_class = np.argmax(pred.detach().numpy())
print(predicted_class)


def make_inference(model):
    pass
