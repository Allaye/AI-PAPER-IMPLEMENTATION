import torch
import numpy as np
from PIL import Image
from lenet import LeNet
from dataset_loader import prepare_testset


def load_checkpoint(filepath):
    torch.no_grad()
    checkpoint = torch.load(filepath)
    model = LeNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


m = load_checkpoint("models/checkpoint18.pth")
# print(m["model_state_dict"])
# print(m["best_accuracy"])
# print(m["epoch"])

data = prepare_testset("img")
torch.no_grad()
im = Image.open("img/cat.jpeg").convert("RGB")
# im.show()
print(data.shape)
print(data[0::].shape)
outputs = m(data)
print(outputs.shape)
_, predicted = torch.max(outputs.data, 1)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print(classes[predicted.item()])
print(predicted)


# print(predicted.data)
# print(outputs.data == outputs)
# for j in range(data.__len__()):
# print("Predicted: ", classes[predicted[j]])
# print(predicted, classes[ predicted[j]])
# pred = m(data[1::])
# predicted_class = np.argmax(pred.detach().numpy())


def make_inference(model, test_set):
    data = prepare_testset(test_set)
    model = load_checkpoint("./models/checkpoint18.pth")

