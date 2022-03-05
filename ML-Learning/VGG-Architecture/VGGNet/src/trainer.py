import torch
from vggnet import VGGNet
from dataloader import CustomDataLoader
from setting import *


def train(model, dataloader, hyperparameters, device):
    epochs, lr, batch_size, momentum = hyperparameters
    loss_fn, optimizer = model.loss_optimizer(lr, momentum)

    for epoch in range(epochs):
        for i, (images, label) in enumerate(dataloader):
            # move image operation to cuda
            images = images.to(device)
            label = label.to(device)

            # perform a forward pass
            outputs = model(images)
            # compute the loss values
            loss = loss_fn(outputs, label)
            # clear the gradient values and perform a backward and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def eval_model(model, test_loader):
    pass


def save_model(model, epoch):
    pass


from setting import config_device, hyperparameter

hyper = hyperparameter()
print(hyper[0])
print(VGGNet("vgg16"))
m = VGGNet("vgg16")
print(m)
