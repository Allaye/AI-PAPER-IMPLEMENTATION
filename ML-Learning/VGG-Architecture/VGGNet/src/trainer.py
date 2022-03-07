import torch
from vggnet import VGGNet
from dataloader import CustomDataLoader
from setting import *


def train(model, dataloader, hyperparameters, device):
    epochs, lr, batch_size, momentum = hyperparameters
    loss_fn, optimizer = model.loss_optimizer(lr, momentum)

    for epoch in range(epochs):
        for i, (images, label) in enumerate(dataloader):  # image batches
            # move image operation to cuda
            images, label = images.to(device), label.to(device)

            # perform a forward pass
            outputs = model(images)
            # compute the loss values
            loss = loss_fn(outputs, label)
            # clear the gradient values and perform a backward and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training statistics and other information
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader[0])}], Loss: {loss.item():.4f}')


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
