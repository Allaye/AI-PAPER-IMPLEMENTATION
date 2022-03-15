import torch
from torchvision.transforms import transforms

from vggnet import VGGNet
from dataloader import CustomDataset
from setting import *


def training(model, dataloader, hyperparameters, device):
    epochs, lr, batch_size, momentum = hyperparameters
    loss_fn, optimizer = model.loss_optimizer(lr, momentum)
    running_loss = 0.0
    last_loss = 0.0
    for epoch in range(epochs):
        model.train(True)
        print('training')
        for images, label in dataloader:  # image b
            # move image operation to cuda
            images = images.to(device)
            label = label.to(device)

            # perform a forward pass
            outputs = model(images).to(device)
            # compute the loss values
            loss = loss_fn(outputs, label)
            # clear the gradient values and perform a backward and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print training statistics and other information

        # eval_model(model, dataloader[1], device, batch_size)


def eval_model(model, test_loader, device, batch_size, loss_fn=None):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_loss += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total_loss} %')


def save_model(model, epoch):
    pass


from setting import config_device, hyperparameter
from dataloader import CustomDataset
from architecture import config

d_path = "C:\Python\Project\Personal\Python Project\Projects\Data\Algo-ML\dataset"

data = CustomDataset(d_path, transform=[transforms.ToTensor(), transforms.Resize((224, 224))])
trainset, testset = data.getdataloader()

# training(m, trainset, hyper, device)
torch.cuda.empty_cache()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)
vgg = VGGNet(config['vgg16-C1']).to(dev)
# vgg.train(True)
# x = torch.randn(1, 3, 224, 224).to(device)

hyper = hyperparameter()

training(vgg, trainset, hyper, dev)
