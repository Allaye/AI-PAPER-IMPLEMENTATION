import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def configure_device():
    '''
    check if GPU is available and use it
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    return device


def hyperparameter():
    '''
    define hyper parameters
    '''
    lr = 0.001
    epochs = 10
    batch_size = 100
    return lr, epochs, batch_size


def prepare_dataset(batch_size):
    '''
    get the CIFAR10 dataset, transform it to tensor and normalize it
    '''
    # MNIST dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./',train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes



class LeNet(nn.Module):
    '''
    implementation of the LeNet5 architecture as proposed by Yann Lecun and others,
    see https://en.wikipedia.org/wiki/LeNet-5
    this implementation might just be close as possible to the original implementation
    
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        pass

    def loss_optimizer(self):
        pass