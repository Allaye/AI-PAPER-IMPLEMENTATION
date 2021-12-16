import torch
import torchvision
import torchvision.transforms as transforms


def prepare_dataset(batch_size):
    '''
    get the CIFAR10 dataset, transform it to tensor and normalize it
    '''
    # MNIST dataset

    train_dataset = torchvision.datasets.CIFAR10(root='./',train=True,
                                               transform=transforms.Compose([transforms.ToTensor(), transforms.Grayscale()]),
                                               download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Grayscale()]))
    
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes
