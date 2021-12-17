import os
import torch
import torchvision
import torchvision.transforms as transforms


def prepare_dataset(batch_size):
    '''
    get the CIFAR10 dataset, transform it to tensor, convert it into grayscale and normalize it
    return a trainloader, testloader and data classes
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


def prepare_testset(imagepath, exten=(".jpg", ".png", ".jpeg")):
    """
    prepare dataset to use for normal testing, this function accepts a path to the file and the extension of the file
    and return a tensor of the images or image
    
    """
    # dataset = datasets.ImageFolder(root=imagepath
    filenames = []
    for file in os.scandir(imagepath):

        if (file.is_file() and file.name.endswith(exten)):
            filenames.append(file.path)
    batch_size = len(filenames)
    batches = torch.zeros(batch_size, 3, 32, 32, dtype=torch.uint8)
    for i, filename in enumerate(filenames):
        batches[i] = transforms.transforms.Resize((32, 32))(torchvision.io.read_image(filename))
    return batches
