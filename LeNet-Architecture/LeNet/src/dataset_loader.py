import os
import torch
import torchvision
import torchvision.transforms as transforms


def prepare_dataset(batch_size) -> tuple:
    """
    get the CIFAR10 dataset, transform it to tensor, convert it into grayscale and normalize it
    return a trainloader, testloader and data classes
    :param batch_size: batch size that the dataset will be loaded in
    :rtype: tuple: contains the trainloader, testloader and data classes
    """
    # CIFAR10 dataset

    train_dataset = torchvision.datasets.CIFAR10(root='./', train=True,
                                                 transform=transforms.Compose(
                                                     [transforms.ToTensor(), transforms.Grayscale()]),
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Grayscale()]))

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


def prepare_testset(imagepath, ext=(".jpg", ".png", ".jpeg")) -> tuple:
    """
    prepare dataset to use for normal testing, this function accepts a path to the file and the extension of the file
    and return a tensor of the image in batch form and classes
    :param imagepath: path to the image
    :param ext: extension of the image
    :rtype: tuple: contains the tensor of the images in batches and their respective classes
    """
    # get the image
    filenames = []
    for file in os.scandir(imagepath):

        if file.is_file() and file.name.endswith(ext):
            filenames.append(file.path)
    batch_size = len(filenames)
    batches = torch.zeros(batch_size, 1, 32, 32, dtype=torch.float)
    for i, filename in enumerate(filenames):
        batches[i] = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale()])(
            torchvision.io.read_image(filename))
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return batches, classes
