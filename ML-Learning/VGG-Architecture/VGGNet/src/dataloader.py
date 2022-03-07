import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CustomDataset(Dataset):
    """
    A class to create a custom dataset, which is a subclass of torch.utils.data.Dataset
    and a data loader, which is a subclass of torch.utils.data.DataLoader
    """

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transforms.Compose(transform)
        self.classes = sorted(os.listdir(data_path))
        self.allimagepaths = []
        self.targets = []

        for target, classname in enumerate(self.classes):
            for img in os.listdir(os.path.join(data_path, classname)):
                self.allimagepaths.append(os.path.join(data_path, classname, img))
                self.targets.append(target)

    def __getitem__(self, index):
        """
        override the __getitem__ method to return the data sample
        """
        imagepath = self.allimagepaths[index]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[index]
        return (image, target)

    def __len__(self):
        """
        override the __len__ method to return the number of data samples
        """
        return len(self.allimagepaths)

    def getdataloader(self, batch_size=5, split_size=(0.8, 0.2), shuffle=True, num_workers=0):
        """
        create a customizable data loader, with the able of been iterable and reshuffle
        :param split_size:
        :param batch_size:
        :param shuffle:
        :param num_workers:
        :return:
        """

        train_set, test_set = torch.utils.data.random_split(self, [round(0.8 * len(self)), round(0.2 * self.__len__())])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader, test_loader


d_path = "C:\Python\Project\Personal\Python Project\Projects\Data\Algo-ML\dataset"
print(os.listdir("C:\Python\Project\Personal\Python Project\Projects\Data\Algo-ML\dataset"))
print(os.listdir(os.path.join(d_path, 'Orangutan')))
imgpath = os.listdir(os.path.join(d_path, 'Orangutan'))
print(os.path.join(d_path, 'Orangutan', '1.jpg'))

data = CustomDataset(d_path, transform=[transforms.ToTensor(), transforms.Resize((224, 224))])
print(data.classes)
# print(data.allimagepaths)
print(data.targets)
# print(data.__getitem__(0))
print(data.__len__())
trainset, testset = data.getdataloader()

print(trainset, testset)
for i, (img, target) in enumerate(trainset):
    print(i, img.shape, target)
    if i != 1:
        break
cor = (0.8, 0.2)
print([*cor])
