import torch
import torch.nn as nn
import torch.nn.functional as F
from VGGNet.src.architecture import config


class VGGNet(nn.Module):
    """
    implementation of the VGG architecture as proposed by Karen Simonyan, Andrew Zisserman,
    see https://paperswithcode.com/paper/very-deep-convolutional-networks-for-large
    this implementation is by no way an efficient implementation of the VGG architecture
    """

    def __init__(self, architecture, in_channels=3, num_classes=1000):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.vgg_conv_layer = self.__make_convo_layers__(architecture)
        nn.Linear(512, num_classes)
        self.fc_layer = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(4096, num_classes))


    def forward(self, x):
        pass

    @staticmethod
    def __make_convo_layers__(architecture) -> torch.nn.Sequential:
        """
        Create convolutional layers from the vgg architecture type passed in.
        :param architecture:
        """
        layers = []
        in_channels = 3
        for layer in architecture:
            if type(layer) == int:
                out_channels = layer
                # layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1), nn.ReLU()]
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1) + nn.ReLU())
                in_channels = layer
            elif (layer == 'M'):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
