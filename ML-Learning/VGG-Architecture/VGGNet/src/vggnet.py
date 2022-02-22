import torch
import torch.nn as nn
import torch.nn.functional as F
from VGGNet.src.architecture import config


class VGGNet(nn.Module):

    def __init__(self, architecture, num_classes=1000):
        super(VGGNet, self).__init__()

    def forward(self, x):
        pass

    def __make_convo_layers__(self, architecture):
        """
        Create convolutional layers from the vgg architecture passed in.
        :param self - the object itself
        :param architecture:
        """
        layers = []
        in_channels = 3
