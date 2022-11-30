# -*- coding: utf-8 -*-
"""
From scratch implementation of the ResNet architecture.

Developed by Kolade Gideon (Allaye) <allaye.nothing dot com>
*
"""
from typing import List, Dict

import torch
import torch.nn as nn
from residual_block import ResidualBlock as Rb
from architecture import config


class ResNet(nn.Module):
    def __init__(self, image_channels, loop: List[int], architecture: int, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.architecture = architecture
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_resnet_layer(64, self.architecture, 1, loop[0])
        self.layer2 = self._make_resnet_layer(128, self.architecture, 2, loop[1])
        self.layer3 = self._make_resnet_layer(256, self.architecture, 2, loop[2])
        self.layer4 = self._make_resnet_layer(512, self.architecture, 2, loop[3])
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    def _make_resnet_layer(self, channel: int, architecture: int, stride: int, loop: int):
        """
        Steps:
        1. loop over the architecture and pass the first block to factory class
        2. we will have to create this block some specific amount of time
        3. create the identity block.
        """
        # architecture [3, 64, 1]
        identity_block = None
        layers = []
        if stride != 1 or self.in_channel != channel * 4:
            identity_block = self.__make_identity_block(self.in_channel, channel, stride)
        layers.append(Rb(self.in_channel, channel, architecture, stride, identity_block))
        self.in_channel = channel * 4
        for _ in range(loop-1):
            layers.append(Rb(self.in_channel, channel, architecture))

        return nn.Sequential(*layers)

    @staticmethod
    def __make_identity_block(in_channel: int, out_channel: int, stride: int) -> nn.Sequential:
        """
        This is the identity block that is used when the input and output dimensions are not the same.
        How it is achieved : create a 1x1 conv block, then add it to the output of the residual block.
        @param in_channel
        @param out_channel:
        @return:
        """
        return nn.Sequential(nn.Conv2d(in_channel, out_channel * 4, kernel_size=1, stride=stride, bias=False),
                             nn.BatchNorm2d(out_channel*4))


data = torch.randn(1, 3, 224, 224)
model = ResNet(3, [3, 4, 6, 3], 3, 1000)
# print('model', model)
print('model size', model(data).size())

# RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[4, 1, 224, 224] to have 3 channels,
# but got 1 channels instead
