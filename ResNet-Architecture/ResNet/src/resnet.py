# -*- coding: utf-8 -*-
"""
From scratch implementation of the ResNet architecture.

Developed by Kolade Gideon (Allaye) <allaye.nothing dot com>
*
"""
from typing import List, Dict

import torch
import torch.nn as nn
from residual_block import ResidualBlock as RB
from architecture import config


class ResNet(nn.Module):
    def __init__(self, image_channels, architecture: List[Dict], num_classes=100):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.architecture = architecture
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, self.layer2, self.layer3, self.layer4 = self._make_resnet_layer(self.architecture)
        self.fc = nn.Linear(self.get_last_channels(architecture), num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    def _make_resnet_layer(self, architecture: List[Dict]):
        """
        Steps:
        1. loop over the architecture and pass the first block to factory class
        2. we will have to create this block some specific amount of time
        3. create the identity block.
        """
        all_layers = []
        layers = []
        for layer in architecture:
            # identity_block = None
            if self.should_make_identity_block(layer):
                print('identity block kicks in>>>>>')
            identity_block = self._make_identity_block(out=layer.get('conv')[-1][-1],)
            for _ in range(layer.get("iteration")):
                layers.append(RB(layer.get("conv"), identity_block))
                # identity_block = None
            all_layers.append(nn.Sequential(*layers))
            layers = []
        return all_layers

    @staticmethod
    def get_last_channels(layer: List[Dict]) -> int:
        """
        Get the last channels of the Architecture been used.
        """
        return layer[-1].get("conv")[-1][-1]

    def should_make_identity_block(self, layer: Dict) -> bool:
        return layer.get('convolutional_block', False)

    def _make_identity_block(self, out, inn=64):
        """
        This is the identity block that is used when the input and output dimensions are not the same.
        How it is achieved : create a 1x1 conv block, then add it to the output of the residual block.
        @param inn:
        @param out:
        @return:
        """
        return nn.Sequential(nn.Conv2d(inn, out, kernel_size=1, stride=1, padding='same', bias=False),
                             nn.BatchNorm2d(out))


data = torch.randn(4, 3, 224, 224)
model = ResNet(3, config.get("res18"))
# print('model', model)
print('nodel size', model(data).size())


# RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[4, 1, 224, 224] to have 3 channels, but got 1 channels instead