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


class ResNet(nn.Module):
    def __init__(self, image_channels, architecture: List[Dict], num_classes=100):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, self.layer2, self.layer3, self.layer4 = self._make_resnet_layer(architecture)
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
        all_layers = ()
        for layer in architecture:
            identity_block = None
            if self.should_make_identity_block(layer):
                identity_block = self.make_identity_block(layer)
            layers = []
            for _ in range(layer.get("iteration")):
                layers.append(RB(layer.get("conv"), identity_block))
                identity_block = None
            all_layers += tuple(nn.Sequential(*layers))
        return all_layers

    @staticmethod
    def get_last_channels(architecture: List[Dict]) -> int:
        """
        Get the last channels of the Architecture been used.
        """
        return architecture[-1].get("conv")[-1][-1]

    @property
    def should_make_identity_block(self):
        return self.architecture.get('convolutional_block', False)

    def _make_identity_block(self):
        """
        This is the identity block that is used when the input and output dimensions are not the same.
        How it is achieved : create a 1x1 conv block, then add it to the output of the residual block.
        """
        return nn.Sequential(nn.Conv2d(self.in_channels, self.next_channels, kernel_size=1, stride=1, padding=0),
                             nn.BatchNorm2d(self.next_channels))
