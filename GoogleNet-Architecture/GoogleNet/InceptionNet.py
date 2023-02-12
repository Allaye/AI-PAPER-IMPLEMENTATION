import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionNet(nn.Module):
    def __init__(self, in_channels, in_1x1, in_3x3reduce, in_3x3, in_5x5reduce, in_5x5, in_1x1pool):
        super(InceptionNet, self).__init__()
        self.incep_1 = ConvBlock(in_channels, in_1x1, kernel_size=1, padding='same')
        self.incep_3 = nn.Sequential(
            ConvBlock(in_channels, in_3x3reduce, kernel_size=1, padding='same'),
            ConvBlock(in_3x3reduce, in_3x3, kernel_size=3, padding='same'))
        self.incep_5 = nn.Sequential(
            ConvBlock(in_channels, in_5x5reduce, kernel_size=1, padding='same'),
            ConvBlock(in_5x5reduce, in_5x5, kernel_size=5, padding='same'))
        self.incep_1pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding='same'),
            ConvBlock(in_channels, in_1x1pool, kernel_size=1, padding='same'))

    def forward(self, x):
        return torch.cat([
            self.incep_1(x),
            self.incep_3(x),
            self.incep_5(x),
            self.incep_1pool(x)
        ], 1)


