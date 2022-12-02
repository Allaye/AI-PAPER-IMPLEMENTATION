# -*- coding: utf-8 -*-
"""
From scratch implementation of the ResNet architecture.

Developed by Kolade Gideon (Allaye) <allaye.nothing dot com>
*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stripe, padding, group=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stripe, padding, groups=group, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class SqueezeExciteBlock(nn.Module):

    def __init__(self, in_channel, reduced_channel):
        super(SqueezeExciteBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, reduced_channel, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_channel, in_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):

    def __init__(self):
        super(InvertedResidualBlock, self).__init__()
