import torch
import torch.nn as nn
from InceptionNet import InceptionBlock, ConvBlock


class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(in_channels=192, in_1x1=64, in_3x3reduce=96, in_3x3=128, in_5x5reduce=16,
                                          in_5x5=32, in_1x1pool=32)
        self.inception3b = InceptionBlock(in_channels=256, in_1x1=128, in_3x3reduce=128, in_3x3=192, in_5x5reduce=32,
                                          in_5x5=96, in_1x1pool=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionBlock(in_channels=480, in_1x1=192, in_3x3reduce=96, in_3x3=208, in_5x5reduce=16,
                                          in_5x5=48, in_1x1pool=64)
        self.inception4b = InceptionBlock(in_channels=512, in_1x1=160, in_3x3reduce=112, in_3x3=224, in_5x5reduce=24,
                                          in_5x5=64, in_1x1pool=64)
        self.inception4c = InceptionBlock(in_channels=512, in_1x1=128, in_3x3reduce=128, in_3x3=256, in_5x5reduce=24,
                                          in_5x5=64, in_1x1pool=64)
        self.inception4d = InceptionBlock(in_channels=512, in_1x1=112, in_3x3reduce=144, in_3x3=288, in_5x5reduce=32,
                                          in_5x5=64, in_1x1pool=64)
        self.inception4e = InceptionBlock(in_channels=528, in_1x1=256, in_3x3reduce=160, in_3x3=320, in_5x5reduce=32,
                                          in_5x5=128, in_1x1pool=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = InceptionBlock(in_channels=832, in_1x1=256, in_3x3reduce=160, in_3x3=320, in_5x5reduce=32,
                                          in_5x5=128, in_1x1pool=128)
        self.inception5b = InceptionBlock(in_channels=832, in_1x1=384, in_3x3reduce=192, in_3x3=384, in_5x5reduce=48,
                                          in_5x5=128, in_1x1pool=128)
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(in_features=1024, out_features=1000)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.inception3b(self.inception3a(x)))
        
