import torch
import torch.nn as nn
from inceptionnet import InceptionBlock, ConvBlock


class GoogleNet(nn.Module):
    """
    implementation of the googlenet/inceptionnet architecture
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(in_channels=192, out_1x1=64, out_3x3reduce=96, out_3x3=128, out_5x5reduce=16,
                                          out_5x5=32, out_1x1pool=32)
        self.inception3b = InceptionBlock(in_channels=256, out_1x1=128, out_3x3reduce=128, out_3x3=192, out_5x5reduce=32,
                                          out_5x5=96, out_1x1pool=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionBlock(in_channels=480, out_1x1=192, out_3x3reduce=96, out_3x3=208, out_5x5reduce=16,
                                          out_5x5=48, out_1x1pool=64)
        self.inception4b = InceptionBlock(in_channels=512, out_1x1=160, out_3x3reduce=112, out_3x3=224, out_5x5reduce=24,
                                          out_5x5=64, out_1x1pool=64)
        self.inception4c = InceptionBlock(in_channels=512, out_1x1=128, out_3x3reduce=128, out_3x3=256, out_5x5reduce=24,
                                          out_5x5=64, out_1x1pool=64)
        self.inception4d = InceptionBlock(in_channels=512, out_1x1=112, out_3x3reduce=144, out_3x3=288, out_5x5reduce=32,
                                          out_5x5=64, out_1x1pool=64)
        self.inception4e = InceptionBlock(in_channels=528, out_1x1=256, out_3x3reduce=160, out_3x3=320, out_5x5reduce=32,
                                          out_5x5=128, out_1x1pool=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = InceptionBlock(in_channels=832, out_1x1=256, out_3x3reduce=160, out_3x3=320, out_5x5reduce=32,
                                          out_5x5=128, out_1x1pool=128)
        self.inception5b = InceptionBlock(in_channels=832, out_1x1=384, out_3x3reduce=192, out_3x3=384, out_5x5reduce=48,
                                          out_5x5=128, out_1x1pool=128)
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(in_features=1024, out_features=1000)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.inception3b(self.inception3a(x)))
        x = self.maxpool4(self.inception4e(self.inception4d(self.inception4c(self.inception4b(self.inception4a(x))))))
        x = self.avgpool1(self.inception5b(self.inception5a(x)))
        x = self.dropout(x.reshape(x.shape[0], -1)) # flatten the tensor and push it into the dropout
        return self.fc1(x)

    def loss_optimizer(self, lr=0.001) -> tuple:
        """
        define the loss and optimizer
        :param lr: learning rate
        :rtype: tuple of torch.nn.CrossEntropyLoss and torch.optim.SGD
        """
        # define the loss function
        loss_fn = nn.CrossEntropyLoss()
        # define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr)
        return loss_fn, optimizer


data = torch.randn(3, 3, 224, 224)
model = GoogleNet()
print(model(data).shape)