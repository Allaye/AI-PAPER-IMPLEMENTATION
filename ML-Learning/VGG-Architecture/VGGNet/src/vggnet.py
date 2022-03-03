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

    def __init__(self, architecture, num_classes=1000):
        super(VGGNet, self).__init__()
        self.vgg_conv_layer = self._make_convo_layers(architecture)
        self.fc_layer = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.vgg_conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    @staticmethod
    def _make_convo_layers(architecture) -> torch.nn.Sequential:
        """
        Create convolutional layers from the vgg architecture type passed in.
        :param architecture:
        """
        layers = []
        in_channels = 3
        for layer in architecture:
            if type(layer) == int:
                out_channels = layer
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1), nn.ReLU()]
                # layers.append([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1) + nn.ReLU()])
                in_channels = layer
            elif (layer == 'Conv1-256'):
                out_channels = 256
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1, stride=1), nn.ReLU()]
            elif (layer == 'LRN'):
                layers += [nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1)]
            elif (layer == 'M'):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vgg = VGGNet(config['vgg16-C1']).to(device)
x = torch.randn(1, 3, 224, 224).to(device)
model = vgg(x).to(device)
print(model.shape)
