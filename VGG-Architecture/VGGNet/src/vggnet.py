import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms


class VGGNet(nn.Module):
    """
    With ConvNets becoming more of a commodity in the computer vision field,
    a number of attempts have been made to improve the original architecture
    of Krizhevsky et al. (2012) in a bid to achieve better accuracy.
    For instance, the best-performing submissions to the ILSVRC-2013 utilised
    smaller receptive window size and smaller stride of the first convolutional
    layer. Another line of improvements dealt with training and testing the
    networks densely over the whole image and over multiple scales.
    In this paper, we address another important aspect of ConvNet architecture
    design – its depth. To this end, we fix other parameters of the architecture,
    and steadily increase the depth of the network by adding more convolutional
    layers, which is feasible due to the use of very small (3 × 3) convolution
    filters in all layers.

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
        # x = x.flatten()
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layer(x)
        return x

    def loss_optimizer(self, lr=0.001, momentum=0.9) -> tuple:
        """
        Create a loss function and an optimizer for the network.
        :param lr:
        :param momentum:
        :return: loss function and optimizer
        """
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        return loss_fn, optimizer

    @staticmethod
    def _make_convo_layers(architecture: list) -> torch.nn.Sequential:
        """
        Create convolutional layers from the vgg architecture type passed in.
        :param architecture:
        """
        layers = []
        in_channels = 3
        for layer in architecture:
            if type(layer) == int:
                out_channels = layer
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                           nn.ReLU()]  # can add a batchnorm2d if you want
                # layers.extend([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1) + nn.ReLU()])
                in_channels = layer
            elif layer == 'Conv1-256':
                out_channels = 256
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1, stride=1), nn.ReLU()]
            elif layer == 'LRN':
                layers += [nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1)]
            elif layer == 'M':
                layers.extend([nn.MaxPool2d(kernel_size=2, stride=2)])
        return nn.Sequential(*layers)

