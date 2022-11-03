from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

q = [{
    "conv": [[3, 1, 0, 64], [3, 1, 0, 64]],
    "iteration": 2,
    "convolutional_block": False
},
    {
        "conv": [[3, 1, 0, 128], [3, 1, 0, 128]],
        "iteration": 2,
        "convolutional_block": False
    },
    {
        "conv": [[3, 1, 0, 256], [3, 1, 0, 256]],
        "iteration": 2,
        "convolutional_block": False
    },
    {
        "conv": [[3, 1, 0, 512], [3, 1, 0, 512]],
        "iteration": 2,
        "convolutional_block": False
    }
]


class ResidualBlock(nn.Module):
    def __init__(self, architecture: List, identity=None):
        super(ResidualBlock, self).__init__()
        # self.architecture = architecture
        self.identity_block = identity
        self.layers = self.__make_layer(architecture)

    def __make_layer(self, architecture) -> List[List[nn.Module]]:
        """
        :param architecture: Dict
        :return: List[List[nn.Module]]
        a factory to create the layers of the residual block
        """
        layers = []
        self.in_channels = None
        # for i in range(architecture[0]['iteration']):
        for conv in architecture:
            layers.append(
                [nn.Conv2d(self.in_channels or conv[3], conv[3], kernel_size=conv[0], stride=conv[1], padding=conv[2]),
                 nn.BatchNorm2d(conv[3])])
            self.in_channels = conv[3]
        return layers

    def forward(self, x: torch.Tensor):
        identity = x.clone()
        deep = len(self.layers)
        for layer in self.layers:
            x = F.relu(layer[1](layer[0](x)))
            deep -= 1
            if deep == 0 and self.identity_block is not None:
                identity = self.identity_block(identity)
        return F.relu(x + identity)


# model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU())
# print('model 1', model)
# layer = [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
#          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU()]
# # model1 = nn.Sequential(*layers)
# # model1 = nn.Sequential(*layers)
# print('model 2', layer)
# [[nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU()],
#  [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU()]]
#
# m = ResidualBlock(q)
# # print('model 3', m)
# # print('model 4', m(torch.randn(4, 64, 224, 224)))
#

# con = 'None' or 2
# print('cons', con)

for i in q:
    print(i.get('iteration'))
print(type(()))