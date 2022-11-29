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
        self.channel_expansion = 4
        # self.architecture = architecture
        self.identity_block = identity
        # self.layers1, self.layers2 = self.__make_layer(architecture)
        # print('layers', self.layers2, self.layers1)
        self.layers = self.__make_layer(architecture)

    def __make_layer(self, architecture) -> List:
        """
        :param architecture: Dict
        :return: List[List[nn.Module]]
        a factory method to create the layers of the residual block
        """
        # for i in range(architecture[0]['iteration']):
        # if the architecture type is type a then we create a 2 block convo self 3.
        spun_up_block = self.__spinup_2_block() if architecture[0] == 2 else self.__spinup_3_block(architecture[1], architecture[2])
        return spun_up_block

        # for conv in architecture:  # [3, 1, 0, 128, 256]
        #     layers.append(
        #         [nn.Conv2d(self.in_channels or conv[3], out_channels=conv[4], kernel_size=conv[0], stride=conv[1],
        #                    padding=conv[2]),
        #          nn.BatchNorm2d(conv[4])])
        #     self.in_channels = conv[4]
        # return layers

    def __spinup_2_block(self):
        return []

    def __spinup_3_block(self, channel, stride):
        conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,
                               padding=0, bias=False)
        bn1 = nn.BatchNorm2d(channel)
        conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        bn2 = nn.BatchNorm2d(channel)
        conv3 = nn.Conv2d(channel, channel * self.channel_expansion, kernel_size=1, stride=1,
                               padding=0, bias=False)
        bn3 = nn.BatchNorm2d(channel * self.channel_expansion)
        relu = nn.ReLU()
        return [conv1, bn1, conv2, bn2, conv3, bn3, relu]




    def forward(self, x: torch.Tensor):
        identity = x.clone()
        deep = len(self.layers)
        print("shape before pass through", identity.shape)
        for layer in self.layers:
            deep -= 1
            print(f"convo: {deep}")
            if deep > 0:
                print("layer", layer)
                x = F.relu(layer[1](layer[0](x)))
                print('shape after pass through', x.shape)
            if deep == 0:
                print("layer", layer)
                x = layer[1](layer[0](x))

                if identity.shape != x.shape:
                    print("identity shape inside pass through", identity.shape)
                    print("x shape inside pass through", x.shape)
                    identity = self.identity_block(identity)
                    print('pass through identity block', identity.shape)
        print('x shape', x.shape)
        print('identity shape', identity.shape)
        return F.relu(x + identity)

    def _require_same_padding(self):
        """
        Perform same padding/zero padding to preserve the image out
        """


# model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU())
# layerss = []
# layerss.append(model)
# print('layers1', layerss)
# layerss.append(model)
# print('layers 2', layerss)
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
# from resnet import ResNet

# model = ResNet(3, q)
# print('model', model)
# a, b = layerss
# print('a', a)
# print('b', b)
# for i in q:
#     print(i.get('iteration'))
# print(type(()))
data = torch.randn(3, 3, 224, 224)

# model = ResidualBlock(q[0]["conv"])
# print(model(data))
