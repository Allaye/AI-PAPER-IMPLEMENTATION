from typing import List, Dict, Tuple, Union, Any
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
    def __init__(self, inn_channel: int, channel: int, architecture: int, stride: int = 1, identity=None):
        super(ResidualBlock, self).__init__()
        # this is the final output channel for each convo block,
        # which is the normal input_channel * 4
        self.inn_channel = inn_channel
        self.stride = stride
        self.channel_expansion = 4
        # identity operation on the conv block if required
        self.identity_block = identity
        # get the conv block and other params
        self.conv, self.bn = self.__make_layer(self.inn_channel, channel, architecture, stride)
        # relu value
        self.relu = nn.ReLU()

    def __make_layer(self, in_channel: int, channel: int, architecture: int, stride: int) -> Tuple[List[F.conv2d], List[F.batch_norm]]:
        """
        :param architecture: Dict
        :return: List[List[nn.Module]]
        a factory method to create the layers of the residual block.
        """
        # for i in range(architecture[0]['iteration']):
        # if the architecture type is type a then we create a 2 block convo self 3.
        spun_up_block = self.__spinup_2_block() if architecture == 2 else self.__spinup_3_block(in_channel, channel,
                                                                                                stride)
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

    def __spinup_3_block(self, in_channel, channel, stride) -> Tuple[List[F.conv2d], List[F.batch_norm]]:
        conv1 = nn.Conv2d(in_channels=in_channel, out_channels=channel, kernel_size=1, stride=1,
                          padding=0, bias=False)
        bn1 = nn.BatchNorm2d(channel)
        conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=stride,
                          padding=1, bias=False)
        bn2 = nn.BatchNorm2d(channel)
        conv3 = nn.Conv2d(channel, channel * self.channel_expansion, kernel_size=1, stride=1,
                          padding=0, bias=False)
        bn3 = nn.BatchNorm2d(channel * self.channel_expansion)
        return [conv1, conv2, conv3, ], [bn1, bn2, bn3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        length = len(self.conv)
        for item in zip(self.conv, self.bn):
            length -= 1
            if length != 0:
                x = self.relu(item[1](item[0](x)))
            else:
                x = item[1](item[0](x))
        if self.identity_block is not None:
            identity = self.identity_block(identity)
        return self.relu(x + identity)
