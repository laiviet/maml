import torch
import torch.nn as nn


class ResiidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResiidualBlock, self).__init__()
        """3*3 convolution with padding,ever time call it the output size become half"""
        self.residual = in_channels == out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


def init_weight(x):
    if isinstance(x, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(x.weight)
    if isinstance(x, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(x.weight)


class CNNModel(torch.nn.Module):

    def __init__(self, n_class):
        super(CNNModel, self).__init__()
        in_size = 3
        layer_size = 64
        self.layer1 = ResiidualBlock(in_size, layer_size)
        self.layer2 = ResiidualBlock(layer_size, layer_size)
        self.layer3 = ResiidualBlock(layer_size, layer_size)
        self.layer4 = ResiidualBlock(layer_size, layer_size)
        self.linear = nn.Linear(layer_size, n_class)

        # self.init_weight()

    def init_weight(self):
        self.apply(init_weight)

    def forward(self, images):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.layer1(images)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
