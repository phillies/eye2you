import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.conv1 = Basic2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = Basic2d(in_channels=32, out_channels=64, kernel_size=7, padding=3)

        self.res_conv1 = Basic2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, relu=False)

        self.sep1 = Basic2d(in_channels=64, out_channels=64, kernel_size=15, padding=7, groups=64)
        self.sep2 = Basic2d(in_channels=64, out_channels=64, kernel_size=15, padding=7, groups=64, relu=False)

        self.res_conv2 = Basic2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, relu=False)

        self.sep3 = Basic2d(in_channels=64, out_channels=64, kernel_size=15, padding=7, groups=64)
        self.sep4 = Basic2d(in_channels=64, out_channels=64, kernel_size=15, padding=7, groups=64, relu=False)

        self.res_conv3 = Basic2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, relu=False)

        self.sep5 = Basic2d(in_channels=64, out_channels=64, kernel_size=15, padding=7, groups=64)
        self.sep6 = Basic2d(in_channels=64, out_channels=64, kernel_size=15, padding=7, groups=64, relu=False)

        self.res_conv4 = Basic2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, relu=False)

        self.sep7 = Basic2d(in_channels=64, out_channels=64, kernel_size=15, padding=7, groups=64)
        self.sep8 = Basic2d(in_channels=64, out_channels=64, kernel_size=15, padding=7, groups=64, relu=False)

        self.sep9 = Basic2d(in_channels=64, out_channels=16, kernel_size=7, padding=3, groups=16)
        self.sep10 = Basic2d(in_channels=16, out_channels=out_channels, kernel_size=5, padding=2, groups=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x_res = self.res_conv1(x)

        x = self.sep1(x)
        x = self.sep2(x)

        x = x + x_res
        x1 = F.relu(x, inplace=True)
        x_res = self.res_conv2(x1)

        x = self.sep3(x1)
        x = self.sep4(x)

        x = x + x_res + x1
        x2 = F.relu(x, inplace=True)
        x_res = self.res_conv3(x2)

        x = self.sep5(x2)
        x = self.sep6(x)

        x = x + x_res + x1 + x2
        x3 = F.relu(x, inplace=True)
        x_res = self.res_conv4(x3)

        x = self.sep7(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.sep8(x)

        x = x + x_res + x1 + x2 + x3
        x = F.relu(x, inplace=True)

        x = self.sep9(x)
        x = self.sep10(x)

        return x


class Basic2d(nn.Module):
    '''Basic 2D convolution block with 2 consecutive 2D convolutions with
    batch normalization and reLU activation after each convolution
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 bias=False,
                 groups=1,
                 momentum=0.1,
                 relu=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialSeparable2d(nn.Module):
    '''Basic 2D convolution block with 2 consecutive 2D convolutions with
    batch normalization and reLU activation after each convolution
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 bias=False,
                 momentum=0.1,
                 relu=True):
        super().__init__()
        self.convH = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=stride,
            bias=bias)
        self.convW = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            stride=stride,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.convH(x)
        x = self.convW(x)
        x = self.bn(x)
        if x.relu is not None:
            x = self.relu(x)
        return x


class DepthSeparable2d(nn.Module):
    '''Basic 2D convolution block with 2 consecutive 2D convolutions with
    batch normalization and reLU activation after each convolution
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 bias=False,
                 momentum=0.1,
                 relu=True):
        super().__init__()
        self.conv_depth = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=bias,)
        self.conv_point = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv_depth(x)
        x = self.conv_pointwise(x)
        x = self.bn(x)
        if x.relu is not None:
            x = self.relu(x)
        return x