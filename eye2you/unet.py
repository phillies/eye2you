import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def u_net128(in_channels=3, out_channels=2, sigmoid=False):
    return Unet128(in_channels=in_channels, out_channels=out_channels, sigmoid=sigmoid)


def u_net256(in_channels=3, out_channels=2, sigmoid=False):
    return Unet256(in_channels=in_channels, out_channels=out_channels, sigmoid=sigmoid)


def u_net(in_channels=3, out_channels=2, depth=2, sigmoid=False):
    return UnetRec(in_channels=in_channels, out_channels=out_channels, depth=depth, sigmoid=sigmoid, top_layer=True)


class UnetRec(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, depth=2, sigmoid=False, top_layer=True):
        super().__init__()

        if top_layer:
            c_inner = 32
        else:
            c_inner = in_channels

        self.conv_in = BasicBlock2d(in_channels=in_channels, out_channels=2 * c_inner)
        self.down = nn.MaxPool2d(kernel_size=2)
        if depth <= 1:
            self.inner = BasicBlock2d(in_channels=2 * c_inner, out_channels=4 * c_inner)
        else:
            self.inner = UnetRec(in_channels=2 * c_inner, out_channels=4 * c_inner, depth=depth - 1, top_layer=False)
        self.up = UpConv2d(in_channels=4 * c_inner, out_channels=2 * c_inner)
        self.conv_out = BasicBlock2d(in_channels=4 * c_inner, out_channels=2 * c_inner)

        if top_layer:
            self.out = nn.Conv2d(
                in_channels=2 * c_inner, out_channels=out_channels, kernel_size=1, padding=0, bias=False)
            if sigmoid:
                self.final = nn.Sigmoid()
            else:
                self.final = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.size())
        x = self.conv_in(x)
        print(x.size())
        x_res = x.clone()
        print(x.size())
        x = self.down(x)
        print(x.size())
        x = self.inner(x)
        print(x.size())
        x = self.up(x)
        print(x.size())
        x = torch.cat((x_res, x), dim=1)
        print(x.size())
        x = self.conv_out(x)
        print(x.size())
        return x


class UnetFlex(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, depth=2, sigmoid=False):
        super().__init__()

        self.sequence = ['conv', 'copy', 'down'] * depth + ['conv'] + ['up', 'merge', 'conv'] * depth + ['out']

        self.net = _assemble_unet(self.sequence, in_channels, out_channels)

        if sigmoid:
            self.final = nn.Sigmoid()
        else:
            self.final = nn.Softmax(dim=1)

    def forward(self, x):
        residual_stack = []

        for ii, ops in enumerate(self.sequence[1:]):
            if ops == 'conv_double':
                x = ops(x)
            elif ops == 'copy':
                residual_stack.append(x.clone())
            elif ops == 'down':
                x = ops(x)
            elif ops == 'up':
                x = ops(x)
            elif ops == 'merge':
                x_res = residual_stack.pop()
                x = torch.cat((x, x_res), dim=1)
            elif ops == 'conv_half':
                x = ops(x)
            elif ops == 'out':
                x = ops(x)
            print(x.size())

        x = self.final(x)
        return x


class Unet128(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, depth=2, sigmoid=False):
        super().__init__()

        self.block1 = BasicBlock2d(in_channels, 64)

        self.max1 = nn.MaxPool2d(kernel_size=2)

        self.block2 = BasicBlock2d(64, 128)

        self.upconv2 = UpConv2d(in_channels=128, out_channels=64)

        self.block5 = BasicBlock2d(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0, bias=False)

        if sigmoid:
            self.final = nn.Sigmoid()
        else:
            self.final = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x_res1 = x.clone()

        x = self.max1(x)

        x = self.block2(x)

        x = self.upconv2(x)

        x = torch.cat((x, x_res1), dim=1)
        x = self.block5(x)

        x = self.out(x)
        x = self.final(x)
        return x


class Unet256(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, depth=2, sigmoid=False):
        super().__init__()

        self.block1 = BasicBlock2d(in_channels, 64)

        self.max1 = nn.MaxPool2d(kernel_size=2)

        self.block2 = BasicBlock2d(64, 128)
        self.max2 = nn.MaxPool2d(kernel_size=2)

        self.block3 = BasicBlock2d(128, 256)

        self.upconv1 = UpConv2d(in_channels=256, out_channels=128)

        self.block4 = BasicBlock2d(256, 128)

        self.upconv2 = UpConv2d(in_channels=128, out_channels=64)

        self.block5 = BasicBlock2d(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0, bias=False)

        if sigmoid:
            self.final = nn.Sigmoid()
        else:
            self.final = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x_res1 = x.clone()

        x = self.max1(x)

        x = self.block2(x)
        x_res2 = x.clone()

        x = self.max2(x)

        x = self.block3(x)

        x = self.upconv1(x)

        x = torch.cat((x, x_res2), dim=1)
        x = self.block4(x)

        x = self.upconv2(x)

        x = torch.cat((x, x_res1), dim=1)
        x = self.block5(x)

        x = self.out(x)
        x = self.final(x)
        return x


class BasicBlock2d(nn.Module):
    '''Basic 2D convolution block with 2 consecutive 2D convolutions with 
    batch normalization and reLU activation after each convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UpConv2d(nn.Module):
    """Upconvolution - 2D transpose convolution (sometimes called deconvolution) to upsample the image. 
    Default parameters create a 2x2 transposed convolution with stride=2 to upsample by factor 2.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, padding=0, stride=2, bias=False, batch_norm=False):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        #x = self.up(x)
        x = self.conv(x)
        if not self.bn is None:
            x = self.bn(x)
        return x