import torch
import torch.nn as nn


def u_net(in_channels=3, out_channels=2, depth=2, final_layer='sigmoid'):
    UnetClass = None
    if depth == 1:
        UnetClass = Unet1
    elif depth == 2:
        UnetClass = Unet2
    elif depth == 3:
        UnetClass = Unet3
    else:
        UnetClass = Unet4
    net = UnetClass(in_channels=in_channels, out_channels=out_channels, final_layer=final_layer)
    return net


def u_net_rec(in_channels=3, out_channels=2, depth=2, bias=False, final_layer='sigmoid'):
    return Unet(in_channels=in_channels,
                out_channels=out_channels,
                depth=depth,
                bias=bias,
                final_layer=final_layer,
                upconv_batch=False,
                top_layer=True)


class Unet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 depth=2,
                 bias=False,
                 final_layer='sigmoid',
                 upconv_batch=False,
                 top_layer=True):
        super().__init__()

        self.top_layer = top_layer
        if top_layer:
            c_inner = 32
        else:
            c_inner = in_channels

        self.conv_in = BasicBlock2d(in_channels=in_channels, out_channels=2 * c_inner, bias=bias)
        self.down = nn.MaxPool2d(kernel_size=2)
        if depth <= 1:
            self.inner = BasicBlock2d(in_channels=2 * c_inner, out_channels=4 * c_inner, bias=bias)
        else:
            self.inner = Unet(in_channels=2 * c_inner,
                              out_channels=4 * c_inner,
                              depth=depth - 1,
                              top_layer=False,
                              bias=bias,
                              upconv_batch=upconv_batch)
        self.up = UpConv2d(in_channels=4 * c_inner, out_channels=2 * c_inner, batch_norm=upconv_batch, bias=bias)
        self.conv_out = BasicBlock2d(in_channels=4 * c_inner, out_channels=2 * c_inner, bias=bias)

        if top_layer:
            self.out = nn.Conv2d(in_channels=2 * c_inner,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 padding=0,
                                 bias=bias)
            if final_layer == 'sigmoid':
                self.final = nn.Sigmoid()
            elif final_layer == 'softmax':
                self.final = nn.Softmax(dim=1)
            else:
                self.final = lambda x: x

    def forward(self, x, *args):
        #print(x.size())
        x = self.conv_in(x)
        #print(x.size())
        x_res = x.clone()
        #print(x.size())
        x = self.down(x)
        #print(x.size())
        x = self.inner(x)
        #print(x.size())
        x = self.up(x)
        #print(x.size())
        h_pad = x_res.shape[2] - x.shape[2]
        w_pad = x_res.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        #print(x.shape, x_res.shape)
        x = torch.cat((x_res, x), dim=1)
        del x_res
        #print(x.size())
        x = self.conv_out(x)
        #print(x.size())
        if self.top_layer:
            x = self.out(x)

            x = self.final(x)
        return x


class Unet1(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, final_layer=False):
        super().__init__()

        self.block1 = BasicBlock2d(in_channels, 64)

        self.max1 = nn.MaxPool2d(kernel_size=2)

        self.block2 = BasicBlock2d(64, 128)

        self.upconv2 = UpConv2d(in_channels=128, out_channels=64)

        self.block5 = BasicBlock2d(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0, bias=False)

        if final_layer == 'sigmoid':
            self.final = nn.Sigmoid()
        elif final_layer == 'softmax':
            self.final = nn.Softmax(dim=1)
        else:
            self.final = lambda x: x

    def forward(self, x, *args):
        x = self.block1(x)
        x_res1 = x.clone()

        x = self.max1(x)

        x = self.block2(x)

        x = self.upconv2(x)

        h_pad = x_res1.shape[2] - x.shape[2]
        w_pad = x_res1.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res1), dim=1)
        x = self.block5(x)

        x = self.out(x)
        x = self.final(x)
        del x_res1
        return x


class Unet2(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, final_layer=False):
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

        if final_layer == 'sigmoid':
            self.final = nn.Sigmoid()
        elif final_layer == 'softmax':
            self.final = nn.Softmax(dim=1)
        else:
            self.final = lambda x: x

    def forward(self, x, *args):
        x = self.block1(x)
        x_res1 = x.clone()
        x = self.max1(x)

        x = self.block2(x)
        x_res2 = x.clone()
        x = self.max2(x)

        x = self.block3(x)

        x = self.upconv1(x)
        h_pad = x_res2.shape[2] - x.shape[2]
        w_pad = x_res2.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res2), dim=1)
        x = self.block4(x)

        x = self.upconv2(x)
        h_pad = x_res1.shape[2] - x.shape[2]
        w_pad = x_res1.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res1), dim=1)
        x = self.block5(x)

        x = self.out(x)
        x = self.final(x)
        del x_res1, x_res2
        return x


class Unet3(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, final_layer=False, bias=False):
        super().__init__()

        self.inblock1 = BasicBlock2d(in_channels, 64)
        self.max1 = nn.MaxPool2d(kernel_size=2)

        self.inblock2 = BasicBlock2d(64, 128)
        self.max2 = nn.MaxPool2d(kernel_size=2)

        self.inblock3 = BasicBlock2d(128, 256)
        self.max3 = nn.MaxPool2d(kernel_size=2)

        self.inner = BasicBlock2d(256, 512)

        self.upconv3 = UpConv2d(in_channels=512, out_channels=256)
        self.outblock3 = BasicBlock2d(512, 256)

        self.upconv2 = UpConv2d(in_channels=256, out_channels=128)
        self.outblock2 = BasicBlock2d(256, 128)

        self.upconv1 = UpConv2d(in_channels=128, out_channels=64)
        self.outblock1 = BasicBlock2d(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0, bias=False)

        if final_layer == 'sigmoid':
            self.final = nn.Sigmoid()
        elif final_layer == 'softmax':
            self.final = nn.Softmax(dim=1)
        else:
            self.final = lambda x: x

    def forward(self, x, *args):
        x = self.inblock1(x)
        x_res1 = x.clone()
        x = self.max1(x)

        x = self.inblock2(x)
        x_res2 = x.clone()
        x = self.max2(x)

        x = self.inblock3(x)
        x_res3 = x.clone()
        x = self.max3(x)

        x = self.inner(x)

        x = self.upconv3(x)
        h_pad = x_res3.shape[2] - x.shape[2]
        w_pad = x_res3.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res3), dim=1)
        x = self.outblock3(x)

        x = self.upconv2(x)
        h_pad = x_res2.shape[2] - x.shape[2]
        w_pad = x_res2.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res2), dim=1)
        x = self.outblock2(x)

        x = self.upconv1(x)
        h_pad = x_res1.shape[2] - x.shape[2]
        w_pad = x_res1.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res1), dim=1)
        x = self.outblock1(x)

        x = self.out(x)
        x = self.final(x)
        del x_res1, x_res2, x_res3
        return x


class Unet4(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, final_layer=False):
        super().__init__()

        self.inblock1 = BasicBlock2d(in_channels, 64)
        self.max1 = nn.MaxPool2d(kernel_size=2)

        self.inblock2 = BasicBlock2d(64, 128)
        self.max2 = nn.MaxPool2d(kernel_size=2)

        self.inblock3 = BasicBlock2d(128, 256)
        self.max3 = nn.MaxPool2d(kernel_size=2)

        self.inblock4 = BasicBlock2d(256, 512)
        self.max4 = nn.MaxPool2d(kernel_size=2)

        self.inner = BasicBlock2d(512, 1024)

        self.upconv4 = UpConv2d(in_channels=1024, out_channels=512)
        self.outblock4 = BasicBlock2d(1024, 512)

        self.upconv3 = UpConv2d(in_channels=512, out_channels=256)
        self.outblock3 = BasicBlock2d(512, 256)

        self.upconv2 = UpConv2d(in_channels=256, out_channels=128)
        self.outblock2 = BasicBlock2d(256, 128)

        self.upconv1 = UpConv2d(in_channels=128, out_channels=64)
        self.outblock1 = BasicBlock2d(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0, bias=False)

        if final_layer == 'sigmoid':
            self.final = nn.Sigmoid()
        elif final_layer == 'softmax':
            self.final = nn.Softmax(dim=1)
        else:
            self.final = lambda x: x

    def forward(self, x, *args):
        x = self.inblock1(x)
        x_res1 = x.clone()
        x = self.max1(x)

        x = self.inblock2(x)
        x_res2 = x.clone()
        x = self.max2(x)

        x = self.inblock3(x)
        x_res3 = x.clone()
        x = self.max3(x)

        x = self.inblock4(x)
        x_res4 = x.clone()
        x = self.max4(x)

        x = self.inner(x)

        x = self.upconv4(x)
        h_pad = x_res4.shape[2] - x.shape[2]
        w_pad = x_res4.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res4), dim=1)
        x = self.outblock4(x)

        x = self.upconv3(x)
        h_pad = x_res3.shape[2] - x.shape[2]
        w_pad = x_res3.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res3), dim=1)
        x = self.outblock3(x)

        x = self.upconv2(x)
        h_pad = x_res2.shape[2] - x.shape[2]
        w_pad = x_res2.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res2), dim=1)
        x = self.outblock2(x)

        x = self.upconv1(x)
        h_pad = x_res1.shape[2] - x.shape[2]
        w_pad = x_res1.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x = torch.cat((x, x_res1), dim=1)
        x = self.outblock1(x)

        x = self.out(x)
        x = self.final(x)
        del x_res1, x_res2, x_res3, x_res4
        return x


class BasicBlock2d(nn.Module):
    '''Basic 2D convolution block with 2 consecutive 2D convolutions with
    batch normalization and reLU activation after each convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
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

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 padding=0,
                 stride=2,
                 bias=False,
                 batch_norm=False,
                 relu=False):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       stride=stride,
                                       bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        #x = self.up(x)
        x = self.conv(x)
        if not self.bn is None:
            x = self.bn(x)
        if not self.relu is None:
            x = self.relu(x)
        return x
