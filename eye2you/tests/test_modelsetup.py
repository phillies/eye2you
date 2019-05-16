# pylint: disable=redefined-outer-name
import numpy as np
import torch

import eye2you
from eye2you import models


def test_inception_initialization():
    inc = models.inception_v3()
    assert inc is not None


def test_inception_v3_s_initialization():
    inc = models.inception_v3_s()
    assert not inc is None
    inc = None

    inc = models.inception_v3_s(pretrained=True)
    assert not inc is None

    inc = None
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    inc = models.inception_v3_s(in_channels=in_channels, num_classes=out_channels)
    assert list(inc.children())[0].conv.in_channels == in_channels
    assert list(inc.children())[-1].out_features == out_channels


def test_inception_v3_s_plus_initialization():
    inc = models.inception_v3_s_plus()
    assert not inc is None
    inc = None

    #TODO: provide pretrained inception v3 s
    #inc = models.inception_v3_s(pretrained=True)
    #assert not inc is None

    inc = None
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    inc = models.inception_v3_s_plus(in_channels=in_channels, num_classes=out_channels)
    assert list(inc.children())[0].conv.in_channels == in_channels
    assert list(inc.children())[-1].out_features == out_channels


def test_inception_v3_xs_initialization():
    inc = models.inception_v3_xs()
    assert not inc is None
    inc = None

    #TODO: provide pretrained inception v3 s

    inc = models.inception_v3_xs(pretrained=True)
    assert not inc is None

    inc = None
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    inc = models.inception_v3_xs(in_channels=in_channels, num_classes=out_channels)
    assert list(inc.children())[0].conv.in_channels == in_channels
    assert list(inc.children())[-1].out_features == out_channels


def test_inception_v3_s_wrap_initialization():
    inc = models.inception_v3_s_wrap()
    assert not inc is None

    inc = None
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    inc = models.inception_v3_s_wrap(in_channels=in_channels, num_classes=out_channels)
    assert list(inc.children())[0].conv.in_channels == in_channels
    assert list(inc.children())[-1].out_features == out_channels


def test_resnet_initialization():
    res = models.resnet18()
    assert res is not None
    res = models.resnet34()
    assert res is not None
    res = models.resnet50()
    assert res is not None
    res = models.resnet101()
    assert res is not None
    res = models.resnet152()
    assert res is not None


def test_unet1_initialization():
    depth = 1
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    final_layer = 'sigmoid'
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet1)
    assert net.block1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Sigmoid)

    final_layer = 'softmax'
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet1)
    assert net.block1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Softmax)

    final_layer = None
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet1)
    assert net.block1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert callable(net.final)
    assert net.final(1234) == 1234


def test_unet2_initialization():
    depth = 2
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    final_layer = 'sigmoid'
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet2)
    assert net.block1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Sigmoid)

    final_layer = 'softmax'
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet2)
    assert net.block1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Softmax)

    final_layer = None
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet2)
    assert net.block1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert callable(net.final)
    assert net.final(1234) == 1234


def test_unet3_initialization():
    depth = 3
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    final_layer = 'sigmoid'
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet3)
    assert net.inblock1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Sigmoid)

    final_layer = 'softmax'
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet3)
    assert net.inblock1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Softmax)

    final_layer = None
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet3)
    assert net.inblock1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert callable(net.final)
    assert net.final(1234) == 1234


def test_unet4_initialization():
    depth = 4
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    final_layer = 'sigmoid'
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet4)
    assert net.inblock1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Sigmoid)

    final_layer = 'softmax'
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet4)
    assert net.inblock1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Softmax)

    final_layer = None
    net = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth, final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet4)
    assert net.inblock1.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert callable(net.final)
    assert net.final(1234) == 1234


def test_unet_recursive_initialization():
    unet_recursive_initialization(1)
    unet_recursive_initialization(2)
    unet_recursive_initialization(3)
    unet_recursive_initialization(4)


def unet_recursive_initialization(depth):
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    final_layer = 'sigmoid'
    net = eye2you.unet.u_net_rec(in_channels=in_channels,
                                 out_channels=out_channels,
                                 depth=depth,
                                 final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet)
    assert net.conv_in.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Sigmoid)

    final_layer = 'softmax'
    net = eye2you.unet.u_net_rec(in_channels=in_channels,
                                 out_channels=out_channels,
                                 depth=depth,
                                 final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet)
    assert net.conv_in.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert isinstance(net.final, torch.nn.Softmax)

    final_layer = None
    net = eye2you.unet.u_net_rec(in_channels=in_channels,
                                 out_channels=out_channels,
                                 depth=depth,
                                 final_layer=final_layer)
    assert isinstance(net, eye2you.unet.Unet)
    assert net.conv_in.conv1.in_channels == in_channels
    assert net.out.out_channels == out_channels
    assert callable(net.final)
    assert net.final(1234) == 1234


def test_directnet_initialization():
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    net = models.directnet(in_channels=in_channels, out_channels=out_channels)
    assert isinstance(net, eye2you.directnet.DirectNet)
    assert net.conv1.conv.in_channels == in_channels
    assert net.sep10.bn.num_features == out_channels
