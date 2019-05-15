# pylint: disable=redefined-outer-name
import numpy as np
import torch

import eye2you
from eye2you import models


def test_inception_v3_s():
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    inc = models.inception_v3_s(in_channels=in_channels, num_classes=out_channels)
    assert not inc is None
    inc.eval()

    x = torch.rand((1, in_channels, 299, 299))
    with torch.no_grad():
        y = inc(x)

    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, out_channels)

    x = torch.rand((2, in_channels, 299, 299))
    with torch.no_grad():
        y = inc(x)

    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, out_channels)


def test_inception_v3_s_plus_initialization():
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    inc = models.inception_v3_s_plus(in_channels=in_channels, num_classes=out_channels)
    assert not inc is None
    inc.eval()

    x = torch.rand((1, in_channels, 299, 299))
    mask = torch.rand((1, 1, 299, 299))
    with torch.no_grad():
        y = inc(x, mask, mask)

    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, out_channels)

    x = torch.rand((2, in_channels, 299, 299))
    mask = torch.rand((2, 1, 299, 299))
    with torch.no_grad():
        y = inc(x, mask, mask)

    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, out_channels)


def test_inception_v3_xs_initialization():
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    inc = models.inception_v3_xs(in_channels=in_channels, num_classes=out_channels)
    assert not inc is None
    inc.eval()

    x = torch.rand((1, in_channels, 299, 299))
    with torch.no_grad():
        y = inc(x)

    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, out_channels)

    x = torch.rand((2, in_channels, 299, 299))
    with torch.no_grad():
        y = inc(x)

    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, out_channels)


def test_inception_v3_s_wrap_initialization():
    in_channels = np.random.randint(3, 10)
    out_channels = np.random.randint(10, 20)
    inc = models.inception_v3_s_wrap(in_channels=in_channels, num_classes=out_channels)
    assert not inc is None
    inc.eval()

    # wrap adds the segment to x as additional channel, so we need to adjust for that
    x = torch.rand((1, in_channels - 1, 299, 299))
    mask = torch.rand((1, 1, 299, 299))
    with torch.no_grad():
        y = inc(x, mask, mask)

    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, out_channels)

    x = torch.rand((2, in_channels - 1, 299, 299))
    mask = torch.rand((2, 1, 299, 299))
    with torch.no_grad():
        y = inc(x, mask, mask)

    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, out_channels)


def test_inception_v3_s_pretrained():
    inc = models.inception_v3_s(num_classes=1000, pretrained=True)
    assert not inc is None


def test_inception_v3_xs_pretrained():
    inc = models.inception_v3_xs(num_classes=1000, pretrained=True)
    assert not inc is None


def test_inception_v3_s_plus_pretrained():
    inc = models.inception_v3_s_plus(in_channels=3, num_classes=1000, pretrained=True)
    assert not inc is None


def test_unet_initialization():
    for depth in range(1, 5):
        for in_channels in range(1, 3):
            for out_channels in range(1, 3):
                unet = models.u_net(in_channels=in_channels, out_channels=out_channels, depth=depth)
                assert not unet is None
                unet.eval()

                x = torch.rand((1, in_channels, 128, 128))
                mask = torch.rand((1, 1, 128, 128))
                with torch.no_grad():
                    y = unet(x, mask)

                assert y is not None
                assert isinstance(y, torch.Tensor)
                assert y.shape == (1, out_channels, 128, 128)


def test_unet_rec_initialization():
    for depth in range(1, 5):
        for in_channels in range(1, 3):
            for out_channels in range(1, 3):
                unet = eye2you.unet.u_net_rec(in_channels=in_channels, out_channels=out_channels, depth=depth)
                assert not unet is None
                unet.eval()

                x = torch.rand((1, in_channels, 128, 128))
                mask = torch.rand((1, 1, 128, 128))
                with torch.no_grad():
                    y = unet(x, mask)

                assert y is not None
                assert isinstance(y, torch.Tensor)
                assert y.shape == (1, out_channels, 128, 128)


def test_directnet_initialization():
    for in_channels in range(1, 3):
        for out_channels in range(1, 3):
            dnet = models.directnet(in_channels=in_channels, out_channels=out_channels)
            assert not dnet is None
            dnet.eval()

            x = torch.rand((1, in_channels, 128, 128))
            mask = torch.rand((1, 1, 128, 128))
            with torch.no_grad():
                y = dnet(x, mask)

            assert y is not None
            assert isinstance(y, torch.Tensor)
            assert y.shape == (1, out_channels, 128, 128)


def test_unet_upconv():
    conv = eye2you.unet.UpConv2d(2, 2, 2, 0, 2, True, True, True)
    assert conv is not None
    x = torch.randn((2, 2, 32, 32))
    y = conv(x)
    assert y.shape == (2, 2, 64, 64)
