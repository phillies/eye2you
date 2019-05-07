# pylint: disable=redefined-outer-name
import os
import configparser
import pathlib

import numpy as np
import pytest
import torch

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def data_labels():
    return torch.Tensor(np.random.rand(100, 3)).round()


@pytest.fixture
def data_outputs():
    return torch.Tensor(np.random.randn(100, 3))

