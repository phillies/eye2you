import pytest
import numpy as np
import torch


@pytest.fixture(scope='module')
def example_config():
    CONFIG_STRING = '''[network]
    model = inception_v3

    [hyperparameter]

    [files]
    train file = ./images/label.csv
    train root = ./images/data/

    [transform]
    [output]
    [input]'''
    return CONFIG_STRING

@pytest.fixture
def data_labels():
    return torch.Tensor(np.random.rand(100,3)).round()

@pytest.fixture
def data_outputs():
    return torch.Tensor(np.random.randn(100,3))