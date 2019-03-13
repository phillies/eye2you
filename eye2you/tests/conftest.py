# pylint: disable=redefined-outer-name
import pytest
import numpy as np
import torch
import configparser
from eye2you import RetinaChecker


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

@pytest.fixture(scope='module')
def retina_checker(example_config):
    # Reading configuration file
    config = configparser.ConfigParser()
    config.read_string(example_config)

    # create the checker class and initialize internal variables
    rc = RetinaChecker()
    rc.initialize(config)

    # Initialize the model
    rc.initialize_model()
    rc.initialize_criterion()
    rc.initialize_optimizer()
    return rc

@pytest.fixture(scope='module')
def checkpoint_file(tmp_path_factory, retina_checker):
    model_path = tmp_path_factory.mktemp('ckpt')
    filename = model_path / 'tmpmodel.ckpt'
    retina_checker.save_state(filename)
    return filename