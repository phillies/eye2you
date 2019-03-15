# pylint: disable=redefined-outer-name
import os
import configparser
import pathlib

import numpy as np
import pytest
import torch

from eye2you import RetinaChecker

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope='module')
def example_config():
    CONFIG_STRING = '''[network]
    model = inception_v3

    [hyperparameter]

    [files]
    train file = ./data.csv
    train root = ./data/

    [transform]
    [output]
    [input]'''
    return CONFIG_STRING


@pytest.fixture
def data_labels():
    return torch.Tensor(np.random.rand(100, 3)).round()


@pytest.fixture
def data_outputs():
    return torch.Tensor(np.random.randn(100, 3))


@pytest.fixture(scope='module')
def retina_checker(example_config):
    # Reading configuration file
    config = configparser.ConfigParser()
    config.read_string(example_config)

    # create the checker class and initialize internal variables
    rc = RetinaChecker()
    rc.initialize(config)
    rc.train_file = LOCAL_DIR / rc.train_file
    rc.train_root = LOCAL_DIR / rc.train_root
    rc.load_datasets(0.5)

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


@pytest.fixture(scope='module')
def multi_checkpoint_file(tmp_path_factory, retina_checker):
    model_path = tmp_path_factory.mktemp('ckpt')
    filename = model_path / 'me_model.ckpt'
    models = []
    config = retina_checker.config_string
    classes = retina_checker.classes
    state_dict = retina_checker.model.state_dict()
    models.append(state_dict)
    models.append(state_dict)
    torch.save({'models': models, 'config': config, 'classes': classes}, filename)
    return filename