import configparser
import os

import numpy as np
import pytest
import torch

import eye2you
from eye2you import RetinaChecker

def test_create_checker(tmp_path, example_config):
    # Reading configuration file
    config = configparser.ConfigParser()
    config.read_string(example_config)

    # create the checker class and initialize internal variables
    rc = RetinaChecker()
    rc.initialize(config)

    assert rc.initialized

    # Initialize the model
    rc.initialize_model()
    rc.initialize_criterion()
    rc.initialize_optimizer()

    assert not rc.model is None
    assert not rc.criterion is None
    assert not rc.optimizer is None

    rc.save_state(tmp_path / 'tmpmodel.ckpt')

    assert os.path.isfile(tmp_path / 'tmpmodel.ckpt')

def test_loading_data():
    assert False #TODO: implement me

def test_creating_dataloader():
    assert False #TODO: implement me

def test_reloading():
    assert False #TODO: implement me

def test_load_data_split():
    assert False #TODO: implement me

def test_train():
    assert False #TODO: implement me

def test_validate():
    assert False #TODO: implement me
