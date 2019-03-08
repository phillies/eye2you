import os

import pytest
import configparser
import numpy as np
from PIL import Image

from eye2you import Service
from eye2you import RetinaChecker

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
def default_service(checkpoint_file):
    service = Service(checkpoint_file)
    return service

def test_create_service(checkpoint_file):
    with pytest.raises(ValueError):
        service = Service(None)

    with pytest.raises(ValueError):
        service = Service('None')

    service = Service(checkpoint_file)
    assert service is not None

def test_classify():
    assert False #TODO: implement me

def test_cam():
    assert False #TODO: implement me

def test_contour():
    assert False #TODO: implement me


