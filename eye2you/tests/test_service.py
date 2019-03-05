import pytest
import configparser
import numpy as np
from PIL import Image
from eye2you import Service
from eye2you import RetinaChecker

CONFIG_STRING = '''[network]
model = inception_v3

[hyperparameter]

[files]
train file = ./images/label.csv
train root = ./images/data/

[transform]
[output]
[input]'''

def test_classify():
    pass

def test_cam():
    pass

def test_contour():
    pass

def test_create_service(tmp_path):
    # Reading configuration file
    config = configparser.ConfigParser()
    config.read_string(CONFIG_STRING)

    # create the checker class and initialize internal variables
    rc = RetinaChecker()
    rc.initialize(config)

    # Initialize the model
    rc.initialize_model()
    rc.initialize_criterion()
    rc.initialize_optimizer()

    rc.save_state(tmp_path / 'tmpmodel.ckpt')

    service = Service(tmp_path / 'tmpmodel.ckpt')    
    assert service is not None
