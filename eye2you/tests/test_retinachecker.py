import configparser
import os
import pathlib

import numpy as np
import pytest
import torch

import eye2you
import eye2you.make_default_config
from eye2you import RetinaChecker


def test_create_checker(tmp_path, example_config):
    # Reading configuration file
    config = configparser.ConfigParser()
    config.read_string(example_config)

    # create the checker class and initialize internal variables
    rc = RetinaChecker()
    assert not rc.initialized
    assert str(rc).count('not initialized')>0

    rc.initialize(config)

    assert rc.initialized
    assert str(rc).count('not initialized')==0

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
    #assert False #TODO: implement me
    pass

def test_creating_dataloader():
    #assert False #TODO: implement me
    pass

def test_reloading():
    #assert False #TODO: implement me
    pass

def test_load_data_split():
    #assert False #TODO: implement me
    pass

def test_train_and_validation():
    cfg = '''[network]
    model = inception_v3_s

    [hyperparameter]
    batch size = 16

    [files]
    train file = {localdir}/data.csv
    train root = {localdir}/data/
    test file = {localdir}/data.csv
    test root = {localdir}/data/
    samples = 32

    [transform]
    [output]
    [input]'''.format(localdir=pathlib.Path(os.path.dirname(os.path.realpath(__file__))))

    # Reading configuration file
    config = configparser.ConfigParser()
    config.read_string(cfg)

    # create the checker class and initialize internal variables
    rc = RetinaChecker()
    rc.initialize(config)

    # Loading data sets based on configuration and enable normaization
    rc.load_datasets()

    # Initializing sampler and data (=patch) loader
    rc.create_dataloader()

    # Initialize the model
    rc.initialize_model()
    rc.initialize_criterion()
    rc.initialize_optimizer()

    # Performance meters initalized (either empty or from file)
    #num_epochs = rc.start_epoch + config['hyperparameter'].getint('epochs', 2)

    # Starting training & evaluation
    #for epoch in range(rc.start_epoch, num_epochs):

    # Train the model and record training loss & accuracy
    losses, accuracy = rc.train()
    assert losses is not None
    assert accuracy is not None
    
    # Validation
    losses, accuracy, confusion = rc.validate()
    assert losses is not None
    assert accuracy is not None
    assert confusion is not None

def test_validate():
    #assert False #TODO: implement me
    pass

def test_printing():
    # Pretrained
    # with dataset and workers
    #assert False #TODO: implement me
    pass

def test_initialize_unknown_config():
    #String but no config or checkpoint
    #not string not config
    #assert False #TODO: implement me
    pass

def test_loading(checkpoint_file):
    # with an without filename (filename in config->inpit->checkpoint)
    # with optimizer not None
        # with scheduler in checkoiunt and not 
    #assert False #TODO: implement me
    pass

def test_parse_config(checkpoint_file):
    rc = RetinaChecker()
    assert rc.config is None
    with pytest.raises(ValueError):
        rc._parse_config()
    
    config = eye2you.make_default_config.get_config()
    rc.initialize(config)
    rc.initialize_model()
    rc.initialize_criterion()
    rc.initialize_optimizer()
    rc.split_indices = ([1,2,3], [4,5,6])

    rc.save_state(checkpoint_file)

def test_warning_unknown_names():
    rc = RetinaChecker()  
    config = eye2you.make_default_config.get_config()
    rc.initialize(config)

    rc.model_name = None
    rc.optimizer_name = None
    rc.criterion_name = None

    with pytest.warns(Warning):
        rc.initialize_model()
    with pytest.warns(Warning):
        rc.initialize_criterion()
    with pytest.warns(Warning):
        rc.initialize_optimizer()
