# pylint: disable=redefined-outer-name
import os
import pathlib
import sys

import numpy as np
import pytest
import torch
import torchvision
import yaml
from PIL import Image

import eye2you
from eye2you import factory

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


def test_yaml_loader():
    config = factory.config_from_yaml(LOCAL_DIR / 'data/example.yaml')
    assert isinstance(config, dict)


def test_load_csv():
    filename = LOCAL_DIR / 'data/test.csv'
    root = LOCAL_DIR / 'data/'

    samples, masks, segmentations, targets, target_labels = factory.load_csv(filename, root)
    assert samples is not None
    assert masks is not None
    assert segmentations is not None
    assert targets is not None
    assert target_labels is not None
    assert len(samples) == 4
    assert len(masks) == 4
    assert len(segmentations) == 4
    assert len(targets) == 4
    assert len(target_labels) == 1

    filename = LOCAL_DIR / 'data/test_classification.csv'
    root = LOCAL_DIR / 'data/'

    samples, masks, segmentations, targets, target_labels = factory.load_csv(filename,
                                                                             root,
                                                                             mask_column_name='xxx',
                                                                             segmentation_column_name='xxx',
                                                                             target_column_names=['class A', 'class B'])
    assert samples is not None
    assert masks is None
    assert segmentations is None
    assert targets is not None
    assert target_labels is not None
    assert len(samples) == 4
    assert len(targets) == 4
    assert len(target_labels) == 2


def test_data_from_config():
    config = yaml.full_load(f'''
    csv: {str(LOCAL_DIR)}/data/test_classification.csv
    root: {str(LOCAL_DIR)}/data/
    validation:
        csv: {str(LOCAL_DIR)}/data/test_classification.csv
        root: {str(LOCAL_DIR)}/data/
    ''')
    training_data, validation_data = factory.data_from_config(config)

    assert training_data is not None
    assert validation_data is not None
    assert len(training_data) == 4
    assert len(validation_data) == 4

    config = yaml.full_load(f'''
    csv: {str(LOCAL_DIR)}/data/test_classification.csv
    root: {str(LOCAL_DIR)}/data/
    test_size: 0.5
    stratified: True
    ''')
    training_data, validation_data = factory.data_from_config(config)

    assert training_data is not None
    assert validation_data is not None
    assert len(training_data) == 2
    assert len(validation_data) == 2

    config = yaml.full_load(f'''
    csv: {str(LOCAL_DIR)}/data/test.csv
    root: {str(LOCAL_DIR)}/data/
    test_size: 0.5
    stratified: False
    ''')
    training_data, validation_data = factory.data_from_config(config)

    assert training_data is not None
    assert validation_data is not None
    assert len(training_data) == 2
    assert len(validation_data) == 2


def test_get_loader():
    config = yaml.full_load(f'''
    csv: {str(LOCAL_DIR)}/data/test_classification.csv
    root: {str(LOCAL_DIR)}/data/
    validation:
        csv: {str(LOCAL_DIR)}/data/test_classification.csv
        root: {str(LOCAL_DIR)}/data/
    ''')
    training_data, _ = factory.data_from_config(config)

    config = yaml.full_load(f'''
    num_samples: 20
    batch_size: 2
    num_workers: 0
    replacement: 
    weighted_sampling_classes: [0,1]
    drop_last: True
    ''')

    loader = factory.get_loader(config, training_data)
    assert loader is not None
    assert loader.batch_size == 2
    assert loader.sampler.num_samples == 20

    config = yaml.full_load(f'''
    num_samples: 20
    batch_size: 2
    num_workers: 0
    replacement: True
    batch_size_increase: 2
    drop_last: 
    ''')

    loader = factory.get_loader(config, training_data, 1)
    assert loader is not None
    assert loader.batch_size == 4
    assert loader.sampler.num_samples == 20

    config = yaml.full_load(f'''
    num_samples: 
    batch_size: 2
    num_workers: 0
    replacement: True
    weighted_sampling_classes: 
    drop_last: 
    ''')

    loader = factory.get_loader(config, training_data)
    assert loader is not None
    assert loader.batch_size == 2
    assert loader.sampler.num_samples == 4

    config = yaml.full_load(f'''
    num_samples: 
    batch_size: 2
    num_workers: 0
    replacement: False
    drop_last: 
    ''')

    loader = factory.get_loader(config, training_data)
    assert loader is not None
    assert loader.batch_size == 2
    assert loader.sampler.num_samples == 4


def test_configparser():
    config = {
        0: dict(),
        1: list(),
        2: tuple(),
        3: int(5),
        4: float(3.5),
        5: str('abc'),
        6: bool(True),
        7: None,
        8: torchvision.transforms.ToTensor(),
    }
    vals = config.values()
    config[0] = config.copy()
    config[1] = list(vals)
    config[2] = tuple(vals)
    yamlcfg = factory.yamlize_config(config)
    assert yamlcfg is not None


# def test_transform():
#     trans_string = repr(torchvision.transforms.ToTensor())
#     trans = factory.get_transform(trans_string)
#     assert repr(trans) == trans_string

#     trans_list = [
#         repr(torchvision.transforms.ToTensor()),
#         repr(torchvision.transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])),
#     ]
#     trans = factory.get_transform(trans_list)
#     for t1, t2 in zip(trans_list, trans):
#         assert t1 == repr(t2)
