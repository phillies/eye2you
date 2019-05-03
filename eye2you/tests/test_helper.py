# pylint: disable=redefined-outer-name
import os
import sys
import pathlib
import configparser
from unittest.mock import patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

import eye2you
from eye2you import models

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
RANDOM_SEED = 1337
SAMPLE_NUMBER = 1000

NUMBER_OF_CLASSES = 2
NUMBER_OF_IMAGES = 4

# convert each random number [0,1) into an integer k=[1,PSEUDO_SAMPLE_NUMBER]
# to simulate that k out of PSEUDO_SAMPLE_SIZE were correct
PSEUDO_SAMPLE_SIZE = np.random.randint(1, 100)

# maximum delta to pass numerical comparison
EPSILON = 1e-9

np.random.seed(RANDOM_SEED)


@pytest.fixture(scope='module')
def sample_data():
    data = np.random.rand(SAMPLE_NUMBER)
    return data


def test_inception():
    inc = models.inception_v3()
    assert inc is not None


def test_inception_v3_s():
    inc = models.inception_v3_s()
    assert not inc is None
    inc = None

    inc = models.inception_v3_s(pretrained=True)
    assert not inc is None


def test_inception_v3_xs():
    inc = models.inception_v3_xs()
    assert not inc is None
    inc = None

    inc = models.inception_v3_xs(pretrained=True)
    assert not inc is None


def test_resnet():
    res = models.resnet18()
    assert res is not None
    res = models.resnet34()
    assert res is not None
    res = models.resnet50()
    assert res is not None
    res = models.resnet101()
    assert res is not None
    res = models.resnet152()
    assert res is not None


def test_performance_meters(data_labels, data_outputs):
    num_correct_all = ((data_outputs > 0).numpy() == data_labels.numpy()).all(axis=1).sum()
    num_correct_single = ((data_outputs > 0).numpy() == data_labels.numpy()).sum(axis=0)

    fake_labels = data_labels - 0.5
    num_correct = eye2you.meter_functions.all_or_nothing_performance(data_labels, fake_labels)
    assert num_correct == len(data_labels)

    num_correct = eye2you.meter_functions.all_or_nothing_performance(data_labels, data_outputs)
    assert num_correct >= 0
    assert num_correct <= len(data_labels)
    assert num_correct == num_correct_all

    for ii in range(data_labels.size()[1]):
        num_correct = eye2you.meter_functions.single_output_performance(data_labels, fake_labels, ii)
        assert num_correct == len(data_labels)
        num_correct = eye2you.meter_functions.single_output_performance(data_labels, data_outputs, ii)
        assert num_correct >= 0
        assert num_correct <= len(data_labels)
        assert num_correct == num_correct_single[ii]


def test_performance_meters_tuple(data_labels, data_outputs):
    num_correct_all = ((data_outputs > 0).numpy() == data_labels.numpy()).all(axis=1).sum()
    num_correct_single = ((data_outputs > 0).numpy() == data_labels.numpy()).sum(axis=0)

    fake_labels = data_labels - 0.5
    num_correct = eye2you.meter_functions.all_or_nothing_performance(data_labels, (fake_labels, 0))
    assert num_correct == len(data_labels)

    num_correct = eye2you.meter_functions.all_or_nothing_performance(data_labels, (data_outputs, 0))
    assert num_correct >= 0
    assert num_correct <= len(data_labels)
    assert num_correct == num_correct_all

    for ii in range(data_labels.size()[1]):
        num_correct = eye2you.meter_functions.single_output_performance(data_labels, (fake_labels, 0), ii)
        assert num_correct == len(data_labels)
        num_correct = eye2you.meter_functions.single_output_performance(data_labels, (data_outputs, 0), ii)
        assert num_correct >= 0
        assert num_correct <= len(data_labels)
        assert num_correct == num_correct_single[ii]


def test_image_reading():
    img = eye2you.helper_functions.pil_loader(LOCAL_DIR / 'data/classA/img0.jpg')
    assert not img is None
    assert isinstance(img, Image.Image)


@patch('eye2you.helper_functions.sys')
def test_data_reading_pre35(mock_sys):
    path = LOCAL_DIR / 'data'
    type(mock_sys).version_info = PropertyMock(return_value=(3, 4))
    classes, class_to_idx = eye2you.helper_functions.find_classes(path)
    assert len(classes) == NUMBER_OF_CLASSES
    assert len(class_to_idx) == NUMBER_OF_CLASSES
    for ii in range(NUMBER_OF_CLASSES):
        assert class_to_idx[classes[ii]] == ii
