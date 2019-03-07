import os
import pathlib
import configparser

import numpy as np
import pytest

import eye2you
import eye2you.make_default_config
from eye2you import models
from eye2you.meter_functions import AccuracyMeter, AverageMeter

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
RANDOM_SEED = 1337
SAMPLE_NUMBER = 1000

# convert each random number [0,1) into an integer k=[1,PSEUDO_SAMPLE_NUMBER]
# to simulate that k out of PSEUDO_SAMPLE_SIZE were correct
PSEUDO_SAMPLE_SIZE = np.random.randint(1,100)

# maximum delta to pass numerical comparison
EPSILON = 1e-9

np.random.seed(RANDOM_SEED)

@pytest.fixture(scope='module')
def sample_data():
    data = np.random.rand(SAMPLE_NUMBER)
    return data

def test_AccuracyMeter(sample_data):
    meter = AccuracyMeter()
    accuracy_data = np.round(sample_data*PSEUDO_SAMPLE_SIZE)
    for ii in range(len(accuracy_data)):
        meter.update(accuracy_data[ii], PSEUDO_SAMPLE_SIZE)
    assert abs(meter.avg - accuracy_data.sum()/sample_data.size/PSEUDO_SAMPLE_SIZE) < EPSILON
    meter.reset()
    assert meter.avg==0
    assert meter.count==0
    assert meter.sum==0
    for ii in range(len(accuracy_data)):
        meter.update(accuracy_data[ii], PSEUDO_SAMPLE_SIZE)
    assert abs(meter.avg - accuracy_data.sum()/sample_data.size/PSEUDO_SAMPLE_SIZE) < EPSILON

def test_AverageMeter(sample_data):
    meter = AverageMeter()
    for ii in range(len(sample_data)):
        meter.update(sample_data[ii])
    assert abs(meter.avg - sample_data.mean()) < EPSILON
    meter.reset()
    assert meter.avg==0
    assert meter.val==0
    assert meter.count==0
    assert meter.sum==0
    for ii in range(len(sample_data)):
        meter.update(sample_data[ii])
    assert abs(meter.avg - sample_data.mean()) < EPSILON

def test_inception():
    inc = models.inception_v3()
    assert inc is not None

def test_inception_v3_s():
    inc = models.inception_v3_s()
    assert not inc is None
    inc = None

    with pytest.warns(Warning):
        inc = models.inception_v3_s(pretrained=True)
        assert not inc is None

def test_inception_v3_xs():
    inc = models.inception_v3_xs()
    assert not inc is None
    inc = None

    with pytest.warns(Warning):
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


def test_default_config(tmp_path):
    config = eye2you.make_default_config.get_config()

    assert not config is None
    assert isinstance(config, configparser.ConfigParser)

    filename = tmp_path / 'test.cfg'
    assert not os.path.exists(filename)

    eye2you.make_default_config.save_config(filename)
    assert os.path.isfile(filename)

    filename = tmp_path / 'test2.cfg'
    assert not os.path.exists(filename)

    eye2you.make_default_config.save_config(filename, config)
    assert os.path.isfile(filename)
