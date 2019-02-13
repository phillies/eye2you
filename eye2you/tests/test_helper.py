import pytest
import numpy as np
from eye2you.meter_functions import AccuracyMeter, AverageMeter
from eye2you import model_wrapper as models

import os
import pathlib

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

def test_AverageMeter(sample_data):
    meter = AverageMeter()
    for ii in range(len(sample_data)):
        meter.update(sample_data[ii])
    assert abs(meter.avg - sample_data.mean()) < EPSILON

def test_inception():
    inc = models.inception_v3_s()
    assert inc is not None
    inc = models.inception_v3()
    assert inc is not None
    inc = models.inception_v3_xs()
    assert inc is not None

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

def test_vgg():
    vgg = models.vgg11()
    assert vgg is not None
    vgg = models.vgg13()
    assert vgg is not None
    vgg = models.vgg16()
    assert vgg is not None
    vgg = models.vgg19()
    assert vgg is not None
    vgg = models.vgg11_bn()
    assert vgg is not None
    vgg = models.vgg13_bn()
    assert vgg is not None
    vgg = models.vgg16_bn()
    assert vgg is not None
    vgg = models.vgg19_bn()
    assert vgg is not None
