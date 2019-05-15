# pylint: disable=redefined-outer-name
import os
import configparser
import pathlib

import numpy as np
import pytest
import torch

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def data_classification_labels():
    return torch.Tensor(np.random.rand(100, 3)).round()


@pytest.fixture
def data_classification_outputs():
    return torch.Tensor(np.random.randn(100, 3))


@pytest.fixture(scope='module')
def image_filename():
    return LOCAL_DIR / 'data/classA/img0.jpg'


@pytest.fixture(scope='module')
def image_set():
    files = [
        LOCAL_DIR / 'data/classA/img0.jpg',
        LOCAL_DIR / 'data/classA/img1.jpg',
        LOCAL_DIR / 'data/classB/img2.jpg',
        LOCAL_DIR / 'data/classB/img3.jpg',
    ]
    masks = [
        LOCAL_DIR / 'data/classA/img0_mask.png',
        LOCAL_DIR / 'data/classA/img1_mask.png',
        LOCAL_DIR / 'data/classB/img2_mask.png',
        LOCAL_DIR / 'data/classB/img3_mask.png',
    ]
    return files, masks


@pytest.fixture
def segmentation_examples():
    targets = torch.zeros((1, 1, 10, 10)).byte()
    output1 = torch.zeros((1, 1, 10, 10))
    output2 = torch.zeros((1, 1, 10, 10))

    targets[0, 0, 3:7, 4:9] = 1  #4x5 patch

    # print(P, N, TN, TP, FP, FN)
    # tensor([20.]) tensor([80.]) tensor([70.]) tensor([15.]) tensor([10.]) tensor([5.])
    # tensor([20.]) tensor([80.]) tensor([64.]) tensor([7.]) tensor([16.]) tensor([12.])

    # 5x5 patch, 0.85 accuracy, 0.75 precision, 0.6 recall, 0.875 specificity, 0.5 iou, 2/3 dice
    output1[0, 0, 1:6, 4:9] = 0.51

    # 4x6 patch, 0.72 accuracy, 1/3 precision, 0.4 recall, 0.8 specificity, 2/9 iou, 16/44 dice
    output2[0, 0, 3:7, 0:6] = 0.51

    return targets, output1, output2
