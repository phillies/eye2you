# pylint: disable=redefined-outer-name
import os
import sys
import pathlib

import numpy as np
import pytest
from PIL import Image
import torch

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

# @pytest.fixture(scope='module')
# def sample_data():
#     data = np.random.rand(SAMPLE_NUMBER)
#     return data


def test_image_reading():
    img = eye2you.helper_functions.pil_loader(LOCAL_DIR / 'data/classA/img0.jpg')
    assert img is not None
    assert isinstance(img, Image.Image)


def test_image_conversion():
    img = eye2you.helper_functions.pil_loader(LOCAL_DIR / 'data/classA/img0.jpg')
    assert img is not None
    w, h = img.size
    d = len(img.getbands())

    torch_img = eye2you.helper_functions.PIL_to_torch(img)
    assert torch_img.shape == (1, d, h, w)

    assert img.size == (w, h)
    assert len(img.getbands()) == d
    cv2_img = eye2you.helper_functions.PIL_to_cv2(img)
    assert cv2_img.shape == (h, w, d)

    assert torch_img.shape == (1, d, h, w)
    pil_img = eye2you.helper_functions.torch_to_PIL(torch_img)
    assert pil_img.size == (w, h)
    assert len(pil_img.getbands()) == d

    assert torch_img.shape == (1, d, h, w)
    cv2_img = eye2you.helper_functions.torch_to_cv2(torch_img)
    assert cv2_img.shape == (h, w, d)

    assert cv2_img.shape == (h, w, d)
    pil_img = eye2you.helper_functions.cv2_to_PIL(cv2_img)
    assert pil_img.size == (w, h)
    assert len(pil_img.getbands()) == d

    assert cv2_img.shape == (h, w, d)
    torch_img = eye2you.helper_functions.cv2_to_torch(cv2_img)
    assert torch_img.shape == (1, d, h, w)

    for ii in range(d):
        np.testing.assert_almost_equal(cv2_img[:, :, ii], torch_img[0, ii, :, :])


def test_split_and_merge_patches():
    patch_size = np.random.randint(5, 40)
    n_h, n_w = np.random.randint(5, 20, 2)
    c = np.random.randint(1, 5)

    img = torch.randn(c, patch_size * n_h, patch_size * n_w)
    patches = eye2you.helper_functions.split_tensor_image_into_patches(img, patch_size)
    assert patches.shape == (n_w * n_h, c, patch_size, patch_size)

    img_new = eye2you.helper_functions.merge_tensor_image_from_patches(patches, (n_h, n_w))
    np.testing.assert_allclose(img, img_new)

    img = torch.randn(c, patch_size * n_h, patch_size * n_h)
    patches = eye2you.helper_functions.split_tensor_image_into_patches(img, patch_size)
    assert patches.shape == (n_h * n_h, c, patch_size, patch_size)

    img_new = eye2you.helper_functions.merge_tensor_image_from_patches(patches)
    np.testing.assert_allclose(img, img_new)
