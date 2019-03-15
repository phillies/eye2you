# pylint: disable=redefined-outer-name
import os
import pathlib

import pytest
import configparser
import numpy as np
from PIL import Image

import eye2you.io_helper
from eye2you import Service
from eye2you import RetinaChecker

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(scope='module')
def default_service(checkpoint_file):
    service = Service(checkpoint_file)
    return service


def test_create_service(checkpoint_file):
    with pytest.raises(ValueError):
        service = Service(None)
        service.initialize()

    with pytest.raises(ValueError):
        service = Service('None')

    service = Service(checkpoint_file)
    assert service is not None


def test_classify(default_service):
    img = eye2you.io_helper.pil_loader(LOCAL_DIR / 'data/classA/img0.jpg')
    assert not img is None
    assert isinstance(img, Image.Image)
    prediction = default_service.classify_image(img)
    assert prediction is not None
    assert len(prediction) == 2
    max_pred = default_service.get_largest_prediction(img)
    assert prediction.argmax() == max_pred

    img2 = eye2you.io_helper.PIL_to_cv2(img)
    assert isinstance(img2, np.ndarray)
    prediction2 = default_service.classify_image(img2)
    np.testing.assert_allclose(prediction, prediction2)


def test_cam(default_service):
    img = eye2you.io_helper.pil_loader(LOCAL_DIR / 'data/classA/img0.jpg')
    assert not img is None
    pil_cam = default_service.get_class_activation_map(img)
    assert isinstance(pil_cam, list)
    assert all([isinstance(cam, Image.Image) for cam in pil_cam])

    pil_cam = default_service.get_class_activation_map(img, 0, False)
    assert len(pil_cam) == 1
    assert isinstance(pil_cam[0], np.ndarray)

    pil_cam = default_service.get_class_activation_map(img, (0, 1), False)
    assert len(pil_cam) == 2

    with pytest.raises(ValueError):
        pil_cam = default_service.get_class_activation_map(img, 'bla', False)


def test_contour():
    assert True  #TODO: implement me
