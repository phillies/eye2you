# pylint: disable=redefined-outer-name
import os

import pytest
import configparser
import numpy as np
from PIL import Image

from eye2you import Service
from eye2you import RetinaChecker


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


def test_classify():
    print(default_service)
    assert True  #TODO: implement me


def test_cam():
    assert True  #TODO: implement me


def test_contour():
    assert True  #TODO: implement me
