# pylint: disable=redefined-outer-name
import pytest
from eye2you import MEService


def test_initialization():
    filename = 'd:/ckpt/model_test.ckpt'
    mes = MEService(filename)
    assert mes is not None
    assert mes.number_of_experts == 2
    assert len(mes.retina_checker) == 2
    for rc in mes.retina_checker:
        assert rc.initialized


def test_wrong_initialization():
    mes = MEService(None)
    with pytest.raises(ValueError):
        mes.initialize()
    with pytest.raises(OSError):
        mes = MEService('abc')