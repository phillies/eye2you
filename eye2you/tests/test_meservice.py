# pylint: disable=redefined-outer-name
import pytest
from eye2you import MEService


def test_initialization(multi_checkpoint_file):
    mes = MEService(multi_checkpoint_file)
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