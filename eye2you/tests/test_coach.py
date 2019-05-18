# pylint: disable=redefined-outer-name
from eye2you import Coach
import pathlib
import os
import pytest
from eye2you.meter_functions import TotalAccuracyMeter

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def coach_example():
    coach = Coach()
    coach.load_config(LOCAL_DIR / 'data/example.yaml')
    return coach


def test_coach_load():
    coach = Coach()
    assert coach is not None
    assert coach.epochs == 0

    coach.load_config(LOCAL_DIR / 'data/example.yaml')

    assert coach.train_data is not None
    assert coach.validate_data is not None
    assert coach.train_loader is not None
    assert coach.validate_loader is not None
    assert coach.net is not None
    assert coach.device is not None
    assert coach.log is not None
    assert coach.log.columns is not None


def test_coach_saveload_checkpoint(tmp_path, coach_example):
    coach_example.save(tmp_path / 'test.ckpt')

    assert os.path.exists(tmp_path / 'test.ckpt')

    coach = Coach()
    coach.load(tmp_path / 'test.ckpt')

    # compare network
    # compare train_data
    # compare validate_data
    # compare loaders
    # compare log
    # compare config
    assert coach.epochs == coach_example.epochs


# def test_coach_saveload_config(tmp_path, coach_example):
#     coach_example.save_config(tmp_path / 'test.yaml')

#     assert os.path.exists(tmp_path / 'test.yaml')

#     coach = Coach()
#     coach.load_config(tmp_path / 'test.yaml')

#     # compare network
#     # compare train_data
#     # compare validate_data
#     # compare loaders
#     # compare log
#     # compare config
#     assert coach.epochs == coach_example.epochs

# def test_coach_train(coach_example):
#     coach_example.run(1)


def test_coach_validate(coach_example):
    coach_example.validate()
