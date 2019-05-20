# pylint: disable=redefined-outer-name
from eye2you import Coach
import pathlib
import os
import pytest
from eye2you.meter_functions import TotalAccuracyMeter  # pylint: disable=unused-import
import os

from test_net import compare_network_weights, compare_network_setup
from test_datasets import compare_datasets, compare_dataloader
from test_logger import compare_logger

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


def compare_config(cfg1, cfg2):
    assert sorted(cfg1.keys()) == sorted(cfg2.keys())
    for key in cfg1.keys():
        if isinstance(cfg1[key], dict) and len(cfg1[key]) > 0:
            compare_config(cfg1[key], cfg2[key])
        elif isinstance(cfg1[key], (list, tuple)):
            for val1, val2 in zip(cfg1[key], cfg2[key]):
                assert val1 == val2
        elif isinstance(cfg1[key], (float, int, str, bool)):
            assert cfg1[key] == cfg2[key]
        else:
            assert str(cfg1[key]) == str(cfg2[key])


def compare_coaches(coach1, coach2, with_weights=True):
    compare_network_setup(coach1.net, coach2.net)
    if with_weights:
        compare_network_weights(coach1.net, coach2.net)
    compare_datasets(coach1.train_data, coach2.train_data)
    compare_datasets(coach1.validate_data, coach2.validate_data)
    compare_dataloader(coach1.train_loader, coach2.train_loader)
    compare_dataloader(coach1.validate_loader, coach2.validate_loader)
    compare_logger(coach1.log, coach2.log)
    compare_config(coach1.config, coach2.config)
    assert coach1.epochs == coach2.epochs


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

    compare_coaches(coach, coach_example, with_weights=True)

    coach.load(tmp_path / 'test.ckpt', device='cpu')
    assert coach.device == 'cpu'


def test_coach_saveload_config(tmp_path, coach_example):
    coach_example.save_config(tmp_path / 'test.yaml')

    assert os.path.exists(tmp_path / 'test.yaml')

    coach = Coach()
    coach.load_config(coach_example.config)

    compare_coaches(coach, coach_example, with_weights=False)

    coach = Coach()
    coach.load_config(tmp_path / 'test.yaml')

    compare_coaches(coach, coach_example, with_weights=False)


def test_coach_train(tmp_path, coach_example):
    assert coach_example.epochs == 0
    coach_example.run(1)
    assert coach_example.epochs == 1
    coach_example.run(1, log_filename=str(tmp_path / 'test.log'), checkpoint=str(tmp_path / 'test'))
    assert os.path.exists(tmp_path / 'test.log')
    assert os.path.exists(tmp_path / 'test.loss.ckpt')
    coach_example.run(10, early_stop_window=2, early_stop_slope=10.0)
    assert coach_example.epochs < 11


def test_coach_validate(coach_example):
    coach_example.validate()
