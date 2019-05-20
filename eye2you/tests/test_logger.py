# pylint: disable=redefined-outer-name,protected-access
import numpy as np
import os

from eye2you.train import Logger


def compare_logger(log1, log2):
    for cat in log1._log.keys():
        np.testing.assert_allclose(log1._log[cat].values, log2._log[cat].values)
    assert all([col in log1.columns for col in log2.columns])
    assert all([col in log2.columns for col in log1.columns])
    assert all([cat in log1._log.keys() for cat in log2._log.keys()])
    for col in log1.columns:
        for cat in log1._log.keys():
            np.testing.assert_allclose(log1._log[cat][col].values, log2._log[cat][col].values)


def test_logger_init():
    log = Logger()
    assert log is not None
    assert isinstance(log._log, dict)
    assert len(log._log) == 0


def test_logger_logging():
    log = Logger()
    category1 = 'Cat 1'
    category2 = 'Cat 2'
    data1 = np.random.randn(10, 3)
    data2 = np.random.randn(10, 3)
    for ii in range(10):
        log.append(data1[ii], category1)
    for ii in range(10):
        log.append(data2[ii], category2)
    np.testing.assert_allclose(data1, log._log[category1].values)
    np.testing.assert_allclose(data2, log._log[category2].values)


def test_logger_slopetest():
    log = Logger()
    log.columns = ['loss']

    # testing the slope estimation
    slope = np.random.randn(1)
    data = np.arange(50) * slope + np.random.rand(1) * 10
    for ii in range(50):
        log.append(data[ii], 'Slopetest')

    for winlength in (5, 10, 20, 50, 100):
        slope_est = log.get_slope('Slopetest', 'loss', winlength)
        np.testing.assert_allclose(slope, slope_est)

    slope_est = log.get_slope('Slopetest', None, 2)
    np.testing.assert_allclose(slope, slope_est)


def test_logger_minmax():
    # testing max/min
    log = Logger()
    log.columns = ['loss', 'data']
    data1 = np.random.randn(50, 2)
    for ii in range(50):
        log.append(data1[ii, :], 'Cat1')
    data2 = np.random.randn(50, 2)
    for ii in range(50):
        log.append(data2[ii, :], 'Cat2')

    idx, val = log.get_best('Cat2', 'loss', 'max')
    np.testing.assert_allclose(val, data2[data2[:, 0].argmax(), :])
    assert idx == data2[:, 0].argmax()

    idx, val = log.get_best('Cat2', 'data', 'min')
    np.testing.assert_allclose(val, data2[data2[:, 1].argmin(), :])
    assert idx == data2[:, 1].argmin()

    idxmax, idxmin = log.idxmaxmin('Cat1')
    assert idxmax[log.columns[0]] == data1[:, 0].argmax()
    assert idxmax[log.columns[1]] == data1[:, 1].argmax()
    assert idxmin[log.columns[0]] == data1[:, 0].argmin()
    assert idxmin[log.columns[1]] == data1[:, 1].argmin()


def test_logger_saveload(tmp_path):
    log = Logger()
    log.columns = ['loss', 'data']
    data1 = np.random.randn(50, 2)
    for ii in range(50):
        log.append(data1[ii, :], 'Cat1')
    data2 = np.random.randn(50, 2)
    for ii in range(50):
        log.append(data2[ii, :], 'Cat2')

    log.to_csv(tmp_path / 'test.csv')

    assert os.path.exists(tmp_path / 'test.csv')

    log2 = Logger()
    log2.read_csv(tmp_path / 'test.csv')

    compare_logger(log, log2)
