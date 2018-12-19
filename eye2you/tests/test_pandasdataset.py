import pytest
from eye2you import PandasDataset

import os
import pathlib

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))

@pytest.fixture(scope='module')
def simple_dataset():
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'))
    return data

def test_modes():
    with pytest.raises(ValueError):
        _ = PandasDataset(mode=None)

def test_create_datasets():
    # Create dataset from csv
    print((LOCAL_DIR / 'data.csv'))
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'))
    assert len(data)==4
    data2 = PandasDataset(source=data.samples, mode='pandas')
    assert len(data2)==4
    assert (data.classes==data2.classes).all()
    assert (data.filenames==data2.filenames).all()
    assert (data.targets==data2.targets).all()
    assert data.transform.__str__()==data2.transform.__str__()
    assert data.target_transform.__str__()==data2.target_transform.__str__()

def test_clone_dataset(simple_dataset):
    data = simple_dataset
    data2 = data.clone()
    assert len(data2)==4
    assert (data.classes==data2.classes).all()
    assert (data.filenames==data2.filenames).all()
    assert (data.targets==data2.targets).all()
    assert data.transform.__str__()==data2.transform.__str__()
    assert data.target_transform.__str__()==data2.target_transform.__str__()
    assert data.root==data2.root
    assert data2.mode=='pandas'
    data2.targets = data2.targets*0
    assert (data.targets != data2.targets).any()
    #print(sys.path)
    #assert False

def test_split(simple_dataset):
    pass

def test_refresh(simple_dataset):
    pass

def test_subset(simple_dataset):
    pass

def test_add(simple_dataset):
    pass