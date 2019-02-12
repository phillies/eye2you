import pytest
import numpy as np
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

    #create data set from pandas object
    data2 = PandasDataset(source=data.samples, mode='pandas')
    assert len(data2)==4

    # checking that both methods lead to the same data set
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

    # testing that data and data2 are deep copies and not just referenced copies
    data2.targets = data2.targets*0
    assert (data.targets != data2.targets).any()
    
def test_split(simple_dataset):
    data = simple_dataset.clone()
    test_len = 0.5
    train, test = data.split(test_len)
    data_len = len(data)
    assert len(test) == int(data_len*test_len)
    assert len(train) == (data_len - int(data_len*test_len))
    return_indices = []
    train, test = data.split(0.5, return_indices=return_indices)
    assert len(return_indices) == 2

def test_refresh(simple_dataset):
    data = simple_dataset.clone()
    data.samples = data.samples.append(data.samples, sort=False)
    assert len(data) != len(data.filenames)
    data.refresh()
    assert len(data.filenames) == len(data)
    assert len(data.targets) == len(data)
    assert len(data.classes) == len(data.samples.columns)
    assert all([a==b for a,b in zip(data.classes, data.samples.columns)])

def test_subset(simple_dataset):
    data = simple_dataset
    indices = np.arange(len(data), dtype=np.int)
    subset_indices = np.random.permutation(indices)[:int(len(data)/2)]
    data_subset = data.subset(subset_indices)
    assert data.transform.__str__()==data_subset.transform.__str__()
    assert data.target_transform.__str__()==data_subset.target_transform.__str__()
    assert data.root==data_subset.root
    for ii in range(len(subset_indices)):
        assert all(data.samples.iloc[subset_indices[ii]] == data_subset.samples.iloc[ii])
    

def test_add(simple_dataset):
    data = simple_dataset
    data2 = simple_dataset.clone()
    data2 = data2 + data
    assert len(data2) == len(data)*2
    assert (data.classes==data2.classes).all()
    assert data.transform.__str__()==data2.transform.__str__()
    assert data.target_transform.__str__()==data2.target_transform.__str__()
    assert data.root==data2.root