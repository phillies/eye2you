import os
import pathlib

import eye2you
import eye2you.io_helper
import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from eye2you import PandasDataset
from PIL import Image

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
NUMBER_OF_CLASSES = 2
NUMBER_OF_IMAGES = 4

@pytest.fixture(scope='module')
def dataset_simple():
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'), root=(LOCAL_DIR / 'data'))
    return data

@pytest.fixture(scope='module')
def dataset_transforms():
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'), root=(LOCAL_DIR / 'data'),
                        transform=transforms.ToTensor(), target_transform=(lambda x: x+1))
    return data

def test_image_reading():
    img = eye2you.io_helper.pil_loader(LOCAL_DIR / 'data/classA/img0.jpg')
    assert not img is None
    assert isinstance(img, Image.Image)

def test_data_reading():
    path = LOCAL_DIR / 'data'
    classes, class_to_idx = eye2you.io_helper.find_classes(path)
    assert len(classes)==NUMBER_OF_CLASSES
    assert len(class_to_idx)==NUMBER_OF_CLASSES
    for ii in range(NUMBER_OF_CLASSES):
        assert class_to_idx[classes[ii]]==ii
    

    class_to_idx['nonexisting_class'] = 2
    images = eye2you.io_helper.make_dataset(path, class_to_idx, eye2you.io_helper.IMG_EXTENSIONS)
    assert len(images)==NUMBER_OF_IMAGES


def test_modes():
    with pytest.raises(ValueError):
        _ = PandasDataset(mode=None)
    
    dataset = PandasDataset(mode='csv', source=(LOCAL_DIR / 'data.csv'))
    assert not dataset is None
    
    dataset = None
    dataset = PandasDataset(mode='ImageFolder', root=(LOCAL_DIR / 'data'))
    assert not dataset is None

    dataset2 = PandasDataset(mode='pandas', source=dataset.samples)
    assert not dataset2 is None
    assert (dataset.samples==dataset2.samples).all().all()

def test_create_datasets():
    # Create dataset from csv
    print((LOCAL_DIR / 'data.csv'))
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'))
    assert len(data)==NUMBER_OF_IMAGES

    #create data set from pandas object
    data2 = PandasDataset(source=data.samples, mode='pandas')
    assert len(data2)==NUMBER_OF_IMAGES

    # checking that both methods lead to the same data set
    assert (data.classes==data2.classes).all()
    assert (data.filenames==data2.filenames).all()
    assert (data.targets==data2.targets).all()
    assert data.transform.__str__()==data2.transform.__str__()
    assert data.target_transform.__str__()==data2.target_transform.__str__()

def test_clone_dataset(dataset_simple):
    data = dataset_simple
    data2 = data.clone()
    assert len(data2)==NUMBER_OF_IMAGES
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
    
def test_split(dataset_simple):
    data = dataset_simple.clone()
    test_len = 0.5
    train, test = data.split(test_len)
    data_len = len(data)
    assert len(test) == int(data_len*test_len)
    assert len(train) == (data_len - int(data_len*test_len))
    return_indices = []
    train, test = data.split(0.5, return_indices=return_indices)
    assert len(return_indices) == NUMBER_OF_IMAGES/2

def test_refresh(dataset_simple):
    data = dataset_simple.clone()
    data.samples = data.samples.append(data.samples, sort=False)
    assert len(data) != len(data.filenames)
    data.refresh()
    assert len(data.filenames) == len(data)
    assert len(data.targets) == len(data)
    assert len(data.classes) == len(data.samples.columns)
    assert all([a==b for a,b in zip(data.classes, data.samples.columns)])

def test_subset(dataset_simple):
    data = dataset_simple
    indices = np.arange(len(data), dtype=np.int)
    subset_indices = np.random.permutation(indices)[:int(len(data)/2)]
    data_subset = data.subset(subset_indices)
    assert data.transform.__str__()==data_subset.transform.__str__()
    assert data.target_transform.__str__()==data_subset.target_transform.__str__()
    assert data.root==data_subset.root
    for ii in range(len(subset_indices)):
        assert all(data.samples.iloc[subset_indices[ii]] == data_subset.samples.iloc[ii])
    

def test_add(dataset_simple):
    data = dataset_simple.clone()
    data2 = dataset_simple.clone()
    data2 = data2 + data
    assert len(data2) == len(data)*2
    assert (data.classes==data2.classes).all()
    assert data.transform.__str__()==data2.transform.__str__()
    assert data.target_transform.__str__()==data2.target_transform.__str__()
    assert data.root==data2.root

def test_add_mismatches(dataset_transforms):
    data = dataset_transforms.clone()
    data2 = data.clone()
    data2.transform=None
    with pytest.raises(ValueError):
        _ = data + data2

    data2 = data.clone()
    data2.target_transform=None
    with pytest.raises(ValueError):
        _ = data + data2

    data2 = data.clone()
    data2.root='Different Root'
    with pytest.raises(ValueError):
        _ = data + data2

def test_join_identical(dataset_simple):
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'), root=(LOCAL_DIR / 'data'))
    data.join(dataset_simple)
    assert not data is None
    assert len(data)==2*NUMBER_OF_IMAGES

def test_join_mismatches(dataset_transforms):
    data = dataset_transforms.clone()
    data.transform = None
    with pytest.raises(ValueError):
        data.join(dataset_transforms)

    data = dataset_transforms.clone()
    data.target_transform = None
    with pytest.raises(ValueError):
        data.join(dataset_transforms)
    
    data = dataset_transforms.clone()
    data.root = 'DifferentRoot'
    with pytest.raises(ValueError):
        data.join(dataset_transforms)
    
def test_join_align_common_root():
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'), root='SameSame/ButDifferent')
    data2 = PandasDataset(source=(LOCAL_DIR / 'data.csv'), root='SameSame')
    data.join(data2, align_root=True)
    assert len(data)==2*NUMBER_OF_IMAGES
    assert data.root=='SameSame'
    for ii in range(NUMBER_OF_IMAGES):
        assert data.samples.index[ii].startswith('SameSame')
    for ii in range(NUMBER_OF_IMAGES,2*NUMBER_OF_IMAGES):
        assert data.samples.index[ii].startswith('SameSame')
        assert not data.samples.index[ii].startswith('SameSame/ButDifferent')

def test_join_align_no_common_root():
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'), root='SameSame')
    data2 = PandasDataset(source=(LOCAL_DIR / 'data.csv'), root='ButDifferent')
    data.join(data2, align_root=True)
    assert len(data)==2*NUMBER_OF_IMAGES
    assert data.root==''
    for ii in range(NUMBER_OF_IMAGES):
        assert data.samples.index[ii].startswith('SameSame')
    for ii in range(NUMBER_OF_IMAGES,2*NUMBER_OF_IMAGES):
        assert data.samples.index[ii].startswith('ButDifferent') 
    

def test_append_csv():
    data = PandasDataset(source=(LOCAL_DIR / 'data.csv'))
    assert not data is None
    data.samples.iloc[0,0] = np.nan
    assert data.samples.isna().any().any()

    data.append_csv(source=(LOCAL_DIR / 'data.csv'), nan_replace=123)
    assert not data is None
    assert not data.samples.isna().any().any()
    assert data.samples.iloc[0,0]==123
    assert len(data)==2*NUMBER_OF_IMAGES

    data.append_csv(source=(LOCAL_DIR / 'data.csv'), root=(LOCAL_DIR / 'data'))
    assert not data is None
    assert len(data)==3*NUMBER_OF_IMAGES
    for ii in range(2*NUMBER_OF_IMAGES):
        assert not data.samples.index[ii].startswith(str((LOCAL_DIR / 'data')))
    for ii in range(2*NUMBER_OF_IMAGES,3*NUMBER_OF_IMAGES):
        assert data.samples.index[ii].startswith(str((LOCAL_DIR / 'data')))


def test_printing(dataset_simple):
    data_description = repr(dataset_simple)
    assert data_description.count('PandasDataset') > 0

def test_indexing(dataset_simple, dataset_transforms):
    sample, target = dataset_simple[0]
    assert not sample is None
    assert isinstance(sample, Image.Image)
    assert not target is None
    assert target.dtype==np.float32

    sample2, target2 = dataset_transforms[dataset_transforms.filenames[0]]
    assert isinstance(sample2, torch.Tensor)
    assert (target2==(target+1)).all()

    sample, target = dataset_simple[torch.Tensor([0])]
    assert not sample is None
    assert isinstance(sample, Image.Image)
    assert not target is None
    assert target.dtype==np.float32

def test_index_errors(dataset_simple):
    with pytest.raises(IndexError):
        _ = dataset_simple[1.0]
    with pytest.raises(IndexError):
        _ = dataset_simple['abc']

def test_data_dump(dataset_simple, tmp_path):
    dataset_simple.dump(tmp_path / 'test.csv')
    assert os.path.isfile(tmp_path / 'test.csv')
    data = PandasDataset(source=(tmp_path / 'test.csv'))
    assert not data is None