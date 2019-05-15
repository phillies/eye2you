# pylint: disable=redefined-outer-name
import os
import pathlib

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image

from eye2you.datasets import DataAugmentation, DataPreparation, TripleDataset

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
NUMBER_OF_CLASSES = 2
NUMBER_OF_IMAGES = 4

#TODO: Create an image with only red pixels to test the loading of a single color band as targets


def test_triple_dataset_setup_empty():
    data = TripleDataset()
    assert data is not None
    assert len(data) == 0


def test_triple_dataset_setup_target_RGB_image(image_set):
    aug = DataAugmentation(angle=23.5,
                           size=100,
                           scale=(0.5, 1.45),
                           ratio=(0.75, 1.3),
                           brightness=0.5,
                           contrast=0.6,
                           saturation=0.4,
                           hue=0.1,
                           hflip=0.25,
                           vflip=0.33)
    prep = DataPreparation(size=100, mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05), crop=90)
    files, masks = image_set

    data = TripleDataset(samples=files,
                         segmentations=masks,
                         masks=masks,
                         targets=files,
                         target_labels=['test label'],
                         augmentation=aug,
                         preparation=prep)

    assert data is not None
    assert len(data) == 4
    source, target = data[0]
    assert isinstance(source, (list, tuple))
    assert len(source) == 3
    for s in source:
        assert isinstance(s, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (4, 90, 90)


def test_triple_dataset_index_error(image_set):
    data = TripleDataset()
    with pytest.raises(IndexError):
        _ = data[0]
    with pytest.raises(IndexError):
        _ = data[-1]

    files, masks = image_set
    data = TripleDataset(samples=files, targets=masks, target_labels=['test label'])
    with pytest.raises(IndexError):
        _ = data[len(data)]


def test_triple_dataset_setup_target_gray_image(image_set):
    aug = DataAugmentation(angle=23.5,
                           size=100,
                           scale=(0.5, 1.45),
                           ratio=(0.75, 1.3),
                           brightness=0.5,
                           contrast=0.6,
                           saturation=0.4,
                           hue=0.1,
                           hflip=0.25,
                           vflip=0.33)
    prep = DataPreparation(size=100, mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05), crop=90)
    files, masks = image_set

    data = TripleDataset(samples=files,
                         segmentations=masks,
                         masks=masks,
                         targets=masks,
                         target_labels=['test label'],
                         loader=Image.open,
                         augmentation=aug,
                         preparation=prep)

    assert data is not None
    assert len(data) == 4
    assert data.size == 4
    source, target = data[0]
    assert isinstance(source, (list, tuple))
    assert len(source) == 3
    assert source[0].shape == (3, 90, 90)
    assert source[1].shape == (1, 90, 90)
    assert source[2].shape == (1, 90, 90)
    for s in source:
        assert isinstance(s, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (1, 90, 90)


def test_triple_dataset_setup_target_labels(image_set):
    aug = DataAugmentation(angle=23.5,
                           size=100,
                           scale=(0.5, 1.45),
                           ratio=(0.75, 1.3),
                           brightness=0.5,
                           contrast=0.6,
                           saturation=0.4,
                           hue=0.1,
                           hflip=0.25,
                           vflip=0.33)
    prep = DataPreparation(size=100, mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05), crop=90)
    files, masks = image_set

    data = TripleDataset(samples=files,
                         segmentations=masks,
                         masks=masks,
                         targets=torch.randn((len(files), 3)),
                         target_labels=['class 1', 'class 2', 'class 3'],
                         loader=Image.open,
                         augmentation=aug,
                         preparation=prep)

    assert data is not None
    assert len(data) == 4
    assert data.size == 4
    source, target = data[0]
    assert isinstance(source, (list, tuple))
    assert len(source) == 3
    assert source[0].shape == (3, 90, 90)
    assert source[1].shape == (1, 90, 90)
    assert source[2].shape == (1, 90, 90)
    for s in source:
        assert isinstance(s, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (3,)


def test_triple_dataset_get_item_without_mask_segmentation(image_set):
    files, _ = image_set
    prep = DataPreparation(size=100, mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05), crop=90)
    data = TripleDataset(samples=files,
                         segmentations=None,
                         masks=None,
                         targets=torch.randn((len(files), 3)),
                         target_labels=['class 1', 'class 2', 'class 3'],
                         preparation=prep,
                         loader=Image.open)

    assert data is not None
    assert len(data) == 4
    assert data.size == 4
    source, target = data[0]
    assert isinstance(source, (list, tuple))
    assert len(source) == 3
    assert source[0].shape == (3, 90, 90)
    assert source[1].shape == (1, 90, 90)
    assert source[2].shape == (1, 90, 90)
    for s in source:
        assert isinstance(s, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (3,)


def test_triple_dataset_print():
    data = TripleDataset()
    output = data.__str__()
    assert 'Dataset' in output
    assert 'Samples' in output
    assert 'Masks' in output
    assert 'Segmentation' in output
    assert 'Targets' in output
    assert 'Target labels' in output
    assert 'classes' in output
