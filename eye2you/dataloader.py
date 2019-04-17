import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import pandas as pd
from .datasets import SegmentationDatasetWithSampler, SegmentationDataset
from PIL import Image


def get_equal_sampler(dataset, num_samples, relevant_slice=None):
    '''The distribution of samples in training and test data is not equal, i.e.
    the normal class is over represented. To get an unbiased sample (for example with 5
    classes each class should make up for about 20% of the samples) we calculate the
    probability of each class in the source data (=weights) then we invert the weights
    and assign to each sample in the class the inverted probability (normalized to 1 over
    all samples) and use this as the probability for the weighted random sampler.

    Arguments:
        dataset {torch.util.data.Dataset} -- the dataset to sample from
        num_samples {int} -- number of samples to be drawn bei the sampler
        relevant_slices {tuple} -- tuple of dimensions on which the distribution should
        be caculated, e.g. only on dimensions (0,1) of a 3 dimensional set.

    Returns:
        torch.util.data.Sampler -- Sampler object from which patches for the training
        or evaluation can be drawn
    '''
    if relevant_slice is None:
        relevant_slice = range(dataset.targets.shape[1])

    class_distribution = np.vstack([s for s in dataset.targets]).astype(np.int)[:, relevant_slice]
    weights = np.zeros(len(relevant_slice))
    for ii in range(len(relevant_slice)):
        weights[ii] = np.sum(class_distribution[:, ii] == 1) / len(class_distribution)

    inverted_weights = (1 / weights) / np.sum(1 / weights)
    sampling_weights = np.zeros(class_distribution.shape[0], dtype=np.float)
    for ii in range(len(relevant_slice)):
        sampling_weights[class_distribution[:, ii] == 1] = inverted_weights[ii]
    sampling_weights /= sampling_weights.sum()

    sampler = torch.utils.data.WeightedRandomSampler(sampling_weights, num_samples, True)
    return sampler


def drive_loader(root, img_size):
    folder = root + 'images/DRIVE/training/images/'
    train_img_files = sorted([os.path.join(folder, d) for d in os.listdir(folder)])
    label_folder = root + 'images/DRIVE/training/1st_manual/'
    train_label_files = sorted([os.path.join(label_folder, d) for d in os.listdir(label_folder)])
    folder = root + 'images/DRIVE/training/mask/'
    train_mask_files = sorted([os.path.join(folder, d) for d in os.listdir(folder)])

    test_folder = root + 'images/DRIVE/test/images/'
    test_img_files = sorted([os.path.join(test_folder, d) for d in os.listdir(test_folder)])
    folder = root + 'images/DRIVE/test/1st_manual/'
    test_label_files = sorted([os.path.join(folder, d) for d in os.listdir(folder)])
    folder = root + 'images/DRIVE/test/mask/'
    test_mask_files = sorted([os.path.join(folder, d) for d in os.listdir(folder)])

    #img_size=128
    train_data = SegmentationDatasetWithSampler(
        img_size, ratio=(1, 1), scale=(0.25, 1), normalize=True).from_lists(train_img_files, train_label_files,
                                                                            train_mask_files).preload()

    test_data = SegmentationDataset().from_lists(test_img_files, test_label_files, test_mask_files).preload()
    test_data.target_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(img_size, interpolation=Image.NEAREST),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor()
    ])
    test_data.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size, interpolation=Image.BILINEAR),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=train_data.mean, std=train_data.std)
    ])

    return train_data, test_data