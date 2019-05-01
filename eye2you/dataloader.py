import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import pandas as pd
from .datasets import SegmentationDatasetWithSampler, SegmentationDataset
from PIL import Image


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
    train_data = SegmentationDatasetWithSampler(img_size, ratio=(1, 1), scale=(0.25, 1),
                                                normalize=True).from_lists(train_img_files, train_label_files,
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