import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from . import datasets
from . import meter_functions as mf


def config_from_yaml(filename):
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # TODO: find better solution than eval()
    if 'performance_meters' in config['net'] and config['net']['performance_meters'] is not None:
        config['net']['performance_meters'] = [eval(m) for m in config['net']['performance_meters']]
    if 'transform' in config['training'] and config['training']['transform'] is not None:
        config['training']['transform'] = eval(config['training']['transform'])
    if 'target_transform' in config['training'] and config['training']['target_transform'] is not None:
        config['training']['target_transform'] = eval(config['training']['target_transform'])
    if 'transform' in config['validation'] and config['validation']['transform'] is not None:
        config['validation']['transform'] = eval(config['validation']['transform'])
    if 'target_transform' in config['training'] and config['validation']['target_transform'] is not None:
        config['validation']['target_transform'] = eval(config['validation']['target_transform'])

    return config


def load_csv(filename, root, mask_column_name='mask', segmentation_column_name='segmentation'):
    df = pd.read_csv(filename, index_col=0)
    samples = np.array([os.path.join(root, v) for v in df.index.values])
    masks = None
    segmentations = None
    if mask_column_name in df:
        masks = np.array([os.path.join(root, v) for v in df[mask_column_name].values])
    if segmentation_column_name in df:
        segmentations = np.array([os.path.join(root, v) for v in df[segmentation_column_name].values])

    cols = df.columns.drop([mask_column_name, segmentation_column_name], errors='ignore')
    if cols.size == 1 and isinstance(df[cols].iloc[0], str):
        targets = np.array([os.path.join(root, v) for v in df[cols].values])
        target_labels = cols
    else:
        targets = np.array(df[cols].values)
        target_labels = cols
    return samples, masks, segmentations, targets, target_labels


def data_from_config(config):
    if 'validation' in config:
        train_samples, train_masks, train_segmentations, train_targets, target_labels = load_csv(
            config['csv'], config['root'])
        validation_samples, validation_masks, validation_segmentations, validation_targets, target_labels = load_csv(
            config['validation']['csv'], config['validation']['root'])

    else:
        samples, masks, segmentations, targets, target_labels = load_csv(config['csv'], config['root'])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=config['test_size'])
        train_index, validation_index = next(iter(sss.split(X=samples, y=targets)))

        train_samples = samples[train_index]
        validation_samples = samples[validation_index]

        if masks is None:
            train_masks = None
            validation_masks = None
        else:
            train_masks = masks[train_index]
            validation_masks = masks[validation_index]

        if segmentations is None:
            train_segmentations = None
            validation_segmentations = None
        else:
            train_segmentations = segmentations[train_index]
            validation_segmentations = segmentations[validation_index]

        if targets.ndim == 2:
            train_targets = targets[train_index, :]
            validation_targets = targets[validation_index, :]
        else:
            train_targets = targets[train_index]
            validation_targets = targets[validation_index]

    training_data = datasets.TripleDataset(samples=train_samples,
                                           masks=train_masks,
                                           segmentations=train_segmentations,
                                           targets=train_targets,
                                           target_labels=target_labels)
    validation_data = datasets.TripleDataset(samples=validation_samples,
                                             masks=validation_masks,
                                             segmentations=validation_segmentations,
                                             targets=validation_targets,
                                             target_labels=target_labels)

    return training_data, validation_data


def get_loader(config, dataset):
    if 'drop_last' not in config or config['drop_last'] is None:
        drop_last = False
    else:
        drop_last = config['drop_last']

    if 'replacement' not in config or config['replacement'] is None:
        replacement = False
    else:
        replacement = config['replacement']

    if 'num_samples' not in config or config['num_samples'] is None:
        if replacement:
            num_samples = len(dataset)
        else:
            num_samples = None
    else:
        num_samples = config['num_samples']

    if 'weighted_sampling_classes' in config:
        sampler = get_equal_sampler(dataset,
                                    num_samples=num_samples,
                                    relevant_slice=config['weighted_sampling_classes'])
    else:
        sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement, num_samples=num_samples)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=config['batch_size'],
                                         shuffle=config['shuffle'],
                                         drop_last=drop_last,
                                         num_workers=config['num_workers'],
                                         sampler=sampler)
    return loader


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
