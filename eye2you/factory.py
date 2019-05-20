import ast
import copy
import inspect
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from . import datasets
from . import meter_functions as mf

METER_FUNCTIONS = dict(inspect.getmembers(mf, inspect.isclass))
# TRANSFORM_FUNCTIONS = dict(inspect.getmembers(torchvision.transforms, inspect.isclass))


def _parse_list(config):
    for ii, val in enumerate(config):
        if isinstance(val, dict):
            config[ii] = _parse_dict(val)
        elif isinstance(val, list):
            config[ii] = _parse_list(val)
        elif isinstance(val, tuple):
            config[ii] = _parse_list(list(val))
        elif isinstance(val, (int, float, str, bool)) or val is None:
            pass
        else:
            config[ii] = repr(val)
    return config


def _parse_dict(config):
    for key, val in config.items():
        if isinstance(val, dict):
            config[key] = _parse_dict(val)
        elif isinstance(val, list):
            config[key] = _parse_list(val)
        elif isinstance(val, tuple):
            config[key] = _parse_list(list(val))
        elif isinstance(val, (int, float, str, bool)) or val is None:
            pass
        else:
            config[key] = repr(val)
    return config


def yamlize_config(config):
    config = copy.deepcopy(config)
    config = _parse_dict(config)
    return config


def parse_constructor(param):
    tree = ast.parse(param)
    funccall = tree.body[0].value
    funcname = tree.body[0].value.func.id

    args = [ast.literal_eval(arg) for arg in funccall.args]
    kwargs = {arg.arg: ast.literal_eval(arg.value) for arg in funccall.keywords}
    return funcname, args, kwargs


def get_meter(meter_string):
    meter_name, args, kwargs = parse_constructor(meter_string)
    meter = METER_FUNCTIONS[meter_name](*args, **kwargs)
    return meter


# def get_transform(transforms):
#     if isinstance(transforms, str):
#         transforms = transforms.replace('torchvision.transforms.', '')
#         trans_name, args, kwargs = parse_constructor(transforms)
#         trans = TRANSFORM_FUNCTIONS[trans_name](*args, **kwargs)
#     else:
#         trans = []
#         for t in transforms:
#             t = t.replace('torchvision.transforms.', '')
#             trans_name, args, kwargs = parse_constructor(t)
#             trans.append(TRANSFORM_FUNCTIONS[trans_name](*args, **kwargs))
#         #trans = torchvision.transforms.Compose(trans)
#     return trans


def config_from_yaml(filename):
    with open(str(filename), 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if 'performance_meters' in config['net'] and config['net']['performance_meters'] is not None:
        config['net']['performance_meters'] = [get_meter(m) for m in config['net']['performance_meters']]
    # if 'transform' in config['training'] and config['training']['transform'] is not None:
    #     config['training']['transform'] = get_transform(config['training']['transform'])
    # if 'transform' in config['validation'] and config['validation']['transform'] is not None:
    #     config['validation']['transform'] = get_transform(config['validation']['transform'])

    return config


def load_csv(filename, root, mask_column_name='mask', segmentation_column_name='segmentation',
             target_column_names=None):
    df = pd.read_csv(filename, index_col=0)
    df = df.sort_index()
    samples = np.array([os.path.join(root, v) for v in df.index.values])
    masks = None
    segmentations = None
    if mask_column_name in df:
        masks = np.array([os.path.join(root, v) for v in df[mask_column_name].values])
    if segmentation_column_name in df:
        segmentations = np.array([os.path.join(root, v) for v in df[segmentation_column_name].values])

    cols = df.columns.drop([mask_column_name, segmentation_column_name], errors='ignore')
    if target_column_names is not None:
        cols = [c for c in cols if c in target_column_names]

    if len(cols) == 1 and isinstance(df[cols].iloc[0].values[0], str):
        targets = np.array([os.path.join(root, v[0]) for v in df[cols].values])
    else:
        targets = np.array(df[cols].values)
    target_labels = list(cols)
    return samples, masks, segmentations, targets, target_labels


def data_from_config(config):

    if 'validation' in config:
        if 'columns' in config:
            columns = config['columns']
        else:
            columns = dict()
        train_samples, train_masks, train_segmentations, train_targets, target_labels = load_csv(
            config['csv'], config['root'], **columns)

        if 'columns' in config['validation']:
            columns = config['validation']['columns']
        else:
            columns = dict()
        validation_samples, validation_masks, validation_segmentations, validation_targets, target_labels = load_csv(
            config['validation']['csv'], config['validation']['root'], **columns)

    else:
        if 'columns' in config:
            columns = config['columns']
        else:
            columns = dict()
        samples, masks, segmentations, targets, target_labels = load_csv(config['csv'], config['root'], **columns)
        if 'stratified' in config and config['stratified']:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=config['test_size'])
            train_index, validation_index = next(iter(sss.split(X=samples, y=targets)))
        else:
            ss = ShuffleSplit(n_splits=1, test_size=config['test_size'])
            train_index, validation_index = next(iter(ss.split(X=samples)))

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


def get_loader(config, dataset, step=None):
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

    if step is not None and 'batch_size_increase' in config and config['batch_size_increase'] is not None:
        batch_size = config['batch_size'] + step * config['batch_size_increase']
    else:
        batch_size = config['batch_size']

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
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
