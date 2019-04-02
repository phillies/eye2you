import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
import sklearn.model_selection
from .io_helper import pil_loader, IMG_EXTENSIONS, find_classes, make_dataset


class PandasDataset(torch.utils.data.Dataset):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self,
                 source=None,
                 root=None,
                 mode='csv',
                 loader=pil_loader,
                 extensions=IMG_EXTENSIONS,
                 transform=None,
                 target_transform=None):

        #TODO: ImageFolderPlus mode loads like ImageFolder plus files in the root directory with class NaN
        #TODO: Video mode loads all video files (don't forget extensions!) and assigns the classes to frames

        self.csv_file = None

        if mode == 'csv':
            samples = pd.read_csv(source, index_col=0)
            if not root is None:
                name = samples.index.name
                samples.index = [str(Path(root) / Path(ind)) for ind in samples.index]
                samples.index.name = name
            classes = samples.columns
            class_to_idx = dict(enumerate(classes))
            self.csv_file = source
        elif mode == 'ImageFolder':
            classes, class_to_idx = find_classes(root)
            images = make_dataset(root, class_to_idx, extensions)
            samples = pd.DataFrame.from_dict({item[0]: item[1] for item in images},
                                             orient='index',
                                             columns=['class_id'])
            samples.index.name = 'filename'
        elif mode == 'pandas':
            samples = source
            classes = samples.columns
            class_to_idx = dict(enumerate(classes))
        else:
            raise ValueError('Other modes not implemented yet')

        self.root = root
        self.mode = mode
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = samples.iloc[:].values.astype(np.float32)
        self.imgs = samples
        self.filenames = samples.index.values

        self.transform = transform
        self.target_transform = target_transform

    def split(self, test_size=0.1, train_size=None, random_state=None, return_indices=None):
        """Splitting the Dataset into test and training set. Uses scikit StratifiedShuffleSplit
        to keep the distribution of classes equal in test and training. Parameters are identical
        (as of scikit 0.20).
        
        Keyword Arguments:
            test_size {float/int} -- If float, should be between 0.0 and 1.0 and represent the proportion 
            of the dataset to include in the test split. If int, represents the absolute number of 
            test samples. If None, the value is set to the complement of the train size. By default, 
            the value is set to 0.1. The default will change in version 0.21. It will remain 0.1 only 
            if train_size is unspecified, otherwise it will complement the specified train_size. (default: {0.1})

            train_size {float/int} -- If float, should be between 0.0 and 1.0 and represent the proportion 
            of the dataset to include in the train split. If int, represents the absolute number of 
            train samples. If None, the value is automatically set to the complement of the test size.
            (default: {None})

            random_state {int/RandomState} -- If int, random_state is the seed used by the random number generator; 
            If RandomState instance, random_state is the random number generator; If None, the random 
            number generator is the RandomState instance used by np.random. (default: {None})

            return_indices {list} -- List where the indices of the split will be appended. (default: {None})
        """

        sss = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, train_size=train_size, random_state=random_state)
        train_index, test_index = next(iter(sss.split(self.filenames, self.targets)))
        train_set = PandasDataset(
            source=self.samples.iloc[train_index],
            root=self.root,
            mode='pandas',
            loader=self.loader,
            extensions=self.extensions,
            transform=self.transform,
            target_transform=self.target_transform)
        test_set = PandasDataset(
            source=self.samples.iloc[test_index],
            root=self.root,
            mode='pandas',
            loader=self.loader,
            extensions=self.extensions,
            transform=self.transform,
            target_transform=self.target_transform)
        if return_indices is not None and isinstance(return_indices, list):
            return_indices.append(train_index)
            return_indices.append(test_index)
        return train_set, test_set

    def clone(self):
        data = PandasDataset(
            source=self.samples.copy(deep=True),
            root=self.root,
            mode='pandas',
            loader=self.loader,
            extensions=self.extensions,
            transform=self.transform,
            target_transform=self.target_transform)
        return data

    def join(self, other, align_root=False):
        if self.transform != other.transform:
            raise ValueError('transform does not match')
        elif self.target_transform != other.target_transform:
            raise ValueError('target_transform does not match')
        elif self.root != other.root:
            if align_root:
                prefix = os.path.commonpath([self.root, other.root])
                if prefix == '':  #if there is no common prefix
                    self_index_prefix = self.root
                    other_index_prefix = other.root
                else:
                    self_index_prefix = os.path.relpath(self.root, prefix)
                    other_index_prefix = os.path.relpath(other.root, prefix)
                self.samples.index = [
                    os.path.join(
                        prefix,  # new root
                        self_index_prefix,  # relative path between old root and new root
                        os.path.relpath(idx, self.root)) for idx in self.samples.index
                ]  # base without old root

                other = other.clone()
                other.samples.index = [
                    os.path.join(prefix, other_index_prefix, os.path.relpath(idx, other.root))
                    for idx in other.samples.index
                ]
                self.root = prefix
            else:
                raise ValueError('root does not match')
        self.samples = self.samples.append(other.samples)
        self.refresh()

    def append_csv(self, source, root=None, nan_replace=None):
        samples = pd.read_csv(source, index_col=0)
        if not root is None:
            name = samples.index.name
            samples.index = [str(Path(root) / Path(ind)) for ind in samples.index]
            samples.index.name = name

        self.samples = self.samples.append(samples, sort=False)

        if nan_replace is not None:
            self.samples[self.samples.isna()] = nan_replace

        self.refresh()

    def refresh(self):
        classes = self.samples.columns
        class_to_idx = dict(enumerate(classes))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = self.samples.iloc[:].values.astype(np.float32)
        self.filenames = self.samples.index.values

    def subset(self, indices):
        data = PandasDataset(
            source=self.samples.iloc[indices].copy(deep=True),
            root=self.root,
            mode='pandas',
            loader=self.loader,
            extensions=self.extensions,
            transform=self.transform,
            target_transform=self.target_transform)
        return data

    def dump(self, filename):
        samples_dump = self.samples.copy(deep=True)
        samples_dump.index = [os.path.relpath(idx, self.root).replace('\\', '/') for idx in samples_dump.index]
        samples_dump.to_csv(filename, sep=',', index_label=self.samples.index.name)

    def __getitem__(self, index):
        """
        Args:
            index (int or str): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if isinstance(index, int):
            path = self.filenames[index]
            target = self.targets[index]
        elif isinstance(index, str):
            if not index in self.filenames:
                raise IndexError('Given index', index, 'not in sample index', self.filenames)
            path = index
            target = self.samples.loc[index].values.astype(np.float32)
        elif isinstance(index, torch.Tensor):
            path = self.filenames[int(index.item())]
            target = self.targets[int(index.item())]
        else:
            raise IndexError('Given index', index, 'not an integer nor in sample index', self.filenames)

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __add__(self, other):
        if self.transform != other.transform:
            raise ValueError('transform does not match')
        elif self.target_transform != other.target_transform:
            raise ValueError('target_transform does not match')
        elif self.root != other.root:
            raise ValueError('root does not match')
        data = self.samples.copy(deep=True)
        data = data.append(other.samples)

        dataset = PandasDataset(
            source=data,
            root=self.root,
            mode='pandas',
            loader=self.loader,
            extensions=self.extensions,
            transform=self.transform,
            target_transform=self.target_transform)
        return dataset


class SegmentationDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        self.samples = None
        self.samples_data = None
        self.targets = None
        self.targets_data = None
        self.masks = None
        self.masks_data = None
        self.transform = lambda x: x
        self.target_transform = lambda x: x
        self.loaded = False

    def __getitem__(self, index):
        if index >= self.__len__():
            raise ValueError('Index {} out of bounds for dataset with length {}'.format(index, self.__len__()))
        sample = self.transform(self.samples_data[index])
        target = self.target_transform(self.targets_data[index])
        return sample, target

    def __len__(self):
        return len(self.samples)

    def from_lists(self, samples, targets, masks=None):
        if len(samples) != len(targets):
            raise ValueError('Samples and targets must have same length.')
        if masks is not None and len(masks) != len(targets):
            raise ValueError('Masks and targets must have same length.')
        self.samples = samples
        self.samples_data = [None] * len(samples)
        self.targets = targets
        self.targets_data = [None] * len(targets)
        self.masks = masks
        if masks is not None:
            self.masks_data = [None] * len(masks)
        return self

    def from_DataFrame(self, data):
        if 'samples' not in data and 'targets' not in data:
            raise ValueError('Samples and targets columns must be in DataFrame.')
        self.samples = data['samples']
        self.samples_data = [None] * len(self.samples)
        self.targets = data['targets']
        self.targets_data = [None] * len(self.targets)
        if 'masks' in data:
            self.masks = data['masks']
            self.masks_data = [None] * len(self.masks)
        return self

    def preload(self):
        for (ii, sample) in enumerate(self.samples):
            self.samples_data[ii] = Image.open(sample)
        for (ii, target) in enumerate(self.targets):
            self.targets_data[ii] = Image.open(target)
        if self.masks is not None:
            for (ii, mask) in enumerate(self.masks):
                self.mask_data[ii] = Image.open(mask)
        self.loaded = True
        return self
