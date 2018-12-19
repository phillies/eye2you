import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
import sklearn.model_selection

# Functions partially copied from torchvision

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset(directory, class_to_idx, extensions):
    images = []
    directory = os.path.expanduser(directory)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def get_image_files(directory, class_index, extensions):
    images = []
    d = os.path.expanduser(directory)
    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, class_index)
                images.append(item)

    return images

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(directory):
    """
    Finds the class folders in a dataset.
    Args:
        directory (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (directiry), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class PandasDataset(torch.utils.data.Dataset):
    """[summary]
    
    Arguments:
        data {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self, source=None, root=None, mode='csv', loader=pil_loader, extensions=IMG_EXTENSIONS, transform=None, target_transform=None):

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
            samples = make_dataset(root, class_to_idx, extensions)
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
            test_size {float} -- If float, should be between 0.0 and 1.0 and represent the proportion 
            of the dataset to include in the test split. If int, represents the absolute number of 
            test samples. If None, the value is set to the complement of the train size. By default, 
            the value is set to 0.1. The default will change in version 0.21. It will remain 0.1 only 
            if train_size is unspecified, otherwise it will complement the specified train_size. (default: {0.1})

            train_size {[type]} -- If float, should be between 0.0 and 1.0 and represent the proportion 
            of the dataset to include in the train split. If int, represents the absolute number of 
            train samples. If None, the value is automatically set to the complement of the test size.
            (default: {None})

            random_state {[type]} -- If int, random_state is the seed used by the random number generator; 
            If RandomState instance, random_state is the random number generator; If None, the random 
            number generator is the RandomState instance used by np.random. (default: {None})
        """

        sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size, random_state=random_state)
        train_index, test_index = next(iter(sss.split(self.filenames, self.targets)))
        train_set = PandasDataset(source=self.samples.iloc[train_index], root=self.root, mode='pandas', 
                                    loader=self.loader, extensions=self.extensions, transform=self.transform, 
                                    target_transform=self.target_transform)
        test_set = PandasDataset(source=self.samples.iloc[test_index], root=self.root, mode='pandas', 
                                    loader=self.loader, extensions=self.extensions, transform=self.transform, 
                                    target_transform=self.target_transform)
        if return_indices is not None and isinstance(return_indices, list):
            return_indices.append(train_index)
            return_indices.append(test_index)
        return train_set, test_set


    def clone(self):
        data = PandasDataset(source=self.samples.copy(deep=True), root=self.root, mode='pandas', 
                            loader=self.loader, extensions=self.extensions, transform=self.transform, 
                            target_transform=self.target_transform)
        return data

    def join(self, other, align_root=False):
        if self.transform != other.transform:
            raise ValueError('transform does not match')
        elif self.target_transform != other.target_transform:
            raise ValueError('target_transform does not match')
        elif self.root != other.root:
            if align_root:
                prefix = os.path.commonprefix([self.root, other.root])
                if prefix == '': #if there is no common prefix
                    self_index_prefix = self.root
                    other_index_prefix = other.root
                else:
                    self_index_prefix = os.path.relpath(self.root, prefix)
                    other_index_prefix = os.path.relpath(other.root, prefix)
                self.samples.index = [self_index_prefix+idx for idx in self.samples.index]
                other = other.clone()
                other.samples.index = [other_index_prefix+idx for idx in other.samples.index]
                self.root = prefix
            else:
                raise ValueError('root does not match')
        self.samples.append(other.samples)
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
        data = PandasDataset(source=self.samples.iloc[indices].copy(deep=True), root=self.root, mode='pandas', 
                            loader=self.loader, extensions=self.extensions, transform=self.transform, 
                            target_transform=self.target_transform)
        return data

    def dump(self, filename):
        self.samples.to_csv(filename, sep=',', index_label=self.samples.index.name)

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
            target = self.targets[index]
        elif isinstance(index, torch.Tensor):
            path = self.filenames[index.item()]
            target = self.targets[index.item()]
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

        dataset = PandasDataset(source=data, root=self.root, mode='pandas', 
                            loader=self.loader, extensions=self.extensions, transform=self.transform, 
                            target_transform=self.target_transform)
        return dataset
