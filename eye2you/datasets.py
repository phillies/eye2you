import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

from .image_helper import parallel_variance
from .io_helper import IMG_EXTENSIONS, find_classes, make_dataset, pil_loader


class TripleDataset(torch.utils.data.Dataset):

    def __init__(self,
                 samples=None,
                 segmentations=None,
                 masks=None,
                 targets=None,
                 target_labels=None,
                 rotation=None,
                 random_resize_crop=None,
                 color_jitter=None,
                 transform=transforms.ToTensor(),
                 target_transform=None,
                 mean=(0.0, 0.0, 0.0),
                 std=(1.0, 1.0, 1.0),
                 loader=pil_loader,
                 **kwargs):

        super().__init__(**kwargs)
        self.samples = samples
        self.segmentations = segmentations
        self.masks = masks
        self.targets = targets
        self.target_labels = target_labels

        self.rotation = rotation
        self.random_resize_crop = random_resize_crop
        self.color_jitter = color_jitter
        self.transform = transform
        self.target_transform = target_transform

        self.mean = mean
        self.std = std

        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise IndexError('Index {0} our of bounds for dataset of length {1}'.format(index, len(self)))

        # get all frames for index
        sample = self.loader(self.samples[index])
        if self.masks is None:
            mask = Image.fromarray(np.ones((sample.height, sample.width), dtype=np.uint8) * 255)
        else:
            mask = self.loader(self.masks[index]).convert('L')
        if self.segmentations is None:
            segment = Image.fromarray(np.zeros((sample.height, sample.width), dtype=np.uint8))
        else:
            segment = self.loader(self.segmentations[index]).convert('L')
        target = self.targets[index]

        # special treatment if target is class labels vs. filename
        if isinstance(target, str):
            target = self.loader(target).convert('L')
            target_is_image = True
        else:
            target_is_image = False

        if self.color_jitter is not None:
            trans = self.color_jitter.get_params(self.color_jitter.brightness, self.color_jitter.saturation, self.color_jitter.contrast, self.color_jitter.hue)
            sample = trans(sample)

        # apply rotation
        if self.rotation is not None:
            angle = self.rotation.get_params(self.rotation.degrees)
            sample = F.rotate(sample, angle, Image.BILINEAR, self.rotation.expand, self.rotation.center)
            mask = F.rotate(mask, angle, Image.NEAREST, self.rotation.expand, self.rotation.center)
            segment = F.rotate(segment, angle, Image.NEAREST, self.rotation.expand, self.rotation.center)
            if target_is_image:
                target = F.rotate(target, angle, Image.NEAREST, self.rotation.expand, self.rotation.center)

        # apply RRC
        if self.random_resize_crop is not None:
            i, j, h, w = self.random_resize_crop.get_params(sample, self.random_resize_crop.scale,
                                                            self.random_resize_crop.ratio)
            sample = F.resized_crop(sample, i, j, h, w, self.random_resize_crop.size, Image.BILINEAR)
            mask = F.resized_crop(mask, i, j, h, w, self.random_resize_crop.size, Image.NEAREST)
            segment = F.resized_crop(segment, i, j, h, w, self.random_resize_crop.size, Image.NEAREST)
            if target_is_image:
                target = F.resized_crop(target, i, j, h, w, self.random_resize_crop.size, Image.NEAREST)

        # apply (target) transform
        if self.transform is not None:
            sample = self.transform(sample)
            mask = self.transform(mask)
            segment = self.transform(segment)
            if target_is_image:
                target = self.transform(target)

        if self.target_transform is not None and not target_is_image:
            target = self.target_transform(target)

        # apply normalization
        transforms.functional.normalize(sample, self.mean, self.std, inplace=True)

        return (sample, mask, segment), target

    def read_csv(self, filename, root=''):
        df = pd.read_csv(filename, index_col=0)
        self.samples = [os.path.join(root, v) for v in df.index.values]
        if 'mask' in df:
            self.masks = [os.path.join(root, v) for v in df['mask'].values]
        if 'segmentation' in df:
            self.segmentations = [os.path.join(root, v) for v in df['segmentation'].values]

        cols = df.columns.drop(['mask', 'segmentation'], errors='ignore')
        if cols.size == 1 and isinstance(df[cols].iloc[0], str):
            self.targets = [os.path.join(root, v) for v in df[cols].values]
            self.target_labels = cols
        else:
            self.targets = df[cols].values
            self.target_labels = cols
        return self

    def to_csv(self, filename):
        #TODO: implement export
        pass
    
    def __str__(self):
        res = ''
        res += str(self.samples) + '\n'
        res += str(self.masks) + '\n'
        res += str(self.segmentations) + '\n'
        res += str(self.targets) + '\n'
        return res

    @property
    def size(self):
        return len(self)

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

        sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,
                                                             test_size=test_size,
                                                             train_size=train_size,
                                                             random_state=random_state)
        train_index, test_index = next(iter(sss.split(self.filenames, self.targets)))
        train_set = PandasDataset(source=self.samples.iloc[train_index],
                                  root=self.root,
                                  mode='pandas',
                                  loader=self.loader,
                                  extensions=self.extensions,
                                  transform=self.transform,
                                  target_transform=self.target_transform)
        test_set = PandasDataset(source=self.samples.iloc[test_index],
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
        data = PandasDataset(source=self.samples.copy(deep=True),
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
        data = PandasDataset(source=self.samples.iloc[indices].copy(deep=True),
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

        dataset = PandasDataset(source=data,
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
        self.transform = torchvision.transforms.ToTensor()
        self.target_transform = torchvision.transforms.ToTensor()
        self.mean = None
        self.std = None
        self.loaded = False
        self.normalize = False

    def __getitem__(self, index):
        if index >= self.__len__():
            raise ValueError('Index {} out of bounds for dataset with length {}'.format(index, self.__len__()))

        if self.loaded:
            sample = self.transform(self.samples_data[index])
            target = self.target_transform(self.targets_data[index])
        else:
            sample = self.transform(Image.open(self.samples[index]))
            target = self.target_transform(Image.open(self.targets[index]))
        return sample, target

    def get_item(self, index):
        sample = self.get_sample(index)
        target = self.get_target(index)
        return sample, target

    def get_sample(self, index):
        if index >= self.__len__():
            raise ValueError('Index {} out of bounds for dataset with length {}'.format(index, self.__len__()))

        if self.loaded:
            sample = self.transform(self.samples_data[index])
        else:
            sample = self.transform(Image.open(self.samples[index]))
        return sample

    def get_target(self, index):
        if index >= self.__len__():
            raise ValueError('Index {} out of bounds for dataset with length {}'.format(index, self.__len__()))

        if self.loaded:
            target = self.target_transform(self.targets_data[index])
        else:
            target = self.target_transform(Image.open(self.targets[index]))
        return target

    def get_mask(self, index):
        if index >= self.__len__():
            raise ValueError('Index {} out of bounds for dataset with length {}'.format(index, self.__len__()))

        if self.masks is None:
            mask = torch.ones_like(self.get_sample(index))
        elif self.loaded:
            mask = self.target_transform(self.masks_data[index])
        else:
            mask = self.target_transform(Image.open(self.masks[index]))
        return mask

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
                self.masks_data[ii] = Image.open(mask)
        self.loaded = True
        self.mean_and_std()
        return self

    def mean_and_std(self):
        if self.loaded:
            x = torchvision.transforms.ToTensor()(self.samples_data[0])
            if self.masks is not None:
                mask = torchvision.transforms.ToTensor()(self.masks_data[0])
                x = x[:, (mask == 1)[0, :, :]]
            else:
                x = x.reshape(x.shape[0], -1)
            mean_x = x.mean(1)
            var_x = x.var(1)
            count_x = x.shape[1]
            for ii in range(1, self.__len__()):
                x_b = torchvision.transforms.ToTensor()(self.samples_data[ii])
                if self.masks is not None:
                    mask = torchvision.transforms.ToTensor()(self.masks_data[0])
                    x_b = x_b[:, (mask == 1)[0, :, :]]
                else:
                    x_b = x_b.reshape(x_b.shape[0], -1)
                mean_x_b = x_b.mean(1)
                var_x_b = x_b.var(1)
                count_x_b = x_b.shape[1]

                var_x = parallel_variance(mean_x, count_x, var_x, mean_x_b, count_x_b, var_x_b)
                mean_x = (mean_x * count_x + mean_x_b * count_x_b) / (count_x + count_x_b)
                count_x += count_x_b
        else:
            raise NotImplementedError('Sorry, not yet implemented. Please preload dataset.')
        self.mean = mean_x.numpy()
        self.std = np.sqrt(var_x).numpy()
        return self.mean, self.std

    @property
    def trans(self):
        if self.normalize:
            return torchvision.transforms.Compose([self.transform, self.normalize_transform])
        else:
            return self.transform

    @trans.setter
    def trans(self, transform):
        self.transform = transform

    def normalize_as(self, dataset):
        self.normalize = True
        self.mean = dataset.mean
        self.std = dataset.std
        self.normalize_transform = torchvision.transforms.Normalize(mean=self.mean, std=self.std)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SegmentationDatasetWithSampler(SegmentationDataset):
    '''Segmentation dataset with built-in RandomResizedCrop transformation so that the
    source images and the segmentation targets are sampled in the same manner.
    '''

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR,
                 normalize=False):
        super().__init__()
        self.update_transform(size=size, scale=scale, ratio=ratio, interpolation=interpolation, normalize=normalize)
        self.normalize_transform = None

    def update_transform(self,
                         size,
                         scale=(0.08, 1.0),
                         ratio=(3. / 4., 4. / 3.),
                         interpolation=Image.BILINEAR,
                         normalize=False):
        self.random_resize_crop = torchvision.transforms.RandomResizedCrop(size=size,
                                                                           scale=scale,
                                                                           ratio=ratio,
                                                                           interpolation=interpolation)
        self.rotation = torchvision.transforms.RandomRotation(180)
        self.normalize = normalize

    def __getitem__(self, index):
        if index >= self.__len__():
            raise ValueError('Index {} out of bounds for dataset with length {}'.format(index, self.__len__()))

        if self.loaded:
            sample = self.samples_data[index]
            target = self.targets_data[index]
        else:
            sample = Image.open(self.samples[index])
            target = Image.open(self.targets[index])

        angle = self.rotation.get_params(self.rotation.degrees)
        sample = F.rotate(sample, angle, Image.BILINEAR, self.rotation.expand, self.rotation.center)
        target = F.rotate(target, angle, Image.NEAREST, self.rotation.expand, self.rotation.center)

        i, j, h, w = self.random_resize_crop.get_params(sample, self.random_resize_crop.scale,
                                                        self.random_resize_crop.ratio)
        sample = F.resized_crop(sample, i, j, h, w, self.random_resize_crop.size, self.random_resize_crop.interpolation)
        target = F.resized_crop(target, i, j, h, w, self.random_resize_crop.size, Image.NEAREST)

        sample = self.transform(sample)
        target = self.target_transform(target)

        if self.normalize:
            sample = self.normalize_transform(sample)

        return sample, target

    def rotate_entries(self, sample, target):
        angle = self.rotation.get_params(self.rotation.degrees)
        sample = F.rotate(sample, angle, Image.BILINEAR, self.rotation.expand, self.rotation.center)
        target = F.rotate(target, angle, Image.NEAREST, self.rotation.expand, self.rotation.center)
        return sample, target

    def resize_crop_entries(self, sample, target):
        i, j, h, w = self.random_resize_crop.get_params(sample, self.random_resize_crop.scale,
                                                        self.random_resize_crop.ratio)
        sample = F.resized_crop(sample, i, j, h, w, self.random_resize_crop.size, self.random_resize_crop.interpolation)
        target = F.resized_crop(target, i, j, h, w, self.random_resize_crop.size, Image.NEAREST)
        return sample, target

    def preload(self):
        super().preload()
        if self.normalize:
            self.normalize_transform = torchvision.transforms.Normalize(self.mean, self.std)
        return self

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class VesselDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        self.samples = None
        self.segmentations = None
        self.targets = None
        self.masks = None
        self.classes = None
        self.transform = torchvision.transforms.ToTensor()
        self.mean = np.array([0.3198, 0.1746, 0.0901])
        self.std = np.array([0.2287, 0.1286, 0.0723])
        self.normalize = None

        self.rotation = None
        self.random_resize_crop = None

    def __len__(self):
        return len(self.samples)

    def load_csv(self, filename, root=''):
        df = pd.read_csv(filename, index_col=0)
        self.samples = [str(Path(root) / Path(ind)) for ind in df.index]
        self.segmentations = [os.path.splitext(f)[0] + '_vessel.png' for f in self.samples]
        self.classes = df.columns
        self.targets = df.iloc[:].values.astype(np.float32)
        return self

    def set_transforms(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), angle=180):
        self.random_resize_crop = torchvision.transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)
        self.rotation = torchvision.transforms.RandomRotation(angle)
        self.normalize = torchvision.transforms.Normalize(self.mean, self.std)
        return self

    def __getitem__(self, index):
        if index >= self.__len__():
            raise ValueError('Index {} out of bounds for dataset with length {}'.format(index, self.__len__()))

        sample = Image.open(self.samples[index])
        segment = Image.open(self.segmentations[index])
        target = self.targets[index]

        if self.rotation is not None:
            angle = self.rotation.get_params(self.rotation.degrees)
            sample = F.rotate(sample, angle, Image.BILINEAR, self.rotation.expand, self.rotation.center)
            segment = F.rotate(segment, angle, Image.NEAREST, self.rotation.expand, self.rotation.center)

        if self.random_resize_crop is not None:
            i, j, h, w = self.random_resize_crop.get_params(sample, self.random_resize_crop.scale,
                                                            self.random_resize_crop.ratio)
            sample = F.resized_crop(sample, i, j, h, w, self.random_resize_crop.size, Image.BILINEAR)
            segment = F.resized_crop(segment, i, j, h, w, self.random_resize_crop.size, Image.NEAREST)

        if self.transform is not None:
            sample = self.transform(sample)
            segment = self.transform(segment)

        if self.normalize is not None:
            sample = self.normalize(sample)

        return (sample, segment), target
