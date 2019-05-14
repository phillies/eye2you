import random

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

from .helper_functions import pil_loader


class DataAugmentation():

    def __init__(self,
                 angle=None,
                 size=None,
                 scale=None,
                 ratio=None,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 hflip=None,
                 vflip=None):
        self.color_jitter = None
        self.rotation = None
        self.random_resize_crop = None

        if angle is not None:
            self.rotation = transforms.RandomRotation(angle)

        if size is not None:
            if scale is None:
                scale = [1, 1]
            if ratio is None:
                ratio = [1, 1]
            self.random_resize_crop = transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

        if any((brightness, contrast, saturation, hue)):
            if brightness is None:
                brightness = 0
            if contrast is None:
                contrast = 0
            if saturation is None:
                saturation = 0
            if hue is None:
                hue = 0
            self.color_jitter = transforms.ColorJitter(brightness=brightness,
                                                       contrast=contrast,
                                                       saturation=saturation,
                                                       hue=hue)

        self.hflip = hflip
        self.vflip = vflip

    def apply(self, source, target):
        target_is_image = isinstance(target[0], Image.Image)
        sample, mask, segment = source

        if self.color_jitter is not None:
            trans = self.color_jitter.get_params(self.color_jitter.brightness, self.color_jitter.contrast,
                                                 self.color_jitter.saturation, self.color_jitter.hue)
            sample = trans(sample)

        # apply rotation
        if self.rotation is not None:
            angle = self.rotation.get_params(self.rotation.degrees)
            sample = F.rotate(sample, angle, Image.BILINEAR, self.rotation.expand, self.rotation.center)
            mask = F.rotate(mask, angle, Image.NEAREST, self.rotation.expand, self.rotation.center)
            segment = F.rotate(segment, angle, Image.NEAREST, self.rotation.expand, self.rotation.center)
            if target_is_image:
                for ii in range(len(target)):
                    target[ii] = F.rotate(target[ii], angle, Image.NEAREST, self.rotation.expand, self.rotation.center)

        # apply RRC
        if self.random_resize_crop is not None:
            i, j, h, w = self.random_resize_crop.get_params(sample, self.random_resize_crop.scale,
                                                            self.random_resize_crop.ratio)
            sample = F.resized_crop(sample, i, j, h, w, self.random_resize_crop.size, Image.BILINEAR)
            mask = F.resized_crop(mask, i, j, h, w, self.random_resize_crop.size, Image.NEAREST)
            segment = F.resized_crop(segment, i, j, h, w, self.random_resize_crop.size, Image.NEAREST)
            if target_is_image:
                for ii in range(len(target)):
                    target[ii] = F.resized_crop(target[ii], i, j, h, w, self.random_resize_crop.size, Image.NEAREST)

        if self.hflip is not None:
            if random.random() < self.hflip:
                sample = F.hflip(sample)
                mask = F.hflip(mask)
                segment = F.hflip(segment)
                if target_is_image:
                    for ii in range(len(target)):
                        target[ii] = F.hflip(target[ii])

        if self.vflip is not None:
            if random.random() < self.hflip:
                sample = F.vflip(sample)
                mask = F.vflip(mask)
                segment = F.vflip(segment)
                if target_is_image:
                    for ii in range(len(target)):
                        target[ii] = F.vflip(target[ii])

        return (sample, mask, segment), target

    def __str__(self):
        trans = []
        if self.color_jitter is not None:
            trans.append(self.color_jitter)
        if self.rotation is not None:
            trans.append(self.rotation)
        if self.random_resize_crop is not None:
            trans.append(self.random_resize_crop)
        if self.hflip is not None:
            trans.append(transforms.RandomHorizontalFlip(self.hflip))
        if self.vflip is not None:
            trans.append(transforms.RandomVerticalFlip(self.vflip))
        return 'Augmentation:\n' + str(transforms.Compose(trans))


class DataPreparation():

    def __init__(self, size=None, mean=None, std=None, crop=None):
        self.mean = mean
        self.std = std
        self.size = size
        self.crop = crop
        self.convert = transforms.ToTensor()

    def apply(self, source, target):
        sample, mask, segment = source
        target_is_image = isinstance(target[0], Image.Image)

        if self.size is not None:
            sample = F.resize(sample, self.size, interpolation=Image.BILINEAR)
            if mask is not None:
                mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
            if segment is not None:
                segment = F.resize(segment, self.size, interpolation=Image.NEAREST)
            if target_is_image:
                for ii in range(len(target)):
                    target[ii] = F.resize(target[ii], self.size, interpolation=Image.NEAREST)

        if self.crop is not None:
            sample = F.center_crop(sample, self.size)
            if mask is not None:
                mask = F.center_crop(mask, self.size)
            if segment is not None:
                segment = F.center_crop(segment, self.size)
            if target_is_image:
                for ii in range(len(target)):
                    target[ii] = F.center_crop(target[ii], self.size)

        sample = self.convert(sample)
        if mask is not None:
            mask = self.convert(mask)
        if segment is not None:
            segment = self.convert(segment)
        if target_is_image:
            for ii in range(len(target)):
                target[ii] = self.convert(target[ii])
            target = torch.cat(target, dim=0)

        if self.mean is not None and self.std is not None:
            sample = transforms.functional.normalize(sample, self.mean, self.std)

        return (sample, mask, segment), target

    def get_transform(self):
        trans = []
        if self.size is not None:
            trans.append(transforms.Resize(self.size))
        if self.crop is not None:
            trans.append(transforms.CenterCrop(self.crop))
        trans.append(self.convert)
        if self.mean is not None and self.std is not None:
            trans.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(trans)

    @property
    def transform(self):
        return self.get_transform()

    def __str__(self):
        return 'Preparation:\n' + str(self.get_transform())


class TripleDataset(torch.utils.data.Dataset):

    def __init__(self,
                 samples=None,
                 segmentations=None,
                 masks=None,
                 targets=None,
                 target_labels=None,
                 loader=pil_loader,
                 augmentation=None,
                 preparation=None):

        super().__init__()
        self.samples = samples
        self.segmentations = segmentations
        self.masks = masks
        self.targets = targets
        self.target_labels = target_labels

        self.loader = loader

        self.augmentation = augmentation
        self.preparation = preparation

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
            target = self.loader(target)
            if len(target.getbands()) > 1:
                targets = []
                for band in target.getbands():
                    channel = target.getchannel(band)
                    if channel.histogram()[0] == np.prod(channel.size):
                        continue
                    targets.append(channel)
                if len(targets) > 1:
                    bg_mask = np.array(targets[0])
                    for t in targets[1:]:
                        bg_mask = np.clip(bg_mask + np.array(t), 0, 255)
                    bg_mask = 255 - bg_mask
                    bg_mask = Image.fromarray(bg_mask)
                    target = [bg_mask, *targets]
            else:
                target = [target]

        source = (sample, mask, segment)

        if self.augmentation is not None:
            source, target = self.augmentation.apply(source, target)

        if self.preparation is not None:
            source, target = self.preparation.apply(source, target)

        return source, target

    def __str__(self):
        res = f'''Dataset:
        Samples: {len(self.samples)}
        Masks: {len(self.masks) if self.masks is not None else 0}
        Segmentation: {len(self.segmentations) if self.segmentations is not None else 0}
        Targets: {len(self.targets)}, classes {self.targets.shape[1]}
        Target labels: {self.target_labels.values}
        '''
        return res

    @property
    def size(self):
        return len(self)
