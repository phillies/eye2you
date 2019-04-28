import configparser
import io
import json
import os
import pathlib
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tqdm
import pandas as pd

from . import models
from . import datasets
from .dataloader import get_equal_sampler

import copy


class Logger():

    def __init__(self):
        self._log = dict()
        self.columns = None

    def append(self, values, category):
        df = pd.DataFrame(data=[values], columns=self.columns)
        if category not in self._log:
            self._log[category] = pd.DataFrame(columns=self.columns)
        self._log[category] = self._log[category].append(df, ignore_index=True, sort=False)
        self._log[category].index.name = category

    def to_csv(self, filename):
        df = pd.concat(self._log, axis=1)
        df.to_csv(filename, index_label='id')

    def read_csv(self, filename):
        df = pd.read_csv(filename, header=[0, 1], skip_blank_lines=True, index_col=0)
        self._log = dict()
        for cat in df.columns.levels[0]:
            self._log[cat] = df[cat]

    def __str__(self):
        res = ''
        for cat in self._log:
            res += str(self._log[cat]) + '\n'
        return res


class Network():

    def __init__(self,
                 device,
                 model_name,
                 criterion_name,
                 optimizer_name=None,
                 performance_meters=None,
                 model_kwargs=None,
                 criterion_kwargs=None,
                 optimizer_kwargs=None,
                 use_scheduler=False,
                 scheduler_kwargs=None):

        self.device = device

        self.model = None
        self.model_name = model_name
        self.criterion = None
        self.criterion_name = criterion_name
        self.optimizer = None
        self.optimizer_name = optimizer_name
        self.scheduler = None

        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.criterion_kwargs = copy.deepcopy(criterion_kwargs)
        self.optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
        self.scheduler_kwargs = copy.deepcopy(scheduler_kwargs)

        if performance_meters is None:
            performance_meters = []
        self.performance_meters = performance_meters

        self.initialize(model_kwargs=model_kwargs,
                        criterion_kwargs=criterion_kwargs,
                        optimizer_kwargs=optimizer_kwargs,
                        use_scheduler=use_scheduler,
                        scheduler_kwargs=scheduler_kwargs)

    def initialize(self,
                   model_kwargs=None,
                   criterion_kwargs=None,
                   optimizer_kwargs=None,
                   use_scheduler=False,
                   scheduler_kwargs=None):
        if model_kwargs is None:
            model_kwargs = dict()
        if criterion_kwargs is None:
            criterion_kwargs = dict()
        if optimizer_kwargs is None:
            optimizer_kwargs = dict()
        if scheduler_kwargs is None:
            scheduler_kwargs = dict()
        self.initialize_model(**model_kwargs)
        self.initialize_criterion(**criterion_kwargs)
        self.initialize_optimizer(optimizer_kwargs=optimizer_kwargs,
                                  use_scheduler=use_scheduler,
                                  scheduler_kwargs=scheduler_kwargs)
        return self

    def initialize_model(self, pretrained=False, **kwargs):
        model_loader = None
        if self.model_name in models.__dict__.keys():
            model_loader = models.__dict__[self.model_name]
        else:
            warnings.warn('Could not identify model')
            return

        self.model = model_loader(pretrained=pretrained, **kwargs)
        self.model = self.model.to(self.device)

    def initialize_criterion(self, **kwargs):
        if self.criterion_name is None:
            return
        criterion_loader = None

        if self.criterion_name in nn.__dict__.keys():
            criterion_loader = nn.__dict__[self.criterion_name]
        else:
            warnings.warn('Could not identify criterion')
            return

        self.criterion = criterion_loader(**kwargs)

    def initialize_optimizer(self, optimizer_kwargs, use_scheduler, scheduler_kwargs):
        if self.optimizer_name is None:
            return

        optimizer_loader = None
        if self.optimizer_name in torch.optim.__dict__.keys():
            optimizer_loader = torch.optim.__dict__[self.optimizer_name]
        else:
            warnings.warn('Could not identify optimizer')
            return

        self.optimizer = optimizer_loader(self.model.parameters(), **optimizer_kwargs)
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **scheduler_kwargs)

    def train(self, loader):
        if self.optimizer is None:
            raise ValueError('No optimizer defined. Cannot run training.')
        self.model.train()
        if self.scheduler is not None:
            self.scheduler.step()

        total_loss = 0
        num_samples = len(loader.dataset)
        num_batches = len(loader)

        for perf_meter in self.performance_meters:
            perf_meter.reset()

        pbar = tqdm.tqdm(total=num_batches, leave=False, desc='Train')
        for source, target in loader:
            if isinstance(source, (tuple, list)):
                source = [v.to(self.device) for v in source]
            else:
                source = [source.to(self.device)]
            target = target.to(self.device).float()

            outputs = self.model(*source)

            if self.criterion is not None:
                if isinstance(outputs, tuple):
                    #TODO: Check if the division by length of outputs make a notable difference
                    loss = sum((self.criterion(o, target) for o in outputs)) / len(outputs)
                else:
                    loss = self.criterion(outputs, target)
                total_loss += loss.item()
            for perf_meter in self.performance_meters:
                if isinstance(outputs, tuple):
                    perf_meter.update(outputs[0], target)
                else:
                    perf_meter.update(outputs, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.update(1)

        return (total_loss / num_batches, *[p.value() for p in self.performance_meters])

    def validate(self, loader):
        self.model.eval()

        total_loss = 0
        num_samples = len(loader.dataset)
        num_batches = len(loader)

        for perf_meter in self.performance_meters:
            perf_meter.reset()

        with torch.no_grad():
            pbar = tqdm.tqdm(total=num_batches, leave=False, desc='Validate')
            for source, target in loader:
                if isinstance(source, (tuple, list)):
                    source = [v.to(self.device) for v in source]
                else:
                    source = [source.to(self.device)]
                target = target.to(self.device).float()

                output = self.model(*source)

                if self.criterion is not None:
                    loss = self.criterion(output, target)
                    total_loss += loss.item()
                for perf_meter in self.performance_meters:
                    perf_meter.update(output, target)

                pbar.update(1)

        return (total_loss / num_batches, *[p.value() for p in self.performance_meters])

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def get_state_dict(self):
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['model_name'] = self.model_name
        state_dict['model_kwargs'] = self.model_kwargs
        if self.optimizer is not None:
            state_dict['optimizer'] = self.optimizer.state_dict()
            state_dict['optimizer_name'] = self.optimizer_name
            state_dict['optimizer_kwargs'] = self.optimizer_kwargs
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
            state_dict['scheduler_kwargs'] = self.scheduler_kwargs
        if self.criterion is not None:
            state_dict['criterion_name'] = self.criterion_name
            state_dict['criterion_kwargs'] = self.criterion_kwargs
        state_dict['performance_meters'] = [repr(p) for p in self.performance_meters]
        return state_dict

    def name_measures(self):
        names = ['loss']
        names += [str(p) for p in self.performance_meters]
        return names


class Coach():

    def __init__(self):
        self.device = None
        self.net = None
        self.train_data = None
        self.train_loader = None
        self.validate_data = None
        self.validate_loader = None
        self.log = None
        self.training_parameter = None
        self.validation_parameter = None

        self.epochs = 0

    def initialize_network(self,
                           device,
                           model_name,
                           criterion_name,
                           optimizer_name=None,
                           performance_meters=None,
                           model_kwargs=None,
                           criterion_kwargs=None,
                           optimizer_kwargs=None,
                           use_scheduler=False,
                           scheduler_kwargs=None):
        self.device = device
        self.net = Network(device=device,
                           model_name=model_name,
                           criterion_name=criterion_name,
                           optimizer_name=optimizer_name,
                           performance_meters=performance_meters,
                           model_kwargs=model_kwargs,
                           criterion_kwargs=criterion_kwargs,
                           optimizer_kwargs=optimizer_kwargs,
                           use_scheduler=use_scheduler,
                           scheduler_kwargs=scheduler_kwargs)
        self.log = Logger()
        self.log.columns = self.net.name_measures()

    def initialize_training_data(self,
                                 csv,
                                 root,
                                 num_samples=None,
                                 batch_size=2,
                                 num_workers=0,
                                 shuffle=False,
                                 weighted_sampling_classes=None,
                                 size=None,
                                 scale=(0.08, 1.0),
                                 ratio=(3. / 4., 4. / 3.),
                                 angle=None,
                                 brightness=0,
                                 contrast=0,
                                 saturation=0,
                                 hue=0,
                                 transform=None,
                                 target_transform=None,
                                 mean=(0, 0, 0),
                                 std=(1, 1, 1)):

        self.training_parameter = dict()
        self.training_parameter['csv'] = csv
        self.training_parameter['root'] = root
        self.training_parameter['num_samples'] = num_samples
        self.training_parameter['batch_size'] = batch_size
        self.training_parameter['num_workers'] = num_workers
        self.training_parameter['shuffle'] = shuffle
        self.training_parameter['weighted_sampling_classes'] = weighted_sampling_classes
        self.training_parameter['size'] = size
        self.training_parameter['scale'] = scale
        self.training_parameter['ratio'] = ratio
        self.training_parameter['angle'] = angle
        self.training_parameter['brightness'] = brightness
        self.training_parameter['contrast'] = contrast
        self.training_parameter['saturation'] = saturation
        self.training_parameter['hue'] = hue
        self.training_parameter['transform'] = transform
        self.training_parameter['target_transform'] = target_transform
        self.training_parameter['mean'] = mean
        self.training_parameter['std'] = std

        rotation = None
        if angle is not None:
            rotation = transforms.RandomRotation(angle)
        rrc = None
        if size is not None:
            rrc = transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)
        color = None
        if not all((brightness == 0, contrast == 0, saturation == 0, hue == 0)):
            color = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

        self.train_data = datasets.TripleDataset(rotation=rotation,
                                                 random_resize_crop=rrc,
                                                 color_jitter=color,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 mean=mean,
                                                 std=std)
        self.train_data.read_csv(csv, root)

        if weighted_sampling_classes is None:
            sampler = torch.utils.data.RandomSampler(self.train_data, replacement=True, num_samples=num_samples)
        else:
            sampler = get_equal_sampler(self.train_data, num_samples, weighted_sampling_classes)
        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=batch_size,
                                                        drop_last=True,
                                                        num_workers=num_workers,
                                                        sampler=sampler)

    def initialize_validation_data(self,
                                   csv,
                                   root,
                                   batch_size=1,
                                   num_workers=0,
                                   shuffle=False,
                                   transform=None,
                                   target_transform=None,
                                   mean=(0, 0, 0),
                                   std=(1, 1, 1)):

        self.validation_parameter = dict()
        self.training_parameter['csv'] = csv
        self.training_parameter['root'] = root
        self.training_parameter['transform'] = transform
        self.training_parameter['target_transform'] = target_transform
        self.training_parameter['mean'] = mean
        self.training_parameter['std'] = std

        self.validate_data = datasets.TripleDataset(transform=transform,
                                                    target_transform=target_transform,
                                                    mean=mean,
                                                    std=std)
        self.validate_data.read_csv(csv, root)
        self.validate_loader = torch.utils.data.DataLoader(self.validate_data,
                                                           batch_size=batch_size,
                                                           drop_last=False,
                                                           shuffle=shuffle,
                                                           num_workers=num_workers)

    def save(self, filename):
        state_dict = self.net.get_state_dict()
        state_dict['epochs'] = self.epochs
        state_dict['training_parameter'] = self.training_parameter
        state_dict['validation_parameter'] = self.validation_parameter
        state_dict['log'] = self.log
        torch.save(state_dict, filename)

    def load(self, filename, device=None):
        if device is None:
            if self.device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device

        state_dict = torch.load(filename, map_location=device)
        self.epochs = state_dict['epochs']
        self.training_parameter = state_dict['training_parameter']
        self.validation_parameter = state_dict['validation_parameter']
        self.log = state_dict['log']

        self.net.load_state_dict(state_dict)

    def run(self, num_epochs, log_filename=None):
        pbar = tqdm.tqdm(total=num_epochs, desc='Epoch')
        for _ in range(num_epochs):
            train_results = self.net.train(self.train_loader)
            validate_results = self.net.validate(self.validate_loader)

            self.log.append(train_results, 'training')
            self.log.append(validate_results, 'validation')
            if log_filename is not None:
                self.log.to_csv(log_filename)

            self.epochs += 1
            pbar.update(1)
