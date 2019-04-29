import pandas as pd
import torch
import torchvision.transforms as transforms
import tqdm

from . import datasets
from .dataloader import get_equal_sampler
from .net import Network
from sklearn.linear_model import LinearRegression
import numpy as np


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

    def idxmaxmin(self, category):
        idxmax = self._log[category].idxmax(axis='index', skipna=True)
        idxmin = self._log[category].idxmin(axis='index', skipna=True)
        return idxmax, idxmin

    def get_slope(self, category, criterion, window_length):
        if criterion is None:
            #TODO: find better way to select default criterion
            criterion = 'loss'
        series = self._log[category][criterion].iloc[-window_length:]
        slope = LinearRegression().fit(np.array(series.index).reshape(-1, 1),
                                       np.array(series).reshape(-1, 1)).coef_[0, 0]
        return slope


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

    def save_config(self, filename):
        # TODO: store config as yaml ?!?
        pass

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

    def run(self,
            num_epochs,
            log_filename=None,
            checkpoint=None,
            early_stop_window=None,
            early_stop_criterion=None,
            early_stop_slope=0.0):
        #TODO: add error message if model is not set up completely
        pbar = tqdm.tqdm(total=num_epochs, desc='Epoch')
        for _ in range(num_epochs):
            train_results = self.net.train(self.train_loader)
            validate_results = self.net.validate(self.validate_loader)

            self.log.append(train_results, 'training')
            self.log.append(validate_results, 'validation')
            if log_filename is not None:
                self.log.to_csv(log_filename)
            if checkpoint is not None:
                #TODO: select max/min by criterion
                idxmax, idxmin = self.log.idxmax('validation')
                for ii, idx in enumerate(idxmax[1:]):
                    if idx == self.epochs:
                        self.save(checkpoint + '.' + idxmax.index[ii] + '.ckpt')
                if idxmin[0] == self.epochs:
                    self.save(checkpoint + '.' + idxmax.index[0] + '.ckpt')

            if early_stop_window is not None and early_stop_window >= self.epochs:
                validation_slope = self.log.get_slope('validation', early_stop_criterion, early_stop_window)
                if validation_slope < early_stop_slope:
                    print('\n\nEarly stop triggered. Slope {}'.format(validation_slope))
                    return
            self.epochs += 1
            pbar.update(1)

    def validate(self):
        validate_results = self.net.validate(self.validate_loader)
        return validate_results
