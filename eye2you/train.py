import pandas as pd
import torch
import torchvision.transforms as transforms
import tqdm
import yaml

from . import datasets, factory
from .net import Network
from sklearn.linear_model import LinearRegression
import numpy as np


class Logger():
    """Logger class for coach results.

    Stores data with column names in Logger.columns for different categories
    (e.g. training and validation).

    """

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
        self.columns = list(df.columns.levels[1])
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

    def get_best(self, category, criterion, min_or_max='max'):
        if min_or_max == 'max':
            idx = self._log[category][criterion].idxmax(axis='index', skipna=True)
        else:
            idx = self._log[category][criterion].idxmin(axis='index', skipna=True)
        return idx, self._log[category][criterion].iloc[idx]


class Coach():

    def __init__(self):
        self.device = None
        self.net = None
        self.train_data = None
        self.train_loader = None
        self.validate_data = None
        self.validate_loader = None
        self.log = None
        self.config = None

        self.epochs = 0

    def load_config(self, config):
        if isinstance(config, dict):
            self.config = config
        else:  # then it should be a filename
            self.config = factory.config_from_yaml(config)

        dataprep = datasets.DataPreparation(**self.config['data_preparation'])
        dataaug = datasets.DataAugmentation(**self.config['data_augmentation'])

        self.train_data, self.validate_data = factory.data_from_config(self.config['dataset'])

        self.train_data.preparation = dataprep
        self.train_data.augmentation = dataaug

        self.validate_data.preparation = dataprep

        self.train_loader = factory.get_loader(self.config['training'], self.train_data)

        self.validate_loader = factory.get_loader(self.config['validation'], self.validate_data)

        self.net = Network(**self.config['net'])
        self.device = self.net.device

        self.log = Logger()
        self.log.columns = self.net.name_measures()

    def save(self, filename):
        state_dict = self.net.get_state_dict()
        state_dict['epochs'] = self.epochs
        state_dict['config'] = self.config
        state_dict['log'] = self.log
        torch.save(state_dict, filename)

    def save_config(self, filename):
        with open(str(filename), 'w') as f:
            yaml.safe_dump(self.config, f)

    def load(self, filename, device=None):
        if device is None:
            if self.device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device
        state_dict = torch.load(filename, map_location=device)
        if self.net is None:
            self.load_config(state_dict['config'])
        self.epochs = state_dict['epochs']
        self.config = state_dict['config']
        self.log = state_dict['log']

        self.net.load_state_dict(state_dict)
        #TODO: load data sets
        print('Data sets not loaded yet.')

    def run(self,
            num_epochs,
            log_filename=None,
            checkpoint=None,
            early_stop_window=None,
            early_stop_criterion='loss',
            early_stop_slope=0.0):
        #TODO: add error message if model is not set up completely
        pbar = tqdm.tqdm(total=num_epochs, desc='Epoch', position=0)
        log_title = tqdm.tqdm(total=0, desc='Criteria', position=2, bar_format='{desc}')
        log_title.set_description_str(
            ('Performance meter: ' + ' {:>6.6}' * (1 + len(self.net.performance_meters))).format(
                'loss', *[p.__str__() for p in self.net.performance_meters]))

        log_train = tqdm.tqdm(total=0, desc='Training results', position=3, bar_format='{desc}')
        log_val = tqdm.tqdm(total=0, desc='Validation results', position=4, bar_format='{desc}')
        log_best = tqdm.tqdm(total=0, desc='Validation best', position=5, bar_format='{desc}')
        log_slope = tqdm.tqdm(total=0, desc='Validation slope', position=6, bar_format='{desc}')
        for _ in range(num_epochs):
            train_results = self.net.train(self.train_loader, position=1)
            validate_results = self.net.validate(self.validate_loader, position=1)

            self.log.append(train_results, 'training')
            self.log.append(validate_results, 'validation')

            best_idx, best_results = self.log.get_best('validation', early_stop_criterion)
            log_train.set_description_str(
                ('Training results:  ' + ' {:.4f}' * len(train_results)).format(*train_results))
            log_val.set_description_str(
                ('Validation results:' + ' {:.4f}' * len(validate_results)).format(*validate_results))
            log_best.set_description_str(
                ('Best after {:7d}:' + ' {:.4f}' * len(best_results)).format(best_idx, *best_results))

            if log_filename is not None:
                self.log.to_csv(log_filename)
            if checkpoint is not None:
                #TODO: select max/min by criterion
                idxmax, idxmin = self.log.idxmaxmin('validation')
                for ii, idx in enumerate(idxmax[1:]):
                    if idx == self.epochs:
                        self.save(checkpoint + '.' + idxmax.index[ii] + '.ckpt')
                if idxmin[0] == self.epochs:
                    self.save(checkpoint + '.' + idxmax.index[0] + '.ckpt')

            if early_stop_window is not None:
                validation_slope = self.log.get_slope('validation', early_stop_criterion, early_stop_window)
                log_slope.set_description_str('Validation slope:  {:.4f}'.format(validation_slope))
                if validation_slope < early_stop_slope and self.epochs >= early_stop_window:
                    pbar.write('Early stop triggered. Slope {}'.format(validation_slope))
                    return
            self.epochs += 1
            pbar.update(1)

    def validate(self):
        validate_results = self.net.validate(self.validate_loader)
        return validate_results
