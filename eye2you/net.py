import copy
import warnings

import torch.nn as nn
import torch.optim
import tqdm

from . import models


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

    def train(self, loader, position=None):
        if self.optimizer is None:
            raise ValueError('No optimizer defined. Cannot run training.')
        self.model.train()
        if self.scheduler is not None:
            self.scheduler.step()

        total_loss = 0
        num_batches = len(loader)
        num_samples = num_batches * loader.batch_size  #due to drop_last it's not len(loader.dataset)

        for perf_meter in self.performance_meters:
            perf_meter.reset()

        pbar = tqdm.tqdm(total=num_batches, leave=False, desc='Train', position=position)
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
                total_loss += loss.item() * target.shape[0]
            for perf_meter in self.performance_meters:
                if isinstance(outputs, tuple):
                    perf_meter.update(outputs[0], target)
                else:
                    perf_meter.update(outputs, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.update(1)

        return (total_loss / num_samples, *[p.value() for p in self.performance_meters])

    def validate(self, loader, position=None):
        self.model.eval()

        total_loss = 0
        num_samples = len(loader.dataset)
        num_batches = len(loader)

        for perf_meter in self.performance_meters:
            perf_meter.reset()

        with torch.no_grad():
            pbar = tqdm.tqdm(total=num_batches, leave=False, desc='Validate', position=position)
            for source, target in loader:
                if isinstance(source, (tuple, list)):
                    source = [v.to(self.device) for v in source]
                else:
                    source = [source.to(self.device)]
                target = target.to(self.device).float()

                output = self.model(*source)

                if self.criterion is not None:
                    loss = self.criterion(output, target)
                    total_loss += loss.item() * target.shape[0]
                for perf_meter in self.performance_meters:
                    perf_meter.update(output, target)

                pbar.update(1)

        return (total_loss / num_samples, *[p.value() for p in self.performance_meters])

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
