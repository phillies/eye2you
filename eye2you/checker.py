import pathlib
import os
import warnings
import json
import io
import configparser

import numpy as np
import torch
import torch.nn as nn
import eye2you.models as models
import torchvision.transforms as transforms  # pylint: disable=unresolved-import

from .datasets import PandasDataset
from .meter_functions import AccuracyMeter, AverageMeter, all_or_nothing_performance


class RetinaChecker():
    """Deep learning model
    """

    def __init__(self, device=None):
        self.device = device
        self.config = None

        self.model = None
        self.model_name = None
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None
        self.criterion = None
        self.criterion_name = None

        self.model_pretrained = False
        self.model_kwargs = {}

        self.train_file = None
        self.train_root = None
        self.test_file = None
        self.test_root = None

        self.train_dataset = None
        self.test_dataset = None
        self.split_indices = None
        self.classes = None
        self.normalize_mean = None
        self.normalize_std = None
        self.image_size = None
        self.test_scale_factor = 1.1

        self.train_loader = None
        self.test_loader = None

        self.start_epoch = 0
        self.epoch = 0

        self.learning_rate = None
        self.learning_rate_decay_gamma = None
        self.learning_rate_decay_step_size = None

        self.initialized = False

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def normalize_factors(self):
        if self.normalize_mean is None or self.normalize_std is None:
            return None
        else:
            return (self.normalize_mean, self.normalize_std)

    @property
    def config_string(self):
        if self.config is None:
            return None
        else:
            with io.StringIO() as fp:
                self.config.write(fp)
                return fp.getvalue()

    def __str__(self):
        desc = 'RetinaChecker\n'
        if self.initialized:
            desc += self._str_core_info()
            desc += self._str_data_info()
        else:
            desc += 'not initialized'
        return desc

    def _str_core_info(self):
        desc = 'Network: ' + self.model_name
        if self.model_pretrained:
            desc += ' (pretrained)\n'
        else:
            desc += '\n'
        desc += 'Optimizer: ' + self.optimizer_name + '\n'
        desc += 'Criterion: ' + self.criterion_name + '\n'
        desc += 'Classes: ' + str(self.classes) + '\n'
        desc += 'Epoch: ' + str(self.epoch) + '\n'
        return desc

    def _str_data_info(self):
        desc = 'Training root: ' + str(self.train_root) + '\n'
        desc += 'Training file: ' + str(self.train_file) + '\n'
        if self.train_dataset is not None:
            desc += str(self.train_dataset) + '\n'
        if self.train_loader is not None:
            desc += 'Batch size: ' + str(self.train_loader.batch_size) + '\n'
            desc += 'Workers: ' + str(self.train_loader.num_workers) + '\n'
        desc += 'Test root: ' + str(self.test_root) + '\n'
        desc += 'Test file: ' + str(self.test_file) + '\n'
        if self.test_dataset is not None:
            desc += str(self.test_dataset) + '\n'
        if self.test_loader is not None:
            desc += 'Batch size: ' + str(self.test_loader.batch_size) + '\n'
            desc += 'Workers: ' + str(self.test_loader.num_workers) + '\n'
        return desc

    def reload(self, checkpoint):
        self.initialize(checkpoint)

        try:
            self.load_datasets()

            # Initializing sampler and data (=patch) loader
            self.create_dataloader()
        except FileNotFoundError:
            print('Could not load data sets. Continue with model.')

        # Initialize the model
        self.initialize_model()
        self.initialize_criterion()
        self.initialize_optimizer()

        self.config['input']['checkpoint'] = checkpoint
        self.load_state()

    def initialize(self, config):
        if config is None:
            raise ValueError('config cannot be None')

        elif isinstance(config, str) or isinstance(config, pathlib.Path):
            if not os.path.isfile(config):
                raise ValueError('File {} does not exist (or is not a file)'.format(config))
            self.config = configparser.ConfigParser()
            try:
                self.config.read(config)
            except (configparser.MissingSectionHeaderError, UnicodeDecodeError):
                try:
                    ckpt = torch.load(config, map_location='cpu')
                    if 'config' in ckpt.keys():
                        self.config.read_string(ckpt['config'])
                        self.classes = ckpt['classes']
                    else:
                        raise ValueError('Checkpoint has no config stored')
                except Exception:
                    raise ValueError('Could not recognize config type')

        elif isinstance(config, configparser.ConfigParser):
            self.config = config
            self.classes = {0: 'undefined'}

        else:
            raise ValueError('Could not recognize config type')

        self._parse_config()

    def _parse_config(self):
        if self.config is None:
            raise ValueError('self.config cannot be None')

        self.model_name = self.config['network'].get('model', 'inception_v3')
        self.optimizer_name = self.config['network'].get('optimizer', 'Adam')
        self.criterion_name = self.config['network'].get('criterion', 'BCEWithLogitsLoss')
        self.train_root = self.config['files'].get('train root', './train')
        self.train_file = self.config['files'].get('train file', 'labels.csv')
        self.test_root = self.config['files'].get('test root', None)
        self.test_file = self.config['files'].get('test file', None)
        normalize_mean = self.config['transform'].get('normalize mean', None)
        if normalize_mean is not None:
            self.normalize_mean = json.loads(normalize_mean)
        normalize_std = self.config['transform'].get('normalize std', None)
        if normalize_std is not None:
            self.normalize_std = json.loads(normalize_std)

        self.model_pretrained = self.config['network'].getboolean('pretrained', False)

        self.learning_rate = self.config['hyperparameter'].getfloat('learning rate', 0.001)
        self.learning_rate_decay_step_size = self.config['hyperparameter'].getfloat('lr decay step', 50)
        self.learning_rate_decay_gamma = self.config['hyperparameter'].getfloat('lr decay gamma', 0.5)

        self.image_size = self.config['files'].getint('image size', 299)

        if self.device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device', self.device)
        self.initialized = True

    def initialize_optimizer(self, **kwargs):
        optimizer_loader = None
        if self.optimizer_name in torch.optim.__dict__.keys():
            optimizer_loader = torch.optim.__dict__[self.optimizer_name]
        else:
            warnings.warn('Could not identify optimizer')
            return

        self.optimizer = optimizer_loader(self.model.parameters(), lr=self.learning_rate, **kwargs)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.learning_rate_decay_step_size, gamma=self.learning_rate_decay_gamma)

    def initialize_criterion(self, **kwargs):
        criterion_loader = None
        if self.criterion_name in nn.__dict__.keys():
            criterion_loader = nn.__dict__[self.criterion_name]
        else:
            warnings.warn('Could not identify criterion')
            return

        self.criterion = criterion_loader(**kwargs)

    def initialize_model(self, **kwargs):
        model_loader = None
        if self.model_name in models.__dict__.keys():
            model_loader = models.__dict__[self.model_name]
        else:
            warnings.warn('Could not identify model')
            return

        self.model = model_loader(pretrained=self.model_pretrained, **kwargs)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        if hasattr(self.model, 'AuxLogits'):
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)

    def load_datasets(self, test_size=0.1):
        '''Loads the data sets from the path given in the config file
        '''

        if self.config['input'].getboolean('evaluation only', False):
            test_transform = self._get_test_transform(self.normalize_factors)
            if self.split_indices is not None:
                dataset = PandasDataset(source=self.test_file, mode='csv', root=self.test_root)
                self.test_dataset = dataset.subset(self.split_indices[1])
                self.test_dataset.transform = test_transform
            else:
                self.test_dataset = PandasDataset(
                    source=self.test_file,
                    mode='csv',
                    root=self.test_root,
                    transform=test_transform,
                    target_transform=None)
        else:
            test_transform = self._get_test_transform(self.normalize_factors)
            train_transform = self._get_training_transform(self.normalize_factors)

            if self.split_indices is not None:
                dataset = PandasDataset(source=self.train_file, mode='csv', root=self.train_root)
                self.train_dataset = dataset.subset(self.split_indices[0])
                self.test_dataset = dataset.subset(self.split_indices[1])
                self.train_dataset.transform = train_transform
                self.test_dataset.transform = test_transform
            elif self.test_file is None or not os.path.isfile(self.test_file):
                dataset = PandasDataset(source=self.train_file, mode='csv', root=self.train_root)
                self.split_indices = []
                self.train_dataset, self.test_dataset = dataset.split(
                    test_size=test_size, return_indices=self.split_indices)
                self.train_dataset.transform = train_transform
                self.test_dataset.transform = test_transform

            else:
                self.train_dataset = PandasDataset(
                    source=self.train_file,
                    mode='csv',
                    root=self.train_root,
                    transform=train_transform,
                    target_transform=None)
                self.test_dataset = PandasDataset(
                    source=self.test_file,
                    mode='csv',
                    root=self.test_root,
                    transform=test_transform,
                    target_transform=None)

        self.classes = self.test_dataset.class_to_idx

    def create_dataloader(self, num_workers=8, sampling_relevance=None):
        """Generates the dataloader (and their respective samplers) for
        training and test data from the training and test data sets.
        Sampler for training data is an unbiased sampler for all classes
        in the training set, i.e. even if the class distribution in the
        data set is biased, all classes are equally contained in the sampling.
        No specific sampler for test data.
        
        Keyword Arguments:
            num_workers {int} -- [description] (default: {8})
            sampling_relevance {[type]} -- [description] (default: {None})
        """

        batch_size = self.config['hyperparameter'].getint('batch size', 32)

        if not self.config['input'].getboolean('evaluation only', False):
            num_samples = self.config['files'].getint('samples', 6400)

            train_sampler = self._get_sampler(self.train_dataset, num_samples, sampling_relevance)

            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=train_sampler,
                num_workers=num_workers)

        test_sampler = None

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers)

    def save_state(self,
                   filename,
                   train_loss=None,
                   train_accuracy=None,
                   test_loss=None,
                   test_accuracy=None,
                   test_confusion=None):
        """Save the current state of the model including history of training and test performance (optionally)
        
        Arguments:
            filename {string} -- target filename        
            train_loss {torch.Array} -- tensor of training losses
            train_accuracy {torch.Array} -- tensor of training accuracy
            test_loss {torch.Array} -- tensor of test losses
            test_accuracy {torch.Array} -- tensor of test accuracy
            test_confusion {torch.Array} -- tensor of confusion matrices
        """

        save_dict = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_confusion': test_confusion,
            'classes': self.classes,
            'scheduler': self.scheduler.state_dict(),
            'description': str(self),
            'config': self.config_string
        }
        if self.train_file is not None:
            save_dict['train_file'] = self.train_file
            save_dict['train_root'] = self.train_root
        if self.test_file is not None:
            save_dict['test_file'] = self.test_file
            save_dict['test_root'] = self.test_root
        if self.split_indices is not None:
            save_dict['train_indices'] = self.split_indices[0]
            save_dict['test_indices'] = self.split_indices[1]

        torch.save(save_dict, filename)

    def load_state(self, filename=None):
        """Load the state stored in the config into the given model and optimizer.
        Model and optimizer must match exactly to the stored model, will crash
        otherwise.
        """
        try:
            if filename is None:
                filename = self.config['input'].get('checkpoint')
            #if torch.cuda.is_available() and self.device.type.startswith('cuda'):
            #    checkpoint = torch.load(filename)
            #else:
            checkpoint = torch.load(filename, map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.epoch = self.start_epoch
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                else:
                    # initial_lr is not set, so we cannot set the last_epoch without creating an error
                    for g in self.optimizer.param_groups:
                        if 'initial_lr' not in g:
                            g['initial_lr'] = g['lr']
                    self.scheduler = torch.optim.lr_scheduler.StepLR(
                        self.optimizer,
                        self.learning_rate_decay_step_size,
                        gamma=self.learning_rate_decay_gamma,
                        last_epoch=self.start_epoch)

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        except OSError as e:
            print("Exception occurred. Did not load model, starting from scratch.\n", e)

    def load_state_data(self):
        """Load the state stored in the config into the given model and optimizer.
        Model and optimizer must match exactly to the stored model, will crash
        otherwise.
        """
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(self.config['input'].get('checkpoint'))
            else:
                checkpoint = torch.load(self.config['input'].get('checkpoint'), map_location='cpu')
            if 'train_file' in checkpoint:
                self.train_file = checkpoint['train_file']
                self.train_root = checkpoint['train_root']
            if 'test_file' in checkpoint:
                self.test_file = checkpoint['test_file']
                self.test_root = checkpoint['test_root']
            if 'train_indices' in checkpoint:
                self.split_indices = [(checkpoint['train_indices'], checkpoint['test_indices'])]
            print("=> loaded data configuration")
        except OSError as e:
            print("Exception occurred. Did not load data\n", e)

    def load_data_split(self):
        """Load the state stored in the config into the given model and optimizer.
        Model and optimizer must match exactly to the stored model, will crash
        otherwise.
        """
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(self.config['input'].get('checkpoint'))
            else:
                checkpoint = torch.load(self.config['input'].get('checkpoint'), map_location='cpu')
            if 'train_indices' in checkpoint:
                self.split_indices = [(checkpoint['train_indices'], checkpoint['test_indices'])]
            else:
                print('No split information found.')

        except OSError as e:
            print("Exception occurred. Did not load data\n", e)

    def _get_training_transform(self, normalize_factors=None):

        rotation_angle = self.config['transform'].getint('rotation angle', 0)
        rotation = transforms.RandomRotation(rotation_angle)

        brightness = self.config['transform'].getfloat('brightness', 0)
        contrast = self.config['transform'].getfloat('contrast', 0)
        saturation = self.config['transform'].getfloat('saturation', 0)
        hue = self.config['transform'].getfloat('hue', 0)
        min_scale = self.config['transform'].getfloat('min scale', 0.25)
        max_scale = self.config['transform'].getfloat('max scale', 1.0)
        color_jitter = transforms.RandomApply(
            [transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)])

        #randcrop = transforms.RandomChoice(
        #    (transforms.RandomResizedCrop(size=self.image_size, scale=(min_scale, max_scale), ratio=(1, 1)),
        #     transforms.RandomResizedCrop(size=self.image_size, scale=(min_scale, 0.4), ratio=(0.8, 1.25))))
        randcrop = transforms.RandomResizedCrop(size=self.image_size, scale=(min_scale, max_scale), ratio=(1, 1))

        transform_list = []
        if brightness != 0 or contrast != 0 or saturation != 0 or hue != 0:
            transform_list.append(color_jitter)

        if rotation_angle != 0:
            transform_list.append(rotation)

        transform_list += [
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            randcrop,
            transforms.ToTensor(),
        ]

        if not normalize_factors is None:
            normalize = transforms.Normalize(mean=normalize_factors[0], std=normalize_factors[1])
            transform_list.append(normalize)

        train_transform = transforms.Compose(transform_list)

        return train_transform

    def _get_test_transform(self, normalize_factors=None):
        # normalization factors for the DMR dataset were manually derived
        transform_list = [
            transforms.Resize(size=int(self.image_size * self.test_scale_factor)),
            transforms.CenterCrop(size=self.image_size),
            transforms.ToTensor(),
        ]

        if not normalize_factors is None:
            normalize = transforms.Normalize(mean=normalize_factors[0], std=normalize_factors[1])
            transform_list.append(normalize)

        test_transform = transforms.Compose(transform_list)

        return test_transform

    def train(self, evaluate_performance=all_or_nothing_performance):
        '''Deep learning training function to optimize the network with all images in the train_loader.
        
        Returns:
            AverageMeter -- training loss
            AccuracyMeter -- training accuracy
        '''
        if not self.initialized:
            print('RetinaChecker not initialized.')
            return

        if self.train_loader is None:
            print('No training loader defined. Check configuration.')
            return

        losses = AverageMeter()
        accuracy = AccuracyMeter()
        self.model.train()
        self.scheduler.step()

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            # inception returns 2 outputs, the regular output and the aux logits output. We need to calculate
            # the loss on both
            outputs = self.model(images)
            if isinstance(outputs, tuple):
                loss = sum((self.criterion(o, labels) for o in outputs))
            else:
                loss = self.criterion(outputs, labels)

            # store results & evaluate accuracy
            losses.update(loss.item(), images.size(0))

            num_correct = evaluate_performance(labels, outputs)
            accuracy.update(num_correct, labels.size(0))

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #current_time = time.time()
            #print('Epoch [{}], Step [{}/{}], Loss: {:.4f},  Samples: {}, Correct: {} ({:.1f}%),  time in epoch {}, epoch remaining {}'
            #    .format(self.epoch + 1, i + 1, len(self.train_loader), loss.item(), labels.size(0), num_correct,
            #        num_correct/labels.size(0)*100,
            #        pretty_print_time(current_time-start_time_epoch),
            #        pretty_time_left(start_time_epoch, i+1, len(self.train_loader))))
        print('Epoch learning completed. Training accuracy {:.1f}%'.format(accuracy.avg * 100))

        self.epoch += 1
        return losses, accuracy

    def validate(self, test_loader=None, evaluate_performance=all_or_nothing_performance, confusion_matrix_label=None):
        '''Evaluates the model given the criterion and the data in test_loader
        
        Arguments:
            test_loader {torch.utils.data.DataLoader} -- contains the data for training, if None takes internal test_loader  
            evaluate_performance {function} -- evaluation function for the accuracy of the output (default: function with correct only if all labels are correct)
            confusion_matrix_label {int} -- on which dimension of the output should the confusion matrix be calculated. Takes 5 or the highest dimension, whichever is smaller {default: None}   
        
        Returns:
            AverageMeter -- training loss
            AccuracyMeter -- training accuracy
            numpy.Array -- [num_classes, num_classes] confusion matrix, columns are true classes, rows predictions
        '''
        if not self.initialized:
            print('RetinaChecker not initialized.')
            return

        if test_loader is None:
            test_loader = self.test_loader

        if confusion_matrix_label is None:
            confusion_matrix_label = min(self.num_classes - 1, 5)

        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            losses = AverageMeter()
            accuracy = AccuracyMeter()

            #confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float)
            confusion = torch.zeros((2, 2), dtype=torch.float)

            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), images.size(0))

                num_correct = evaluate_performance(labels, outputs)
                accuracy.update(num_correct, labels.size(0))

                predicted = torch.nn.Sigmoid()(outputs).round()

                for pred, lab in zip(predicted[:, confusion_matrix_label], labels[:, confusion_matrix_label]):
                    confusion[int(pred.item()), int(lab.item())] += 1

                #print('Test - samples: {}, correct: {} ({:.1f}%), loss: {}'.format(labels.size(0), num_correct, num_correct/labels.size(0)*100, loss.item()))

        return losses, accuracy, confusion

    def _get_sampler(self, dataset, num_samples, relevant_slice=None):
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
            relevant_slice = range(self.num_classes)

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
