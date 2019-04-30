import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
import pandas as pd

from . import datasets
from . import meter_functions as mf
from .net import Network
from .train import Coach
import argparse
import yaml


def coach_from_dict(net_config, train_config, validate_config):
    coach = Coach()
    coach.initialize_training_data(**train_config)
    coach.initialize_validation_data(**validate_config)
    coach.initialize_network(**net_config)

    return coach


def network_from_dict(config):
    net = Network(**config)
    return net


def config_from_yaml(filename):
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # TODO: find better solution than eval()
    if 'performance_meters' in config['net'] and config['net']['performance_meters'] is not None:
        config['net']['performance_meters'] = [eval(m) for m in config['net']['performance_meters']]
    if 'transform' in config['training'] and config['training']['transform'] is not None:
        config['training']['transform'] = eval(config['training']['transform'])
    if 'target_transform' in config['training'] and config['training']['target_transform'] is not None:
        config['training']['target_transform'] = eval(config['training']['target_transform'])
    if 'transform' in config['validation'] and config['validation']['transform'] is not None:
        config['validation']['transform'] = eval(config['validation']['transform'])
    if 'target_transform' in config['training'] and config['validation']['target_transform'] is not None:
        config['validation']['target_transform'] = eval(config['validation']['target_transform'])

    return config


def coach_from_yaml(filename):
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # TODO: find better solution than eval()
    if 'performance_meters' in config['net'] and config['net']['performance_meters'] is not None:
        config['net']['performance_meters'] = [eval(m) for m in config['net']['performance_meters']]
    if 'transform' in config['training'] and config['training']['transform'] is not None:
        config['training']['transform'] = eval(config['training']['transform'])
    if 'target_transform' in config['training'] and config['training']['target_transform'] is not None:
        config['training']['target_transform'] = eval(config['training']['target_transform'])
    if 'transform' in config['validation'] and config['validation']['transform'] is not None:
        config['validation']['transform'] = eval(config['validation']['transform'])
    if 'target_transform' in config['training'] and config['validation']['target_transform'] is not None:
        config['validation']['target_transform'] = eval(config['validation']['target_transform'])

    coach = coach_from_dict(config['net'], config['training'], config['validation'])

    return coach, config['coach']


def coach_from_commandline(argv):
    parser = argparse.ArgumentParser(description='eye2you Coach')

    # Network parameter
    netparser = parser.add_argument_group(title='Network', description='Parameter for the neural network')
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    netparser.add_argument('--device',
                           type=str,
                           help='the pytorch device to run on, e.g. cuda:0 or cpu (default: cuda if available)',
                           default=default_device)
    netparser.add_argument('--model',
                           dest='model_name',
                           type=str,
                           help='Model loader function name, e.g. inception_v3_s',
                           required=True)
    netparser.add_argument('--criterion',
                           dest='criterion_name',
                           type=str,
                           help='Criterion name from torch.nn, e.g. L1Loss',
                           required=True)
    netparser.add_argument('--optimizer',
                           dest='optimizer_name',
                           type=str,
                           help='Optimizer name from torch.optim, e.g. Adam')
    netparser.add_argument('--use_scheduler',
                           help='Use learning rate step scheduler (default: False)',
                           action='store_true',
                           default=False)
    netparser.add_argument('--model_kwargs',
                           type=str,
                           nargs='*',
                           help='Additional arguments passed to the model loader in format "num_classes:int(7)"')
    netparser.add_argument('--criterion_kwargs',
                           type=str,
                           nargs='*',
                           help='Additional arguments passed to the criterion in format "size_average:bool(True)"')
    netparser.add_argument('--optimizer_kwargs',
                           type=str,
                           nargs='*',
                           help='Additional arguments passed to the optimizer in format "lr:float(0.001)"')
    netparser.add_argument('--scheduler_kwargs',
                           type=str,
                           nargs='*',
                           help='Additional arguments passed to the scheduler in format "step_size:int(20)"')
    netparser.add_argument('--performance_meters',
                           type=str,
                           nargs='*',
                           help='Performance meters (such that they can be instantiated by eval)')

    # Training parameter
    trainparser = parser.add_argument_group(title='Training data',
                                            description='Parameter for the training dataset specification')

    trainparser.add_argument('--training-csv', help='Filename of the training data csv file', required=True)
    trainparser.add_argument('--training-root', help='Root directory of the filenames in csv file', required=True)
    trainparser.add_argument('--training-num_samples', help='Number of samples to draw', type=int)
    trainparser.add_argument('--training-batch-size',
                             help='Batch size, must be >1 if batch norm is used in model (default: 2)',
                             default=2,
                             type=int)
    trainparser.add_argument('--training-num_workers',
                             help='Number of parallel workers for data loading (default: 0)',
                             default=0,
                             type=int)
    trainparser.add_argument('--training-shuffle',
                             help='Shuffle training data each epoch',
                             default=False,
                             action='store_true')
    trainparser.add_argument('--training-weighted_sampling_classes',
                             help='Set of outputs to use for weigted sampling, e.g. "range(5)"')
    trainparser.add_argument('--training-size', help='Patch size', nargs='?', type=int)
    trainparser.add_argument('--training-scale',
                             help='Scale for RandomResizedCrop (default [0.08, 1.0])',
                             nargs=2,
                             default=(0.08, 1.0),
                             type=float)
    trainparser.add_argument('--training-ratio',
                             help='Ratio for RandomResizedCrop (default [3/4, 4/3]',
                             nargs=2,
                             default=(3. / 4., 4. / 3.),
                             type=float)
    trainparser.add_argument('--training-angle', help='Angle for RandomRotation', type=int)
    trainparser.add_argument('--training-brightness', help='Brightness for ColorJitter', default=0, type=float)
    trainparser.add_argument('--training-contrast', help='Contrast for ColorJitter', default=0, type=float)
    trainparser.add_argument('--training-saturation', help='Saturation for ColorJitter', default=0, type=float)
    trainparser.add_argument('--training-hue', help='Hue for ColorJitter', default=0, type=float)
    trainparser.add_argument('--training-transform',
                             help='Additional transform for samples, e.g. "transforms.ToTensor()"')
    trainparser.add_argument('--training-target_transform', help='Additional transform for targets')
    trainparser.add_argument('--training-mean',
                             help='Normalization mean (default [0 0 0])',
                             default=(0, 0, 0),
                             nargs=3,
                             type=float)
    trainparser.add_argument('--training-std',
                             help='Normalization std (default [1 1 1])',
                             default=(1, 1, 1),
                             nargs=3,
                             type=float)

    # Validation parameter
    valparser = parser.add_argument_group(title='Validation data',
                                          description='Parameter for the validation dataset specification')
    valparser.add_argument('--validation-csv', help='Filename of the validation data csv file', required=True)
    valparser.add_argument('--validation-root', help='Root directory of the filenames in csv file', required=True)
    valparser.add_argument('--validation-batch-size', help='Batch size (default: 1)', default=1, type=int)
    valparser.add_argument('--validation-num_workers',
                           help='Number of parallel workers for data loading (default: 0)',
                           default=0,
                           type=int)
    valparser.add_argument('--validation-shuffle',
                           help='Shuffle validation data each epoch',
                           default=False,
                           action='store_true')
    trainparser.add_argument('--validation-transform',
                             help='Additional transform for samples, e.g. "transforms.ToTensor()"')
    trainparser.add_argument('--validation-target_transform', help='Additional transform for targets')
    trainparser.add_argument('--validation-mean',
                             help='Normalization mean (default [0 0 0])',
                             default=(0, 0, 0),
                             nargs=3,
                             type=float)
    trainparser.add_argument('--validation-std',
                             help='Normalization std (default [1 1 1])',
                             default=(1, 1, 1),
                             nargs=3,
                             type=float)

    parser.add_argument('--coach_epochs', type=int, required=True)
    parser.add_argument('--coach_logfile')
    parser.add_argument('--coach_checkpoint')

    ns = parser.parse_args(args=argv)
    parse_dict = vars(ns)
    train_dict = dict()
    net_dict = dict()
    validate_dict = dict()
    coach_dict = dict()
    for key in parse_dict:
        if key.startswith('training_'):
            train_dict[key[9:]] = parse_dict[key]
        elif key.startswith('validation_'):
            validate_dict[key[11:]] = parse_dict[key]
        elif key.startswith('coach_'):
            coach_dict[key[6:]] = parse_dict[key]
        else:
            net_dict[key] = parse_dict[key]

    print(net_dict, train_dict, validate_dict)

    require_eval = ('transform', 'target_transform')
    for key in train_dict:
        if key in require_eval:
            train_dict[key] = eval(str(train_dict[key]))
    for key in validate_dict:
        if key in require_eval:
            validate_dict[key] = eval(str(validate_dict[key]))

    coach = coach_from_dict(net_dict, train_dict, validate_dict)

    return coach, coach_dict
