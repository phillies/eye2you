import torch.nn as nn


def single_output_performance(labels, outputs, feature_number=5):
    if isinstance(outputs, tuple):
        predicted = nn.Sigmoid()(outputs[0])
    else:
        predicted = nn.Sigmoid()(outputs)
    perf2 = (predicted[:, feature_number].round() == labels[:, feature_number])
    num_correct = float(perf2.sum())
    return num_correct


def all_or_nothing_performance(labels, outputs):
    if isinstance(outputs, tuple):
        predicted = nn.Sigmoid()(outputs[0])
    else:
        predicted = nn.Sigmoid()(outputs)
    perf = (predicted.round() == labels).sum(1) == labels.size()[1]
    num_correct = float(perf.sum())
    return num_correct


class AverageMeter(object):
    """Computes and stores the average and current value
    
    Usage: 
    am = AverageMeter()
    am.update(123)
    am.update(456)
    am.update(789)
    
    last_value = am.val
    average_value = am.avg
    
    am.reset() #set all to 0"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores the correctly classified samples
    
    Usage: pass the number of correct (val) and the total number (n)
    of samples to update(val, n). Then .avg contains the 
    percentage correct.
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
