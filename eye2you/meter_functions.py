import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn

import yaml


def _to_float(pred, targ):
    if isinstance(pred, torch.Tensor):
        p = pred.float().detach().cpu()
    else:
        p = torch.Tensor(pred).float()
    if isinstance(targ, torch.Tensor):
        t = targ.float().detach().cpu()
    else:
        t = torch.Tensor(targ).float()
    return p, t


def accuracy_all(predictions, targets):
    pred, targ = _to_float(predictions, targets)
    res = np.apply_along_axis(all, 1, (pred == targ)).astype(np.float)
    res = res.sum() / res.size
    res = np.nan_to_num(res)
    return res


def accuracy_classes(predictions, targets):
    pred, targ = _to_float(predictions, targets)

    res = (pred == targ).sum(0).float()
    res = res / pred.shape[0]
    res = np.nan_to_num(res)
    return res


def sensitivity_classes(predictions, targets):
    pred, targ = _to_float(predictions, targets)

    P = (targ == 1).sum(0).float()
    #N = (targ == 0).sum(0)
    #TN = ((targ == 0) * (pred == 0)).sum(0)
    TP = ((targ == 1) * (pred == 1)).sum(0).float()
    #FP = ((targ == 0) * (pred == 1)).sum(0)
    #SFN = ((targ == 1) * (pred == 0)).sum(0)
    res = TP / P
    res = np.nan_to_num(res)
    return res


def specificity_classes(predictions, targets):
    pred, targ = _to_float(predictions, targets)

    N = (targ == 0).sum(0).float()
    TN = ((targ == 0) * (pred == 0)).sum(0).float()
    res = TN / N
    res = np.nan_to_num(res)
    return res


def precision_classes(predictions, targets):
    pred, targ = _to_float(predictions, targets)

    TP = ((targ == 1) * (pred == 1)).sum(0).float()
    FP = ((targ == 0) * (pred == 1)).sum(0).float()
    res = TP / (TP + FP)
    res = np.nan_to_num(res)
    return res


def f1_score_classes(predictions, targets):
    pred, targ = _to_float(predictions, targets)

    P = (targ == 1).sum(0).float()
    TP = ((targ == 1) * (pred == 1)).sum(0).float()
    FP = ((targ == 0) * (pred == 1)).sum(0).float()
    res = 2 * TP / (TP + FP + P)
    res = np.nan_to_num(res)
    return res


def roc_auc_classes(predictions, targets):
    pred, targ = _to_float(predictions, targets)

    res = np.zeros(pred.shape[1])
    for ii in range(res.size):
        try:
            res[ii] = sklearn.metrics.roc_auc_score(targ[:, ii], pred[:, ii])
        except ValueError:
            res[ii] = 0
    res = np.nan_to_num(res)
    return res


def roc_auc_all(predictions, targets):
    pred, targ = _to_float(predictions, targets)
    try:
        res = sklearn.metrics.roc_auc_score(targ, pred)
    except ValueError:
        res = np.zeros((pred.shape[0]))
    res = np.nan_to_num(res)
    return res


def average_precision_score_classes(predictions, targets):
    pred, targ = _to_float(predictions, targets)

    res = np.zeros(pred.shape[1])
    for ii in range(res.size):
        try:
            res[ii] = sklearn.metrics.average_precision_score(targ[:, ii], pred[:, ii])
        except ValueError:
            res[ii] = 0
    res = np.nan_to_num(res)
    return res


def segmentation_accuracy(predictions, targets):
    '''[summary]

    Arguments:
        predictions {[type]} -- [description]
        targets {[type]} -- [description]

    Raises:
        ValueError -- [description]
        ValueError -- [description]

    Returns:
        [type] -- [description]
    '''

    if not predictions.shape == targets.shape:
        raise ValueError('Shape of targets {0} does not match shape of predictions {1}'.format(
            targets.shape, predictions.shape))

    n, _, _, _ = predictions.shape
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    P = (targets == 1).reshape(n, -1).sum(1)
    N = (targets == 0).reshape(n, -1).sum(1)
    TN = ((targets == 0) * (predictions == 0)).reshape(n, -1).sum(1)
    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    #FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    #FN = ((targets == 1) * (predictions == 0)).reshape(n, -1).sum(1)
    res = (TP + TN) / (P + N)
    res = np.nan_to_num(res)
    return res


def segmentation_iou(predictions, targets):
    '''[summary]

    Arguments:
        predictions {[type]} -- [description]
        targets {[type]} -- [description]
    '''
    if not predictions.shape == targets.shape:
        raise ValueError('Shape of targets {0} does not match shape of predictions {1}'.format(
            targets.shape, predictions.shape))

    n, _, _, _ = predictions.shape
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    P = (targets == 1).reshape(n, -1).sum(1)
    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    res = TP / (P + FP)
    res = np.nan_to_num(res)
    return res


def segmentation_precision(predictions, targets):
    '''[summary]

    Arguments:
        predictions {[type]} -- [description]
        targets {[type]} -- [description]

    Raises:
        ValueError -- [description]
        ValueError -- [description]

    Returns:
        [type] -- [description]
    '''

    if not predictions.shape == targets.shape:
        raise ValueError('Shape of targets {0} does not match shape of predictions {1}'.format(
            targets.shape, predictions.shape))

    n, _, _, _ = predictions.shape
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    res = TP / (TP + FP)
    res = np.nan_to_num(res)
    return res


def segmentation_recall(predictions, targets):
    '''[summary]

    Arguments:
        predictions {[type]} -- [description]
        targets {[type]} -- [description]

    Raises:
        ValueError -- [description]
        ValueError -- [description]

    Returns:
        [type] -- [description]
    '''

    if not predictions.shape == targets.shape:
        raise ValueError('Shape of targets {0} does not match shape of predictions {1}'.format(
            targets.shape, predictions.shape))

    n, _, _, _ = predictions.shape
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    FN = ((targets == 1) * (predictions == 0)).reshape(n, -1).sum(1)
    res = TP / (TP + FN)
    res = np.nan_to_num(res)
    return res


def segmentation_specificity(predictions, targets):
    '''[summary]

    Arguments:
        predictions {[type]} -- [description]
        targets {[type]} -- [description]

    Raises:
        ValueError -- [description]
        ValueError -- [description]

    Returns:
        [type] -- [description]
    '''

    if not predictions.shape == targets.shape:
        raise ValueError('Shape of targets {0} does not match shape of predictions {1}'.format(
            targets.shape, predictions.shape))

    n, _, _, _ = predictions.shape
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    TN = ((targets == 0) * (predictions == 0)).reshape(n, -1).sum(1)
    FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    res = TN / (TN + FP)
    res = np.nan_to_num(res)
    return res


def segmentation_dice(predictions, targets):
    '''[summary]

    Arguments:
        predictions {[type]} -- [description]
        targets {[type]} -- [description]
    '''
    if not predictions.shape == targets.shape:
        raise ValueError('Shape of targets {0} does not match shape of predictions {1}'.format(
            targets.shape, predictions.shape))

    n, _, _, _ = predictions.shape
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    FN = ((targets == 1) * (predictions == 0)).reshape(n, -1).sum(1)
    res = 2 * TP / (2 * TP + FN + FP)
    res = np.nan_to_num(res)
    return res


def segmentation_all(predictions, targets):
    return [
        segmentation_accuracy(predictions, targets),
        segmentation_precision(predictions, targets),
        segmentation_recall(predictions, targets),
        segmentation_specificity(predictions, targets),
        segmentation_iou(predictions, targets),
        segmentation_dice(predictions, targets)
    ]


class PerformanceMeter():

    def update(self, output, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def __repr__(self):
        '''
        This should return a string that, when called with eval(), creates the same PerformanceMeter
        '''
        raise NotImplementedError

    def __str__(self):
        '''
        This will be the function that returns the column name in the logger
        '''
        raise NotImplementedError

    def __eq__(self, comp):
        return repr(self) == repr(comp)


class TotalAccuracyMeter(PerformanceMeter):

    def __init__(self, preprocessing=nn.Sigmoid()):
        self.total = 0
        self.correct = 0
        self.preprocess = preprocessing

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = self.preprocess(output[0]).round()
        else:
            predicted = self.preprocess(output).round()
        res = accuracy_all(predicted, target)
        num_correct = int(res * target.shape[0])
        self.correct += num_correct
        self.total += target.shape[0]
        return self.value()

    def reset(self):
        self.total = 0
        self.correct = 0

    def value(self):
        if self.total == 0:
            return 0
        return self.correct / self.total

    def __repr__(self):
        return 'TotalAccuracyMeter()'

    def __str__(self):
        return 'accuracy'


class SingleAccuracyMeter(PerformanceMeter):

    def __init__(self, index=0, preprocessing=nn.Sigmoid()):
        self.total = 0
        self.correct = 0
        self.index = int(index)
        self.preprocess = preprocessing

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = self.preprocess(output[0]).round()
        else:
            predicted = self.preprocess(output).round()

        res = accuracy_classes(predicted, target)[self.index]
        num_correct = int(res * target.shape[0])
        self.correct += num_correct
        self.total += target.shape[0]
        return self.value()

    def reset(self):
        self.total = 0
        self.correct = 0

    def value(self):
        if self.total == 0:
            return 0
        return self.correct / self.total

    def __repr__(self):
        return 'SingleAccuracyMeter(index={})'.format(self.index)

    def __str__(self):
        return '[{}]accuracy'.format(self.index)


class SingleSensitivityMeter(PerformanceMeter):

    def __init__(self, index=0, preprocessing=nn.Sigmoid()):
        self.total = 0
        self.correct = 0
        self.index = int(index)
        self.preprocess = preprocessing

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = self.preprocess(output[0]).round()
        else:
            predicted = self.preprocess(output).round()

        res = sensitivity_classes(predicted, target)[self.index]
        num_correct = int(res * target.shape[0])
        self.correct += num_correct
        self.total += target.shape[0]
        return self.value()

    def reset(self):
        self.total = 0
        self.correct = 0

    def value(self):
        if self.total == 0:
            return 0
        return self.correct / self.total

    def __repr__(self):
        return 'SingleSensitivityMeter(index={})'.format(self.index)

    def __str__(self):
        return '[{}]sensitivity'.format(self.index)


class SingleSpecificityMeter(PerformanceMeter):

    def __init__(self, index=0, preprocessing=nn.Sigmoid()):
        self.total = 0
        self.correct = 0
        self.index = int(index)
        self.preprocess = preprocessing

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = self.preprocess(output[0]).round()
        else:
            predicted = self.preprocess(output).round()

        res = specificity_classes(predicted, target)[self.index]
        num_correct = int(res * target.shape[0])
        self.correct += num_correct
        self.total += target.shape[0]
        return self.value()

    def reset(self):
        self.total = 0
        self.correct = 0

    def value(self):
        if self.total == 0:
            return 0
        return self.correct / self.total

    def __repr__(self):
        return 'SingleSpecificityMeter(index={})'.format(self.index)

    def __str__(self):
        return '[{}]specificity'.format(self.index)


class SinglePrecisionMeter(PerformanceMeter):

    def __init__(self, index=0, preprocessing=nn.Sigmoid()):
        self.total = 0
        self.correct = 0
        self.index = int(index)
        self.preprocess = preprocessing

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = self.preprocess(output[0]).round()
        else:
            predicted = self.preprocess(output).round()

        res = precision_classes(predicted, target)[self.index]
        num_correct = int(res * target.shape[0])
        self.correct += num_correct
        self.total += target.shape[0]
        return self.value()

    def reset(self):
        self.total = 0
        self.correct = 0

    def value(self):
        if self.total == 0:
            return 0
        return self.correct / self.total

    def __repr__(self):
        return 'SinglePrecisionMeter(index={})'.format(self.index)

    def __str__(self):
        return '[{}]precision'.format(self.index)


class SingleF1Meter(PerformanceMeter):

    def __init__(self, index=0, preprocessing=nn.Sigmoid()):
        self.total = 0
        self.correct = 0
        self.index = int(index)
        self.preprocess = preprocessing

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = self.preprocess(output[0]).round()
        else:
            predicted = self.preprocess(output).round()

        res = f1_score_classes(predicted, target)[self.index]
        num_correct = int(res * target.shape[0])
        self.correct += num_correct
        self.total += target.shape[0]
        return self.value()

    def reset(self):
        self.total = 0
        self.correct = 0

    def value(self):
        if self.total == 0:
            return 0
        return self.correct / self.total

    def __repr__(self):
        return 'SingleF1Meter(index={})'.format(self.index)

    def __str__(self):
        return '[{}]f1'.format(self.index)


class ROCAUCMeter(PerformanceMeter):

    def __init__(self, index=None):
        self.outputs = []
        self.targets = []
        self.index = None if index is None else int(index)

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = (output[0])
        else:
            predicted = output

        self.outputs.append(predicted.detach().cpu())
        self.targets.append(target.detach().cpu())

        return self.value()

    def reset(self):
        self.outputs = []
        self.targets = []

    def value(self):
        if len(self.outputs) == 0:
            return 0

        if self.index is None:
            res = roc_auc_all(np.vstack(self.outputs), np.vstack(self.targets))
        else:
            res = roc_auc_classes(np.vstack(self.outputs), np.vstack(self.targets))[self.index]

        return res

    def __repr__(self):
        return 'ROCAUCMeter(index={})'.format(str(self.index))

    def __str__(self):
        if self.index is None:
            return 'roc_auc'
        else:
            return '[{}]roc_auc'.format(self.index)


class SegmentationAccuracyMeter(PerformanceMeter):

    def __init__(self):
        self.results = []

    def update(self, output, target):
        res = segmentation_accuracy(output.round().detach().cpu().numpy(), target.detach().cpu().numpy())
        self.results.append(res)
        return self.value()

    def value(self):
        if len(self.results) == 0:
            return 0
        val = np.concatenate(self.results).mean()
        return val

    def reset(self):
        self.results = []

    def __repr__(self):
        return 'SegmentationAccuracyMeter()'

    def __str__(self):
        return 'accuracy'


class SegmentationPrecisionMeter(PerformanceMeter):

    def __init__(self):
        self.results = []

    def update(self, output, target):
        res = segmentation_precision(output.round().detach().cpu().numpy(), target.detach().cpu().numpy())
        self.results.append(res)
        return self.value()

    def value(self):
        if len(self.results) == 0:
            return 0
        val = np.concatenate(self.results).mean()
        return val

    def reset(self):
        self.results = []

    def __repr__(self):
        return 'SegmentationPrecisionMeter()'

    def __str__(self):
        return 'precision'


class SegmentationRecallMeter(PerformanceMeter):

    def __init__(self):
        self.results = []

    def update(self, output, target):
        res = segmentation_recall(output.round().detach().cpu().numpy(), target.detach().cpu().numpy())
        self.results.append(res)
        return self.value()

    def value(self):
        if len(self.results) == 0:
            return 0
        val = np.concatenate(self.results).mean()
        return val

    def reset(self):
        self.results = []

    def __repr__(self):
        return 'SegmentationRecallMeter()'

    def __str__(self):
        return 'recall'


class SegmentationSpecificityMeter(PerformanceMeter):

    def __init__(self):
        self.results = []

    def update(self, output, target):
        res = segmentation_specificity(output.round().detach().cpu().numpy(), target.detach().cpu().numpy())
        self.results.append(res)
        return self.value()

    def value(self):
        if len(self.results) == 0:
            return 0
        val = np.concatenate(self.results).mean()
        return val

    def reset(self):
        self.results = []

    def __repr__(self):
        return 'SegmentationSpecificityMeter()'

    def __str__(self):
        return 'specificity'


class SegmentationIOUMeter(PerformanceMeter):

    def __init__(self):
        self.results = []

    def update(self, output, target):
        res = segmentation_iou(output.round().detach().cpu().numpy(), target.detach().cpu().numpy())
        self.results.append(res)
        return self.value()

    def value(self):
        if len(self.results) == 0:
            return 0
        val = np.concatenate(self.results).mean()
        return val

    def reset(self):
        self.results = []

    def __repr__(self):
        return 'SegmentationIOUMeter()'

    def __str__(self):
        return 'iou'


class SegmentationDiceMeter(PerformanceMeter):

    def __init__(self):
        self.results = []

    def update(self, output, target):
        res = segmentation_dice(output.round().detach().cpu().numpy(), target.detach().cpu().numpy())
        self.results.append(res)
        return self.value()

    def value(self):
        if len(self.results) == 0:
            return 0
        val = np.concatenate(self.results).mean()
        return val

    def reset(self):
        self.results = []

    def __repr__(self):
        return 'SegmentationDiceMeter()'

    def __str__(self):
        return 'dice'
