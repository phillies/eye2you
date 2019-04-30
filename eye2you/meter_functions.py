import torch.nn as nn
import numpy as np


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
    print('#######in#######')
    print(outputs)
    print(predicted.round())
    print(labels)
    print('########out######')
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


def measure_iou(output, label):
    ious = []
    for ii in range(output.shape[0]):
        out = (output[ii, ...] > 0.5).detach().cpu().numpy()
        lab = (label[ii, ...] > 0.5).detach().cpu().numpy()
        union = out | lab
        intersect = out & lab
        iou = intersect.sum() / union.sum()
        if union.sum() == 0:
            iou = 0
        ious.append(iou)
    return ious


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

    n, c, h, w = predictions.shape
    if c != 1:
        raise ValueError('Only images with 1 channel supported')

    res = np.empty(n)
    for ii in range(n):
        t = targets[ii, 0, :, :]
        p = predictions[ii, 0, :, :]
        correct = (t == p)
        res[ii] = float(correct.sum()) / float(h * w)

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

    n, c, h, w = predictions.shape
    if c != 1:
        raise ValueError('Only images with 1 channel supported')

    res = np.empty(n)
    for ii in range(n):
        t = targets[ii, 0, :, :]
        p = predictions[ii, 0, :, :]
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        res[ii] = float(intersection.sum()) / float(union.sum())

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

    n, c, h, w = predictions.shape
    if c != 1:
        raise ValueError('Only images with 1 channel supported')

    res = np.empty(n)
    for ii in range(n):
        t = targets[ii, 0, :, :]
        p = predictions[ii, 0, :, :]
        correct = np.logical_and(t, p)
        correct_plus_false_positive = np.logical_or(correct, p)
        res[ii] = float(correct.sum()) / float(correct_plus_false_positive.sum())

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

    n, c, h, w = predictions.shape
    if c != 1:
        raise ValueError('Only images with 1 channel supported')

    res = np.empty(n)
    for ii in range(n):
        t = targets[ii, 0, :, :]
        p = predictions[ii, 0, :, :]
        correct = np.logical_and(t, p)
        res[ii] = float(correct.sum()) / float(t.sum())

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

    n, c, h, w = predictions.shape
    if c != 1:
        raise ValueError('Only images with 1 channel supported')

    res = np.empty(n)
    for ii in range(n):
        t = targets[ii, 0, :, :]
        p = predictions[ii, 0, :, :]
        correct = np.logical_and(1 - t, 1 - p)
        res[ii] = float(correct.sum()) / float((1 - t).sum())

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

    n, c, h, w = predictions.shape
    if c != 1:
        raise ValueError('Only images with 1 channel supported')

    res = np.empty(n)
    for ii in range(n):
        t = targets[ii, 0, :, :]
        p = predictions[ii, 0, :, :]
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        res[ii] = float(2 * intersection.sum()) / float(union.sum() + intersection.sum())

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


class TotalAccuracyMeter(PerformanceMeter):

    def __init__(self):
        self.total = 0
        self.correct = 0

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = nn.Sigmoid()(output[0])
        else:
            predicted = nn.Sigmoid()(output)
        perf = (predicted.round() == target).sum(1) == target.shape[1]
        num_correct = float(perf.sum())
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

    def __init__(self, index=0):
        self.total = 0
        self.correct = 0
        self.index = index

    def update(self, output, target):
        if isinstance(output, tuple):
            predicted = nn.Sigmoid()(output[0])
        else:
            predicted = nn.Sigmoid()(output)
        perf = (predicted[:, self.index].round() == target[:, self.index])
        num_correct = float(perf.sum())
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
        return 'accuracy_class_{}'.format(self.index)


class SegmentationMeter(PerformanceMeter):

    def __init__(self):
        self.results = []

    def update(self, output, target):
        res = segmentation_all(output, target)
        self.results.append(res)
        return self.value()

    def value(self):
        val = np.concatenate(self.results, axis=1).mean(1)
        return val

    def reset(self):
        self.results = []

    def __repr__(self):
        return 'SegmentationMeter()'

    def __str__(self):
        return ('accuracy', 'precicion', 'recall', 'specificity', 'iou', 'dice')


class SegmentationAccuracyMeter(PerformanceMeter):

    def __init__(self):
        self.results = []

    def update(self, output, target):
        res = segmentation_accuracy(output, target)
        self.results.append(res)
        return self.value()

    def value(self):
        val = np.array(self.results).mean()
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
        res = segmentation_recall(output, target)
        self.results.append(res)
        return self.value()

    def value(self):
        val = np.array(self.results).mean()
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
        res = segmentation_recall(output, target)
        self.results.append(res)
        return self.value()

    def value(self):
        val = np.array(self.results).mean()
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
        res = segmentation_specificity(output, target)
        self.results.append(res)
        return self.value()

    def value(self):
        val = np.array(self.results).mean()
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
        res = segmentation_iou(output, target)
        self.results.append(res)
        return self.value()

    def value(self):
        val = np.array(self.results).mean()
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
        res = segmentation_dice(output, target)
        self.results.append(res)
        return self.value()

    def value(self):
        val = np.array(self.results).mean()
        return val

    def reset(self):
        self.results = []

    def __repr__(self):
        return 'SegmentationDiceMeter()'

    def __str__(self):
        return 'dice'
