import torch.nn as nn
import numpy as np

# def single_output_performance(labels, outputs, feature_number=5):
#     if isinstance(outputs, tuple):
#         predicted = nn.Sigmoid()(outputs[0])
#     else:
#         predicted = nn.Sigmoid()(outputs)
#     perf2 = (predicted[:, feature_number].round() == labels[:, feature_number])
#     num_correct = float(perf2.sum())
#     return num_correct

# def all_or_nothing_performance(labels, outputs):
#     if isinstance(outputs, tuple):
#         predicted = nn.Sigmoid()(outputs[0])
#     else:
#         predicted = nn.Sigmoid()(outputs)
#     perf = (predicted.round() == labels).sum(1) == labels.size()[1]
#     num_correct = float(perf.sum())
#     return num_correct


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
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    P = (targets == 1).reshape(n, -1).sum(1)
    N = (targets == 0).reshape(n, -1).sum(1)
    TN = ((targets == 0) * (predictions == 0)).reshape(n, -1).sum(1)
    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    #FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    #FN = ((targets == 1) * (predictions == 0)).reshape(n, -1).sum(1)
    res = (TP + TN) / (P + N)

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
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    P = (targets == 1).reshape(n, -1).sum(1)
    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    res = TP / (P + FP)

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
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    res = TP / (TP + FP)

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
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    FN = ((targets == 1) * (predictions == 0)).reshape(n, -1).sum(1)
    res = TP / (TP + FN)

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
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    TN = ((targets == 0) * (predictions == 0)).reshape(n, -1).sum(1)
    FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    res = TN / (TN + FP)

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
    #if c != 1:
    #    raise ValueError('Only images with 1 channel supported')

    TP = ((targets == 1) * (predictions == 1)).reshape(n, -1).sum(1)
    FP = ((targets == 0) * (predictions == 1)).reshape(n, -1).sum(1)
    FN = ((targets == 1) * (predictions == 0)).reshape(n, -1).sum(1)
    res = 2 * TP / (2 * TP + FN + FP)

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
