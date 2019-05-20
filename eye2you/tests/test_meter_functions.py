# pylint: disable=redefined-outer-name
import numpy as np
import torch

from eye2you import meter_functions as mf

# TotalAccuracyMeter
# SingleAccuracyMeter
# SegmentationAccuracyMeter
# SegmentationPrecisionMeter
# SegmentationRecallMeter
# SegmentationSpecificityMeter
# SegmentationIOUMeter
# SegmentationDiceMeter


def test_totalaccuracymeter():
    meter = mf.TotalAccuracyMeter()

    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    targets = torch.Tensor((-0, -0, -0, -0, +1, +1, +1, +1, -0, +1)).reshape(10, 1)
    output1 = torch.Tensor((-1, -1, -1, +1, -1, +1, +1, +1, -1, +1)).reshape(10, 1)  # 0.8 correct
    output2 = torch.Tensor((-1, +1, -1, +1, -1, +1, -1, +1, +1, -1)).reshape(10, 1)  # 0.4 correct

    meter.update(output1, targets)
    assert meter.value() == 0.8
    meter.update(output2, targets)
    assert meter.value() == 0.6

    meter.reset()
    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    targets = torch.cat((targets, targets), dim=1)
    output2 = torch.cat((output1, output2), dim=1)  # 0.4 correct
    output1 = torch.cat((output1, output1), dim=1)  # 0.8 correct

    meter.update(output2, targets)
    assert meter.value() == 0.4
    meter.update((output1,), targets)
    assert meter.value() == 0.6

    assert str(meter) == 'accuracy'
    assert repr(meter) == 'TotalAccuracyMeter()'


def test_singleaccuracymeter():
    meter = mf.SingleAccuracyMeter(0)

    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    targets = torch.Tensor((-0, -0, -0, -0, +1, +1, +1, +1, -0, +1)).reshape(10, 1)
    output1 = torch.Tensor((-1, -1, -1, +1, -1, +1, +1, +1, -1, +1)).reshape(10, 1)  # 0.8 correct
    output2 = torch.Tensor((-1, +1, -1, +1, -1, +1, -1, +1, +1, -1)).reshape(10, 1)  # 0.4 correct

    meter.update(output1, targets)
    assert meter.value() == 0.8
    meter.update(output2, targets)
    assert meter.value() == 0.6

    meter.reset()
    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    meter = mf.SingleAccuracyMeter(1)

    targets = torch.cat((targets, targets), dim=1)
    output2 = torch.cat((output1, output2), dim=1)  # 0.4 correct
    output1 = torch.cat((output1, output1), dim=1)  # 0.8 correct

    meter.update(output2, targets)
    assert meter.value() == 0.4
    meter.update((output1,), targets)
    assert meter.value() == 0.6

    assert str(meter) == '[1]accuracy'
    assert repr(meter) == 'SingleAccuracyMeter(index=1)'


def test_singlesensitivitymeter():
    meter = mf.SingleSensitivityMeter(0)

    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    targets = torch.Tensor((-0, -0, -0, -0, +1, +1, +1, +1, -0, +1)).reshape(10, 1)
    output1 = torch.Tensor((-1, -1, -1, +1, -1, +1, +1, +1, -1, +1)).reshape(10, 1)  # 0.8 correct
    output2 = torch.Tensor((-1, +1, -1, +1, -1, +1, -1, +1, +1, -1)).reshape(10, 1)  # 0.4 correct

    meter.update(output1, targets)
    assert meter.value() == 0.8
    meter.update(output2, targets)
    assert meter.value() == 0.6

    meter.reset()
    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    meter = mf.SingleSensitivityMeter(1)

    targets = torch.cat((targets, targets), dim=1)
    output2 = torch.cat((output1, output2), dim=1)  # 0.4 correct
    output1 = torch.cat((output1, output1), dim=1)  # 0.8 correct

    meter.update(output2, targets)
    assert meter.value() == 0.4
    meter.update((output1,), targets)
    assert meter.value() == 0.6

    assert str(meter) == '[1]sensitivity'
    assert repr(meter) == 'SingleSensitivityMeter(index=1)'


def test_singlespecificitymeter():
    meter = mf.SingleSpecificityMeter(0)

    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    targets = torch.Tensor((-0, -0, -0, -0, +1, +1, +1, +1, -0, +1)).reshape(10, 1)
    output1 = torch.Tensor((-1, -1, -1, +1, -1, +1, +1, +1, -1, +1)).reshape(10, 1)  # 0.8 correct
    output2 = torch.Tensor((-1, +1, -1, +1, -1, +1, -1, +1, +1, -1)).reshape(10, 1)  # 0.4 correct

    meter.update(output1, targets)
    assert meter.value() == 0.8
    meter.update(output2, targets)
    assert meter.value() == 0.6

    meter.reset()
    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    meter = mf.SingleSpecificityMeter(1)

    targets = torch.cat((targets, targets), dim=1)
    output2 = torch.cat((output1, output2), dim=1)  # 0.4 correct
    output1 = torch.cat((output1, output1), dim=1)  # 0.8 correct

    meter.update(output2, targets)
    assert meter.value() == 0.4
    meter.update((output1,), targets)
    assert meter.value() == 0.6

    assert str(meter) == '[1]specificity'
    assert repr(meter) == 'SingleSpecificityMeter(index=1)'


def test_singleprecisionmeter():
    meter = mf.SinglePrecisionMeter(0)

    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    targets = torch.Tensor((-0, -0, -0, -0, +1, +1, +1, +1, -0, +1)).reshape(10, 1)
    output1 = torch.Tensor((-1, -1, -1, +1, -1, +1, +1, +1, -1, +1)).reshape(10, 1)  # 0.8 correct
    output2 = torch.Tensor((-1, +1, -1, +1, -1, +1, -1, +1, +1, -1)).reshape(10, 1)  # 0.4 correct

    meter.update(output1, targets)
    assert meter.value() == 0.8
    meter.update(output2, targets)
    assert meter.value() == 0.6

    meter.reset()
    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    meter = mf.SinglePrecisionMeter(1)

    targets = torch.cat((targets, targets), dim=1)
    output2 = torch.cat((output1, output2), dim=1)  # 0.4 correct
    output1 = torch.cat((output1, output1), dim=1)  # 0.8 correct

    meter.update(output2, targets)
    assert meter.value() == 0.4
    meter.update((output1,), targets)
    assert meter.value() == 0.6

    assert str(meter) == '[1]precision'
    assert repr(meter) == 'SinglePrecisionMeter(index=1)'


def test_singlef1nmeter():
    meter = mf.SingleF1Meter(0)

    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    targets = torch.Tensor((-0, -0, -0, -0, +1, +1, +1, +1, -0, +1)).reshape(10, 1)
    output1 = torch.Tensor((-1, -1, -1, +1, -1, +1, +1, +1, -1, +1)).reshape(10, 1)  # 0.8 correct
    output2 = torch.Tensor((-1, +1, -1, +1, -1, +1, -1, +1, +1, -1)).reshape(10, 1)  # 0.4 correct

    meter.update(output1, targets)
    assert meter.value() == 0.8
    meter.update(output2, targets)
    assert meter.value() == 0.6

    meter.reset()
    assert meter.total == 0
    assert meter.correct == 0
    assert meter.value() == 0

    meter = mf.SingleF1Meter(1)

    targets = torch.cat((targets, targets), dim=1)
    output2 = torch.cat((output1, output2), dim=1)  # 0.4 correct
    output1 = torch.cat((output1, output1), dim=1)  # 0.8 correct

    meter.update(output2, targets)
    assert meter.value() == 0.4
    meter.update((output1,), targets)
    assert meter.value() == 0.6

    assert str(meter) == '[1]f1'
    assert repr(meter) == 'SingleF1Meter(index=1)'


def test_rocaucmeter():
    meter = mf.ROCAUCMeter(0)

    assert len(meter.outputs) == 0
    assert len(meter.targets) == 0
    assert meter.value() == 0

    targets = torch.Tensor((-0, -0, -0, -0, +1, +1, +1, +1, -0, +1)).reshape(10, 1)
    output1 = torch.Tensor((-1, -1, -1, +1, -1, +1, +1, +1, -1, +1)).reshape(10, 1)  # 0.8 correct
    output2 = torch.Tensor((-1, +1, -1, +1, -1, +1, -1, +1, +1, -1)).reshape(10, 1)  # 0.4 correct

    meter.update(output1, targets)
    np.testing.assert_allclose(meter.value(), 0.8)
    meter.update(output2, targets)
    np.testing.assert_allclose(meter.value(), 0.6)

    meter.reset()
    assert len(meter.outputs) == 0
    assert len(meter.targets) == 0
    assert meter.value() == 0

    meter = mf.ROCAUCMeter(1)

    targets = torch.cat((targets, targets), dim=1)
    output2 = torch.cat((output1, output2), dim=1)  # 0.4 correct
    output1 = torch.cat((output1, output1), dim=1)  # 0.8 correct

    meter.update(output2, targets)
    np.testing.assert_allclose(meter.value(), 0.4)
    meter.update((output1,), targets)
    np.testing.assert_allclose(meter.value(), 0.6)

    assert str(meter) == 'roc_auc 1'
    assert repr(meter) == 'ROCAUCMeter(index=1)'

    meter = mf.ROCAUCMeter()

    targets = torch.cat((targets, targets), dim=1)
    output2 = torch.cat((output1, output2), dim=1)  # 0.4 correct
    output1 = torch.cat((output1, output1), dim=1)  # 0.8 correct

    meter.update(output2, targets)
    np.testing.assert_allclose(meter.value(), 0.7)
    meter.update((output1,), targets)
    np.testing.assert_allclose(meter.value(), 0.75)


def test_segmentationaccuracymeter(segmentation_examples):
    targets, output1, output2 = segmentation_examples
    pred1 = output1.round()
    pred2 = output2.round()
    meter = mf.SegmentationAccuracyMeter()

    assert meter.value() == 0

    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 0.85)

    meter.reset()
    assert meter.value() == 0

    meter.update(pred2, targets)
    np.testing.assert_allclose(meter.value(), 0.72)
    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 0.785)


def test_segmentationprecisionmeter(segmentation_examples):
    targets, output1, output2 = segmentation_examples
    pred1 = output1.round()
    pred2 = output2.round()
    meter = mf.SegmentationPrecisionMeter()

    assert meter.value() == 0

    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 0.6)

    meter.reset()
    assert meter.value() == 0

    meter.update(pred2, targets)
    np.testing.assert_allclose(meter.value(), 1 / 3)
    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 14 / 30)


def test_segmentationrecallmeter(segmentation_examples):
    targets, output1, output2 = segmentation_examples
    pred1 = output1.round()
    pred2 = output2.round()
    meter = mf.SegmentationRecallMeter()

    assert meter.value() == 0

    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 0.75)

    meter.reset()
    assert meter.value() == 0

    meter.update(pred2, targets)
    np.testing.assert_allclose(meter.value(), 0.4)
    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 0.575)


def test_segmentationspecificitymeter(segmentation_examples):
    targets, output1, output2 = segmentation_examples
    pred1 = output1.round()
    pred2 = output2.round()
    meter = mf.SegmentationSpecificityMeter()

    assert meter.value() == 0

    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 0.875)

    meter.reset()
    assert meter.value() == 0

    meter.update(pred2, targets)
    np.testing.assert_allclose(meter.value(), 0.8)
    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 0.8375)


def test_segmentationioumeter(segmentation_examples):
    targets, output1, output2 = segmentation_examples
    pred1 = output1.round()
    pred2 = output2.round()
    meter = mf.SegmentationIOUMeter()

    assert meter.value() == 0

    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 0.5)

    meter.reset()
    assert meter.value() == 0

    meter.update(pred2, targets)
    np.testing.assert_allclose(meter.value(), 2 / 9)
    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 13 / 36)


# 5x5 patch, 0.85 accuracy, 0.6 precision, 0.75 recall, 0.875 specificity, 0.5 iou, 2/3 dice
# 4x6 patch, 0.72 accuracy, 1/3 precision, 0.4 recall, 0.8 specificity, 2/9 iou, 16/44 dice


def test_segmentationdicemeter(segmentation_examples):
    targets, output1, output2 = segmentation_examples
    pred1 = output1.round()
    pred2 = output2.round()
    meter = mf.SegmentationDiceMeter()

    assert meter.value() == 0

    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 2 / 3)

    meter.reset()
    assert meter.value() == 0

    meter.update(pred2, targets)
    np.testing.assert_allclose(meter.value(), 16 / 44)
    meter.update(pred1, targets)
    np.testing.assert_allclose(meter.value(), 68 / 132)


def test_segmentation_all(segmentation_examples):
    targets, output1, _ = segmentation_examples
    pred1 = output1.round()
    _ = mf.segmentation_all(pred1, targets)
