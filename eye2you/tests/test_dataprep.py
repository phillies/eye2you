# pylint: disable=redefined-outer-name
import torch
import torchvision.transforms as transforms
from PIL import Image

from eye2you.datasets import DataPreparation


def test_preparation_initialization():
    prep = DataPreparation()
    assert prep is not None

    prep = DataPreparation(size=100, mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05), crop=90)
    assert prep is not None
    assert prep.mean == (0.5, 0.25, 0.1)
    assert prep.std == (0.2, 0.1, 0.05)
    assert prep.crop == 90
    assert isinstance(prep.convert, transforms.ToTensor)


def test_preparation_print():
    prep = DataPreparation(size=100, mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05), crop=90)
    expected_output = '''Preparation:
Compose(
    Resize(size=100, interpolation=PIL.Image.BILINEAR)
    CenterCrop(size=(90, 90))
    ToTensor()
    Normalize(mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05))
)'''

    output = prep.__str__()
    assert expected_output == output


def test_preparation_transforms():
    prep = DataPreparation(size=100, mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05), crop=90)
    trans = prep.get_transform()
    assert trans.__str__() == prep.transform.__str__()


def test_preparation_transform_correctness(image_filename):
    prep = DataPreparation(size=100, mean=(0.5, 0.25, 0.1), std=(0.2, 0.1, 0.05), crop=90)

    img = Image.open(image_filename)
    assert img is not None
    img_trans = prep.transform(img)
    assert isinstance(img_trans, torch.Tensor)
    assert img_trans.shape == (3, 90, 90)

    #TODO: get better test data
    (sample, mask, segment), target = prep.apply((img, img, img), [img])
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 90, 90)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (3, 90, 90)
    assert isinstance(segment, torch.Tensor)
    assert segment.shape == (3, 90, 90)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (3, 90, 90)

    (sample, mask, segment), target = prep.apply((img, img, img), torch.Tensor([0.0, 1.0, -1.0]))
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 90, 90)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (3, 90, 90)
    assert isinstance(segment, torch.Tensor)
    assert segment.shape == (3, 90, 90)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (3,)
