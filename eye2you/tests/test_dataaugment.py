# pylint: disable=redefined-outer-name
import torch
from PIL import Image

from eye2you.datasets import DataAugmentation


def test_augmentation_initialization():

    aug = DataAugmentation()
    assert aug is not None

    aug = DataAugmentation(angle=23.5,
                           size=100,
                           scale=(0.5, 1.45),
                           ratio=(0.75, 1.3),
                           brightness=0.5,
                           contrast=0.6,
                           saturation=0.4,
                           hue=0.1,
                           hflip=0.25,
                           vflip=0.33)
    assert aug is not None
    assert aug.rotation is not None
    assert aug.rotation.degrees == (-23.5, 23.5)
    assert aug.color_jitter is not None
    assert aug.color_jitter.brightness == [0.5, 1.5]
    assert aug.color_jitter.contrast == [0.4, 1.6]
    assert aug.color_jitter.hue == [-0.1, 0.1]
    assert aug.color_jitter.saturation == [0.6, 1.4]
    assert aug.random_resize_crop is not None
    assert aug.random_resize_crop.size == (100, 100)
    assert aug.random_resize_crop.scale == (0.5, 1.45)
    assert aug.random_resize_crop.ratio == (0.75, 1.3)
    assert aug.hflip == 0.25
    assert aug.vflip == 0.33

    aug = DataAugmentation(angle=23.5, size=100, brightness=0.5, hflip=0.25, vflip=0.33)
    assert aug is not None
    assert aug.rotation is not None
    assert aug.rotation.degrees == (-23.5, 23.5)
    assert aug.color_jitter is not None
    assert aug.color_jitter.brightness == [0.5, 1.5]
    assert aug.color_jitter.contrast is None
    assert aug.color_jitter.hue is None
    assert aug.color_jitter.saturation is None
    assert aug.random_resize_crop is not None
    assert aug.random_resize_crop.size == (100, 100)
    assert aug.random_resize_crop.scale == (1, 1)
    assert aug.random_resize_crop.ratio == (1, 1)
    assert aug.hflip == 0.25
    assert aug.vflip == 0.33

    aug = DataAugmentation(angle=23.5, size=100, saturation=0.5, hflip=0.25, vflip=0.33)
    assert aug is not None
    assert aug.color_jitter is not None
    assert aug.color_jitter.saturation == [0.5, 1.5]
    assert aug.color_jitter.contrast is None
    assert aug.color_jitter.hue is None
    assert aug.color_jitter.brightness is None


def test_augmentation_print():
    aug = DataAugmentation(angle=23.5,
                           size=100,
                           scale=(0.5, 1.45),
                           ratio=(0.75, 1.3),
                           brightness=0.5,
                           contrast=0.6,
                           saturation=0.4,
                           hue=0.1,
                           hflip=0.25,
                           vflip=0.33)

    expected_output = '''Augmentation:
Compose(
    ColorJitter(brightness=[0.5, 1.5], contrast=[0.4, 1.6], saturation=[0.6, 1.4], hue=[-0.1, 0.1])
    RandomRotation(degrees=(-23.5, 23.5), resample=False, expand=False)
    RandomResizedCrop(size=(100, 100), scale=(0.5, 1.45), ratio=(0.75, 1.3), interpolation=PIL.Image.BILINEAR)
    RandomHorizontalFlip(p=0.25)
    RandomVerticalFlip(p=0.33)
)'''

    output = aug.__str__()
    assert expected_output == output


def test_augmentations_transforms():
    aug = DataAugmentation(angle=23.5,
                           size=100,
                           scale=(0.5, 1.45),
                           ratio=(0.75, 1.3),
                           brightness=0.5,
                           contrast=0.6,
                           saturation=0.4,
                           hue=0.1,
                           hflip=0.25,
                           vflip=0.33)
    trans = aug.get_transform()
    assert trans.__str__() == aug.transform.__str__()


def test_augmentation_transform_correctness(image_filename):
    aug = DataAugmentation(angle=23.5,
                           size=100,
                           scale=(0.5, 1.45),
                           ratio=(0.75, 1.3),
                           brightness=0.5,
                           contrast=0.6,
                           saturation=0.4,
                           hue=0.1,
                           hflip=1.0,
                           vflip=1.0)

    img = Image.open(image_filename)
    assert img is not None
    img_trans = aug.transform(img)
    assert isinstance(img_trans, Image.Image)
    assert img_trans.size == (100, 100)
    #TODO: more content checking after transformation

    #TODO: get better test data
    (sample, mask, segment), target = aug.apply((img, img, img), [img])
    assert isinstance(sample, Image.Image)
    assert sample.size == (100, 100)
    assert isinstance(mask, Image.Image)
    assert mask.size == (100, 100)
    assert isinstance(segment, Image.Image)
    assert segment.size == (100, 100)
    assert isinstance(target, (list, tuple))
    assert len(target) == 1
    assert isinstance(target[0], Image.Image)
    assert target[0].size == (100, 100)

    (sample, mask, segment), target = aug.apply((img, img, img), torch.Tensor([0.0, 1.0, -1.0]))
    assert isinstance(sample, Image.Image)
    assert sample.size == (100, 100)
    assert isinstance(mask, Image.Image)
    assert mask.size == (100, 100)
    assert isinstance(segment, Image.Image)
    assert segment.size == (100, 100)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (3,)
